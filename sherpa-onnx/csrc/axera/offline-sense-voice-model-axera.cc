// sherpa-onnx/csrc/axera/offline-sense-voice-model-axera.cc
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#include "sherpa-onnx/csrc/axera/offline-sense-voice-model-axera.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "ax_engine_api.h"  // NOLINT
#include "ax_sys_api.h"     // NOLINT
#include "sherpa-onnx/csrc/axera/ax-engine-guard.h"
#include "sherpa-onnx/csrc/axera/utils.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

class OfflineSenseVoiceModelAxera::Impl {
 public:
  ~Impl() {
    FreeIO(&io_data_);
    if (handle_) {
      AX_ENGINE_DestroyHandle(handle_);
    }
  }

  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    auto buf = ReadFile(config_.sense_voice.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    auto buf = ReadFile(mgr, config_.sense_voice.model);
    Init(buf.data(), buf.size());
  }

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

  std::vector<float> Run(std::vector<float> features, int32_t language,
                         int32_t text_norm) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Apply LFR (model does CMVN internally)
    features = ApplyLFR(std::move(features));

    int32_t num_frames = features.size() / feat_dim_;
    if (num_frames > num_input_frames_) {
      SHERPA_ONNX_LOGE(
          "Input has %d frames but model supports %d. Truncating.",
          num_frames, num_input_frames_);
      num_frames = num_input_frames_;
    }

    // Pad features to num_input_frames_
    std::vector<float> padded(num_input_frames_ * feat_dim_, 0.0f);
    std::copy(features.data(), features.data() + num_frames * feat_dim_,
              padded.data());

    // Create attention mask: shape [1, 1, num_input_frames_ + 4]
    std::vector<float> mask((num_input_frames_ + 4), 0.0f);
    for (int32_t i = 0; i < num_frames + 4; ++i) {
      mask[i] = 1.0f;
    }

    // Language token
    std::vector<int32_t> lang_vec = {language};

    // Copy inputs
    const auto &in0 = io_info_->pInputs[0];
    const auto &in1 = io_info_->pInputs[1];
    const auto &in2 = io_info_->pInputs[2];

    std::memcpy(io_data_.pInputs[0].pVirAddr, padded.data(), in0.nSize);

    // mask is [1, 1, T], stored flat
    size_t mask_bytes = mask.size() * sizeof(float);
    if (mask_bytes != in1.nSize) {
      SHERPA_ONNX_LOGE("Mask size mismatch: expected %u, got %zu", in1.nSize,
                       mask_bytes);
      SHERPA_ONNX_EXIT(-1);
    }
    std::memcpy(io_data_.pInputs[1].pVirAddr, mask.data(), mask_bytes);

    size_t lang_bytes = lang_vec.size() * sizeof(int32_t);
    if (lang_bytes != in2.nSize) {
      SHERPA_ONNX_LOGE("Language size mismatch: expected %u, got %zu",
                       in2.nSize, lang_bytes);
      SHERPA_ONNX_EXIT(-1);
    }
    std::memcpy(io_data_.pInputs[2].pVirAddr, lang_vec.data(), lang_bytes);

    auto ret = AX_ENGINE_RunSync(handle_, &io_data_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync failed, ret = %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    // Read ctc_logits output
    const auto &out0 = io_info_->pOutputs[0];
    auto &out_buf0 = io_data_.pOutputs[0];
    size_t out0_elems = out0.nSize / sizeof(float);
    std::vector<float> out(out0_elems);
    std::memcpy(out.data(), out_buf0.pVirAddr, out0.nSize);

    return out;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    // SenseVoice metadata
    meta_data_.window_size = 7;
    meta_data_.window_shift = 6;
    meta_data_.vocab_size = 25055;
    meta_data_.normalize_samples = 0;
    meta_data_.blank_id = 0;
    meta_data_.lang2id = {{"auto", 0}, {"zh", 3},   {"en", 4},
                          {"yue", 7},  {"ja", 11},   {"ko", 12}};
    meta_data_.with_itn_id = 14;
    meta_data_.without_itn_id = 15;
    meta_data_.is_funasr_nano = false;

    InitContext(model_data, model_data_length, config_.debug, &handle_);
    InitInputOutputAttrs(handle_, config_.debug, &io_info_);
    PrepareIO(io_info_, &io_data_, config_.debug);

    if (!io_info_ || io_info_->nInputSize != 3 || !io_info_->pInputs) {
      SHERPA_ONNX_LOGE("SenseVoice axera model expects 3 input tensors, got %u",
                       io_info_ ? io_info_->nInputSize : 0);
      SHERPA_ONNX_EXIT(-1);
    }

    // speech input: [1, num_frames, 560]
    auto &in0 = io_info_->pInputs[0];
    if (in0.nShapeSize < 3) {
      SHERPA_ONNX_LOGE("Speech input rank too small (nShapeSize = %u)",
                       in0.nShapeSize);
      SHERPA_ONNX_EXIT(-1);
    }
    num_input_frames_ = in0.pShape[1];
    feat_dim_ = in0.pShape[2];

    if (io_info_->nOutputSize < 1) {
      SHERPA_ONNX_LOGE("SenseVoice axera model expects at least 1 output");
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Axera SenseVoice model init done.");
      SHERPA_ONNX_LOGE("  num_input_frames = %d, feat_dim = %d",
                       num_input_frames_, feat_dim_);
    }
  }

  std::vector<float> ApplyLFR(std::vector<float> in) const {
    int32_t lfr_window_size = meta_data_.window_size;
    int32_t lfr_window_shift = meta_data_.window_shift;
    int32_t in_feat_dim = 80;
    int32_t in_num_frames = in.size() / in_feat_dim;
    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;
    int32_t out_feat_dim = in_feat_dim * lfr_window_size;

    std::vector<float> out(out_num_frames * out_feat_dim);
    const float *p_in = in.data();
    float *p_out = out.data();

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);
      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }

    return out;
  }

  std::mutex mutex_;
  AxEngineGuard ax_engine_guard_;

  OfflineModelConfig config_;
  AX_ENGINE_HANDLE handle_ = nullptr;
  AX_ENGINE_IO_INFO_T *io_info_ = nullptr;
  AX_ENGINE_IO_T io_data_;
  OfflineSenseVoiceModelMetaData meta_data_;
  int32_t num_input_frames_ = -1;
  int32_t feat_dim_ = 0;
};

OfflineSenseVoiceModelAxera::~OfflineSenseVoiceModelAxera() = default;

OfflineSenseVoiceModelAxera::OfflineSenseVoiceModelAxera(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSenseVoiceModelAxera::OfflineSenseVoiceModelAxera(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::vector<float> OfflineSenseVoiceModelAxera::Run(
    std::vector<float> features, int32_t language, int32_t text_norm) const {
  return impl_->Run(std::move(features), language, text_norm);
}

const OfflineSenseVoiceModelMetaData &
OfflineSenseVoiceModelAxera::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

#if __ANDROID_API__ >= 9
template OfflineSenseVoiceModelAxera::OfflineSenseVoiceModelAxera(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineSenseVoiceModelAxera::OfflineSenseVoiceModelAxera(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
