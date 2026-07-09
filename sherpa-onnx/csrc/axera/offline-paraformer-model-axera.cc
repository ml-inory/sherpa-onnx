// sherpa-onnx/csrc/axera/offline-paraformer-model-axera.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axera/offline-paraformer-model-axera.h"

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
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineParaformerModelAxera::Impl {
 public:
  ~Impl() {
    FreeIO(&encoder_io_);
    if (encoder_handle_) {
      AX_ENGINE_DestroyHandle(encoder_handle_);
    }

    FreeIO(&predictor_io_);
    if (predictor_handle_) {
      AX_ENGINE_DestroyHandle(predictor_handle_);
    }

    FreeIO(&decoder_io_);
    if (decoder_handle_) {
      AX_ENGINE_DestroyHandle(decoder_handle_);
    }
  }

  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    std::vector<std::string> filenames;
    SplitStringToVector(config_.paraformer.model, ",", false, &filenames);
    if (filenames.size() != 3) {
      SHERPA_ONNX_LOGE("Invalid Paraformer Axera model '%s'",
                       config_.paraformer.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    {
      auto buf = ReadFile(filenames[0]);
      InitEncoder(buf.data(), buf.size());
    }
    {
      auto buf = ReadFile(filenames[1]);
      InitPredictor(buf.data(), buf.size());
    }
    {
      auto buf = ReadFile(filenames[2]);
      InitDecoder(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    std::vector<std::string> filenames;
    SplitStringToVector(config_.paraformer.model, ",", false, &filenames);
    if (filenames.size() != 3) {
      SHERPA_ONNX_LOGE("Invalid Paraformer Axera model '%s'",
                       config_.paraformer.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    {
      auto buf = ReadFile(mgr, filenames[0]);
      InitEncoder(buf.data(), buf.size());
    }
    {
      auto buf = ReadFile(mgr, filenames[1]);
      InitPredictor(buf.data(), buf.size());
    }
    {
      auto buf = ReadFile(mgr, filenames[2]);
      InitDecoder(buf.data(), buf.size());
    }
  }

  std::vector<float> Run(std::vector<float> features) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<float> encoder_out = RunEncoder(features);
    if (encoder_out.empty()) return {};

    std::vector<float> alphas = RunPredictor(encoder_out);

    std::vector<float> acoustic_embedding =
        ComputeAcousticEmbedding(encoder_out, alphas, encoder_out_dim_);
    if (acoustic_embedding.empty()) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE("No speech found in the input audio");
      }
      return {};
    }

    int32_t num_tokens = acoustic_embedding.size() / encoder_out_dim_;
    acoustic_embedding.resize(encoder_out.size());

    return RunDecoder(std::move(encoder_out), std::move(acoustic_embedding),
                      num_tokens);
  }

  int32_t VocabSize() const { return vocab_size_; }

 private:
  std::vector<float> RunEncoder(std::vector<float> features) {
    features = ApplyLFR(std::move(features));
    if (features.empty()) return {};

    const auto &in_meta = encoder_io_info_->pInputs[0];
    std::memcpy(encoder_io_.pInputs[0].pVirAddr, features.data(),
                features.size() * sizeof(float));

    auto ret = AX_ENGINE_RunSync(encoder_handle_, &encoder_io_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync encoder failed, ret = %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out_meta = encoder_io_info_->pOutputs[0];
    size_t n = out_meta.nSize / sizeof(float);
    std::vector<float> out(n);
    std::memcpy(out.data(), encoder_io_.pOutputs[0].pVirAddr, out_meta.nSize);
    if (config_.debug) {
      SHERPA_ONNX_LOGE("Encoder output first 5: [%.3f, %.3f, %.3f, %.3f, %.3f]", 
        out[0], out[1], out[2], out[3], out[4]);
    }
    return out;
  }

  std::vector<float> RunPredictor(const std::vector<float> &encoder_out) {
    const auto &in_meta = predictor_io_info_->pInputs[0];
    std::memcpy(predictor_io_.pInputs[0].pVirAddr, encoder_out.data(),
                encoder_out.size() * sizeof(float));

    auto ret = AX_ENGINE_RunSync(predictor_handle_, &predictor_io_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync predictor failed, ret = %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out_meta = predictor_io_info_->pOutputs[0];
    size_t n = out_meta.nSize / sizeof(float);
    std::vector<float> out(n);
    std::memcpy(out.data(), predictor_io_.pOutputs[0].pVirAddr, out_meta.nSize);
    if (config_.debug) {
      SHERPA_ONNX_LOGE("Encoder output first 5: [%.3f, %.3f, %.3f, %.3f, %.3f]", 
        out[0], out[1], out[2], out[3], out[4]);
    }
    return out;
  }

  std::vector<float> RunDecoder(std::vector<float> encoder_out,
                                std::vector<float> acoustic_embedding,
                                int32_t num_tokens) {
    auto &io = decoder_io_;
    auto &info = decoder_io_info_;

    std::memcpy(io.pInputs[0].pVirAddr, encoder_out.data(),
                encoder_out.size() * sizeof(float));
    std::memcpy(io.pInputs[1].pVirAddr, acoustic_embedding.data(),
                acoustic_embedding.size() * sizeof(float));

    std::vector<float> mask(decoder_mask_size_, 0);
    for (int32_t i = 0; i < num_tokens && i < decoder_mask_size_; ++i) {
      mask[i] = 1.0f;
    }
    std::memcpy(io.pInputs[2].pVirAddr, mask.data(),
                mask.size() * sizeof(float));

    auto ret = AX_ENGINE_RunSync(decoder_handle_, &io);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync decoder failed, ret = %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out_meta = info->pOutputs[0];
    size_t n = out_meta.nSize / sizeof(float);
    std::vector<float> out(n);
    std::memcpy(out.data(), io.pOutputs[0].pVirAddr, out_meta.nSize);
    if (config_.debug) {
      SHERPA_ONNX_LOGE("Decoder output first 20: [%.3f, %.3f, %.3f, %.3f, %.3f]", 
        out[0], out[1], out[2], out[3], out[4]);
      float mx = *std::max_element(out.data(), out.data() + vocab_size_);
      SHERPA_ONNX_LOGE("Decoder frame0 max_val=%.3f max_idx=%td", mx, std::max_element(out.data(), out.data()+vocab_size_) - out.data());
    }
    if (config_.debug) {
      SHERPA_ONNX_LOGE("Encoder output first 5: [%.3f, %.3f, %.3f, %.3f, %.3f]", 
        out[0], out[1], out[2], out[3], out[4]);
    }
    return out;
  }

  static std::vector<float> ComputeAcousticEmbedding(
      const std::vector<float> &encoder_out,
      const std::vector<float> &alphas, int32_t encoder_out_dim) {
    if (alphas.empty()) return {};

    int32_t num_frames = encoder_out.size() / encoder_out_dim;
    std::vector<float> acoustic_embedding;
    const float *p = encoder_out.data();
    const float *alpha_ptr = alphas.data();

    for (int32_t i = 0; i != num_frames; ++i) {
      int32_t a = static_cast<int32_t>(alpha_ptr[i]);
      if (a <= 0) {
        continue;
      }

      int32_t start = i;
      int32_t end = std::min(start + a, num_frames);

      for (int32_t j = start; j != end; ++j) {
        auto begin = acoustic_embedding.size();
        acoustic_embedding.resize(begin + encoder_out_dim);
        float *dest = acoustic_embedding.data() + begin;
        const float *src = p + j * encoder_out_dim;
        std::copy(src, src + encoder_out_dim, dest);
      }
    }

    return acoustic_embedding;
  }

  void InitEncoder(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug,
                &encoder_handle_);
    InitInputOutputAttrs(encoder_handle_, config_.debug, &encoder_io_info_);
    PrepareIO(encoder_io_info_, &encoder_io_, config_.debug);

    auto &in = encoder_io_info_->pInputs[0];
    num_input_frames_ = in.pShape[1];
    encoder_out_dim_ = encoder_io_info_->pOutputs[0].pShape[2];

    if (config_.debug) {
      SHERPA_ONNX_LOGE("num_input_frames_: %d", num_input_frames_);
      SHERPA_ONNX_LOGE("encoder_out_dim: %d", encoder_out_dim_);
    }
  }

  void InitPredictor(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug,
                &predictor_handle_);
    InitInputOutputAttrs(predictor_handle_, config_.debug,
                         &predictor_io_info_);
    PrepareIO(predictor_io_info_, &predictor_io_, config_.debug);
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug,
                &decoder_handle_);
    InitInputOutputAttrs(decoder_handle_, config_.debug, &decoder_io_info_);
    PrepareIO(decoder_io_info_, &decoder_io_, config_.debug);

    decoder_mask_size_ = decoder_io_info_->pInputs[2].pShape[0];
    vocab_size_ = decoder_io_info_->pOutputs[0].pShape[2];

    if (config_.debug) {
      SHERPA_ONNX_LOGE("vocab_size: %d", vocab_size_);
      SHERPA_ONNX_LOGE("decoder_mask_size: %d", decoder_mask_size_);
    }
  }

  std::vector<float> ApplyLFR(std::vector<float> in) const {
    int32_t lfr_window_size = 7;
    int32_t lfr_window_shift = 6;
    int32_t in_feat_dim = 80;

    int32_t in_num_frames = in.size() / in_feat_dim;
    if (in_num_frames < lfr_window_size) return {};

    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;

    if (out_num_frames > num_input_frames_) {
      SHERPA_ONNX_LOGE(
          "Number of input frames %d is too large. Truncate it to %d frames.",
          out_num_frames, num_input_frames_);
      out_num_frames = num_input_frames_;
    }

    int32_t out_feat_dim = in_feat_dim * lfr_window_size;
    std::vector<float> out(num_input_frames_ * out_feat_dim);
    const float *p_in = in.data();
    float *p_out = out.data();

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);
      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Encoder output first 5: [%.3f, %.3f, %.3f, %.3f, %.3f]", 
        out[0], out[1], out[2], out[3], out[4]);
    }
    return out;
  }

 private:
  std::mutex mutex_;
  AxEngineGuard ax_engine_guard_;
  OfflineModelConfig config_;

  AX_ENGINE_HANDLE encoder_handle_ = nullptr;
  AX_ENGINE_HANDLE predictor_handle_ = nullptr;
  AX_ENGINE_HANDLE decoder_handle_ = nullptr;

  AX_ENGINE_IO_INFO_T *encoder_io_info_ = nullptr;
  AX_ENGINE_IO_INFO_T *predictor_io_info_ = nullptr;
  AX_ENGINE_IO_INFO_T *decoder_io_info_ = nullptr;

  AX_ENGINE_IO_T encoder_io_;
  AX_ENGINE_IO_T predictor_io_;
  AX_ENGINE_IO_T decoder_io_;

  int32_t vocab_size_ = 0;
  int32_t num_input_frames_ = -1;
  int32_t encoder_out_dim_ = -1;
  int32_t decoder_mask_size_ = 0;
};

OfflineParaformerModelAxera::~OfflineParaformerModelAxera() = default;

OfflineParaformerModelAxera::OfflineParaformerModelAxera(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineParaformerModelAxera::OfflineParaformerModelAxera(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::vector<float> OfflineParaformerModelAxera::Run(
    std::vector<float> features) const {
  return impl_->Run(std::move(features));
}

int32_t OfflineParaformerModelAxera::VocabSize() const {
  return impl_->VocabSize();
}

#if __ANDROID_API__ >= 9
template OfflineParaformerModelAxera::OfflineParaformerModelAxera(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineParaformerModelAxera::OfflineParaformerModelAxera(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
