// sherpa-onnx/csrc/axera/speaker-embedding-extractor-model-axera.cc
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#include "sherpa-onnx/csrc/axera/speaker-embedding-extractor-model-axera.h"

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

class SpeakerEmbeddingExtractorAxeraImpl::Impl {
 public:
  ~Impl() {
    FreeIO(&io_data_);
    if (handle_) {
      AX_ENGINE_DestroyHandle(handle_);
    }
  }

  explicit Impl(const SpeakerEmbeddingExtractorConfig &config)
      : config_(config) {
    auto buf = ReadFile(config_.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const SpeakerEmbeddingExtractorConfig &config)
      : config_(config) {
    auto buf = ReadFile(mgr, config_.model);
    Init(buf.data(), buf.size());
  }

  int32_t OutputDim() const { return output_dim_; }
  int32_t FeatDim() const { return feat_dim_; }
  int32_t MaxFrames() const { return max_frames_; }

  std::vector<float> Run(const float *features, int32_t num_frames,
                          int32_t feat_dim) {
    std::lock_guard<std::mutex> lock(mutex_);

    int32_t total = num_frames * feat_dim;
    std::vector<float> padded(max_frames_ * feat_dim_, 0.0f);
    int32_t copy_frames = std::min(num_frames, max_frames_);
    std::copy(features, features + copy_frames * feat_dim, padded.data());

    const auto &in_meta = io_info_->pInputs[0];
    size_t expected = max_frames_ * feat_dim_ * sizeof(float);
    if (in_meta.nSize != expected) {
      SHERPA_ONNX_LOGE("Input size mismatch: %u vs %zu", in_meta.nSize,
                       expected);
      SHERPA_ONNX_EXIT(-1);
    }
    std::memcpy(io_data_.pInputs[0].pVirAddr, padded.data(), expected);

    auto ret = AX_ENGINE_RunSync(handle_, &io_data_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync failed, ret = %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out_meta = io_info_->pOutputs[0];
    size_t out_elems = out_meta.nSize / sizeof(float);
    std::vector<float> out(out_elems);
    std::memcpy(out.data(), io_data_.pOutputs[0].pVirAddr, out_meta.nSize);

    return out;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug, &handle_);
    InitInputOutputAttrs(handle_, config_.debug, &io_info_);
    PrepareIO(io_info_, &io_data_, config_.debug);

    if (!io_info_ || io_info_->nInputSize != 1) {
      SHERPA_ONNX_LOGE("Speaker model expects 1 input, got %u",
                       io_info_ ? io_info_->nInputSize : 0);
      SHERPA_ONNX_EXIT(-1);
    }
    auto &in0 = io_info_->pInputs[0];
    max_frames_ = in0.pShape[1];
    feat_dim_ = in0.pShape[2];

    if (io_info_->nOutputSize != 1) {
      SHERPA_ONNX_LOGE("Speaker model expects 1 output");
      SHERPA_ONNX_EXIT(-1);
    }
    output_dim_ = io_info_->pOutputs[0].pShape[1];

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Axera Speaker model init: max_frames=%d feat_dim=%d output_dim=%d",
                       max_frames_, feat_dim_, output_dim_);
    }
  }

  std::mutex mutex_;
  AxEngineGuard ax_engine_guard_;
  SpeakerEmbeddingExtractorConfig config_;
  AX_ENGINE_HANDLE handle_ = nullptr;
  AX_ENGINE_IO_INFO_T *io_info_ = nullptr;
  AX_ENGINE_IO_T io_data_;
  int32_t max_frames_ = 0;
  int32_t feat_dim_ = 0;
  int32_t output_dim_ = 0;
};

SpeakerEmbeddingExtractorAxeraImpl::~SpeakerEmbeddingExtractorAxeraImpl() =
    default;

SpeakerEmbeddingExtractorAxeraImpl::SpeakerEmbeddingExtractorAxeraImpl(
    const SpeakerEmbeddingExtractorConfig &config)
    : impl_(std::make_unique<Impl>(config)) {
  output_dim_ = impl_->OutputDim();
  feat_dim_ = impl_->FeatDim();
}

template <typename Manager>
SpeakerEmbeddingExtractorAxeraImpl::SpeakerEmbeddingExtractorAxeraImpl(
    Manager *mgr, const SpeakerEmbeddingExtractorConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {
  output_dim_ = impl_->OutputDim();
  feat_dim_ = impl_->FeatDim();
}

int32_t SpeakerEmbeddingExtractorAxeraImpl::Dim() const { return output_dim_; }

std::unique_ptr<OnlineStream>
SpeakerEmbeddingExtractorAxeraImpl::CreateStream() const {
  FeatureExtractorConfig feat_config;
  feat_config.sampling_rate = 16000;
  feat_config.normalize_samples = 0;
  feat_config.feature_dim = feat_dim_;
  return std::make_unique<OnlineStream>(feat_config);
}

bool SpeakerEmbeddingExtractorAxeraImpl::IsReady(OnlineStream *s) const {
  return s->GetNumProcessedFrames() < s->NumFramesReady();
}

std::vector<float> SpeakerEmbeddingExtractorAxeraImpl::Compute(
    OnlineStream *s) const {
  int32_t num_frames = s->NumFramesReady() - s->GetNumProcessedFrames();
  if (num_frames <= 0) {
    SHERPA_ONNX_LOGE("IsReady must return true before calling Compute");
    return {};
  }

  std::vector<float> features =
      s->GetFrames(s->GetNumProcessedFrames(), num_frames);
  s->GetNumProcessedFrames() += num_frames;

  return impl_->Run(features.data(), num_frames, feat_dim_);
}

#if __ANDROID_API__ >= 9
template SpeakerEmbeddingExtractorAxeraImpl::
    SpeakerEmbeddingExtractorAxeraImpl(
        AAssetManager *mgr, const SpeakerEmbeddingExtractorConfig &config);
#endif

#if __OHOS__
template SpeakerEmbeddingExtractorAxeraImpl::
    SpeakerEmbeddingExtractorAxeraImpl(
        NativeResourceManager *mgr,
        const SpeakerEmbeddingExtractorConfig &config);
#endif

}  // namespace sherpa_onnx
