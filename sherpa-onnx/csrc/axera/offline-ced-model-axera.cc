// sherpa-onnx/csrc/axera/offline-ced-model-axera.cc
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#include "sherpa-onnx/csrc/axera/offline-ced-model-axera.h"

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

class OfflineCEDModelAxera::Impl {
 public:
  ~Impl() {
    FreeIO(&io_data_);
    if (handle_) {
      AX_ENGINE_DestroyHandle(handle_);
    }
  }

  explicit Impl(const AudioTaggingModelConfig &config) : config_(config) {
    auto buf = ReadFile(config_.ced);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const AudioTaggingModelConfig &config) : config_(config) {
    auto buf = ReadFile(mgr, config_.ced);
    Init(buf.data(), buf.size());
  }

  int32_t NumEventClasses() const { return num_event_classes_; }

  std::vector<float> Run(std::vector<float> features) {
    std::lock_guard<std::mutex> lock(mutex_);

    int32_t num_frames = features.size() / feat_dim_;
    int32_t total = max_time_dim_ * feat_dim_;

    if (num_frames > max_time_dim_) {
      SHERPA_ONNX_LOGE(
          "Input has %d frames, but model supports at most %d frames. "
          "Truncating.",
          num_frames, max_time_dim_);
      num_frames = max_time_dim_;
      total = num_frames * feat_dim_;
    }

    std::vector<float> transposed(total, 0.0f);
    const float *src = features.data();
    float *dst = transposed.data();
    for (int32_t i = 0; i < num_frames; ++i) {
      for (int32_t j = 0; j < feat_dim_; ++j) {
        dst[j * max_time_dim_ + i] = src[i * feat_dim_ + j];
      }
    }

    const auto &in_meta = io_info_->pInputs[0];
    size_t expected = max_time_dim_ * feat_dim_ * sizeof(float);

    if (in_meta.nSize != expected) {
      SHERPA_ONNX_LOGE(
          "Input size mismatch. model expects %u bytes, but got %zu bytes",
          in_meta.nSize, expected);
      SHERPA_ONNX_EXIT(-1);
    }

    std::memcpy(io_data_.pInputs[0].pVirAddr, transposed.data(), expected);

    auto ret = AX_ENGINE_RunSync(handle_, &io_data_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync failed, ret = %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out_meta = io_info_->pOutputs[0];
    auto &out_buf = io_data_.pOutputs[0];

    size_t out_elems = out_meta.nSize / sizeof(float);
    std::vector<float> out(out_elems);
    std::memcpy(out.data(), out_buf.pVirAddr, out_meta.nSize);

    return out;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug, &handle_);
    InitInputOutputAttrs(handle_, config_.debug, &io_info_);
    PrepareIO(io_info_, &io_data_, config_.debug);

    if (!io_info_ || io_info_->nInputSize != 1 || !io_info_->pInputs) {
      SHERPA_ONNX_LOGE("CED axera model expects 1 input tensor");
      SHERPA_ONNX_EXIT(-1);
    }

    auto &in0 = io_info_->pInputs[0];
    if (in0.nShapeSize < 3) {
      SHERPA_ONNX_LOGE("Input tensor rank too small (nShapeSize = %u)",
                       in0.nShapeSize);
      SHERPA_ONNX_EXIT(-1);
    }
    feat_dim_ = in0.pShape[1];
    max_time_dim_ = in0.pShape[2];

    if (io_info_->nOutputSize != 1) {
      SHERPA_ONNX_LOGE("CED axera model expects 1 output tensor");
      SHERPA_ONNX_EXIT(-1);
    }
    num_event_classes_ = io_info_->pOutputs[0].pShape[1];

    if (config_.debug) {
      SHERPA_ONNX_LOGE("CED Axera model init done.");
      SHERPA_ONNX_LOGE("  feat_dim = %d, max_time = %d, num_events = %d",
                       feat_dim_, max_time_dim_, num_event_classes_);
    }
  }

  std::mutex mutex_;
  AxEngineGuard ax_engine_guard_;

  AudioTaggingModelConfig config_;
  AX_ENGINE_HANDLE handle_ = nullptr;
  AX_ENGINE_IO_INFO_T *io_info_ = nullptr;
  AX_ENGINE_IO_T io_data_;
  int32_t feat_dim_ = 0;
  int32_t max_time_dim_ = 0;
  int32_t num_event_classes_ = 0;
};

OfflineCEDModelAxera::~OfflineCEDModelAxera() = default;

OfflineCEDModelAxera::OfflineCEDModelAxera(
    const AudioTaggingModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {
  num_event_classes_ = impl_->NumEventClasses();
}

template <typename Manager>
OfflineCEDModelAxera::OfflineCEDModelAxera(
    Manager *mgr, const AudioTaggingModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {
  num_event_classes_ = impl_->NumEventClasses();
}

std::vector<float> OfflineCEDModelAxera::Run(
    std::vector<float> features) const {
  return impl_->Run(std::move(features));
}

#if __ANDROID_API__ >= 9
template OfflineCEDModelAxera::OfflineCEDModelAxera(
    AAssetManager *mgr, const AudioTaggingModelConfig &config);
#endif

#if __OHOS__
template OfflineCEDModelAxera::OfflineCEDModelAxera(
    NativeResourceManager *mgr, const AudioTaggingModelConfig &config);
#endif

}  // namespace sherpa_onnx
