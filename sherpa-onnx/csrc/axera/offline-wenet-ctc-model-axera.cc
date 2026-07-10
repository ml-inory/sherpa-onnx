// sherpa-onnx/csrc/axera/offline-wenet-ctc-model-axera.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axera/offline-wenet-ctc-model-axera.h"

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

class OfflineWenetCtcModelAxera::Impl {
 public:
  ~Impl() {
    FreeIO(&io_);
    if (handle_) {
      AX_ENGINE_DestroyHandle(handle_);
    }
  }

  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    auto buf = ReadFile(config_.wenet_ctc.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    auto buf = ReadFile(mgr, config_.wenet_ctc.model);
    Init(buf.data(), buf.size());
  }

  std::pair<std::vector<float>, int32_t> Run(
      const float *features, int32_t num_frames, int32_t feat_dim) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!io_info_ || io_info_->nInputSize < 2) {
      SHERPA_ONNX_LOGE("Invalid IO info: nInputSize=%d",
                       io_info_ ? io_info_->nInputSize : 0);
      return {{}, 0};
    }

    // Input 0: speech (1, num_frames, 80)
    const auto &in0_meta = io_info_->pInputs[0];
    int32_t frame_size = num_frames * feat_dim;
    std::memcpy(io_.pInputs[0].pVirAddr, features,
                frame_size * sizeof(float));

    // Input 1: speech_lengths (1,) int32
    const auto &in1_meta = io_info_->pInputs[1];
    int32_t speech_len = num_frames;
    std::memcpy(io_.pInputs[1].pVirAddr, &speech_len, sizeof(int32_t));

    auto ret = AX_ENGINE_RunSync(handle_, &io_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync failed, ret = %d", ret);
      return {{}, 0};
    }

    // Output 0: ctc_log_probs (1, T', vocab_size)
    const auto &out0_meta = io_info_->pOutputs[0];
    int32_t num_elements = out0_meta.nSize / sizeof(float);
    std::vector<float> log_probs(num_elements);
    std::memcpy(log_probs.data(), io_.pOutputs[0].pVirAddr,
                num_elements * sizeof(float));

    // Output 2: encoder_out_lens (1,) int32
    int32_t output_len = 0;
    if (io_info_->nOutputSize > 2) {
      output_len = *reinterpret_cast<int32_t *>(io_.pOutputs[2].pVirAddr);
    } else {
      output_len = num_elements / vocab_size_;
    }

    return {std::move(log_probs), output_len};
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    auto ret = AX_ENGINE_CreateHandle(&handle_, model_data,
                                       model_data_length);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("Failed to create AX engine handle. Return code: %d",
                       ret);
      SHERPA_ONNX_EXIT(-1);
    }

    ret = AX_ENGINE_GetIOInfo(handle_, &io_info_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("Failed to get IO info. Return code: %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    PrepareIO(io_info_, &io_, config_.debug);
  }

  OfflineModelConfig config_;
  AX_ENGINE_HANDLE handle_{};
  AX_ENGINE_IO_INFO_T *io_info_{};
  AX_ENGINE_IO_T io_{};
  AxEngineGuard ax_engine_guard_;
  int32_t vocab_size_ = 4233;
  mutable std::mutex mutex_;
};

OfflineWenetCtcModelAxera::~OfflineWenetCtcModelAxera() = default;

OfflineWenetCtcModelAxera::OfflineWenetCtcModelAxera(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineWenetCtcModelAxera::OfflineWenetCtcModelAxera(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::pair<std::vector<float>, int32_t> OfflineWenetCtcModelAxera::Run(
    const float *features, int32_t num_frames, int32_t feat_dim) const {
  return impl_->Run(features, num_frames, feat_dim);
}

#if __ANDROID_API__ >= 9
template OfflineWenetCtcModelAxera::OfflineWenetCtcModelAxera(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineWenetCtcModelAxera::OfflineWenetCtcModelAxera(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
