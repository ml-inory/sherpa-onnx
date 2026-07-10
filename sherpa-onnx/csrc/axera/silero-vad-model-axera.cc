// sherpa-onnx/csrc/axera/silero-vad-model-axera.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axera/silero-vad-model-axera.h"

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

class SileroVadModelAxera::Impl {
 public:
  explicit Impl(const VadModelConfig &config)
      : config_(config),
        sample_rate_(config.sample_rate) {
    if (sample_rate_ != 16000) {
      SHERPA_ONNX_LOGE("Silero VAD axera requires sample rate 16000. Got %d",
                       sample_rate_);
      SHERPA_ONNX_EXIT(-1);
    }

    auto buf = ReadFile(config.silero_vad.model);
    Init(buf.data(), buf.size());

    context_.resize(context_size_, 0.0f);
    ResetState();
  }

  template <typename Manager>
  Impl(Manager *mgr, const VadModelConfig &config)
      : config_(config),
        sample_rate_(config.sample_rate) {
    if (sample_rate_ != 16000) {
      SHERPA_ONNX_LOGE("Silero VAD axera requires sample rate 16000. Got %d",
                       sample_rate_);
      SHERPA_ONNX_EXIT(-1);
    }

    auto buf = ReadFile(mgr, config.silero_vad.model);
    Init(buf.data(), buf.size());

    context_.resize(context_size_, 0.0f);
    ResetState();
  }

  void Reset() {
    std::fill(context_.begin(), context_.end(), 0.0f);
    ResetState();
  }

  bool IsSpeech(const float *samples, int32_t n) {
    return Compute(samples, n) > config_.silero_vad.threshold;
  }

  float Compute(const float *samples, int32_t n) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Build input: context(64) + samples(n=576) + pad(64) = 704
    // Actually: the Sherpa V5 interface gives WindowSize()=576 samples
    // which is context(64) + audio(512). The model expects [1, 640] input:
    // context(64) + audio(512) + pad(64)
    int32_t total_input = context_size_ + n + context_size_;
    std::vector<float> input_data(total_input, 0.0f);

    // Fill context
    std::memcpy(input_data.data(), context_.data(),
                context_size_ * sizeof(float));
    // Fill audio
    std::memcpy(input_data.data() + context_size_, samples,
                n * sizeof(float));

    // Reflect pad the last 64
    int32_t audio_end = context_size_ + n - 1;
    for (int32_t i = 0; i < context_size_; ++i) {
      int32_t src = audio_end - i;
      if (src >= context_size_) {
        input_data[context_size_ + n + i] = samples[src - context_size_];
      }
    }

    // Copy to AX engine input
    std::memcpy(io_.pInputs[0].pVirAddr, input_data.data(),
                total_input * sizeof(float));
    std::memcpy(io_.pInputs[1].pVirAddr, state_.data(),
                state_.size() * sizeof(float));

    auto ret = AX_ENGINE_RunSync(handle_, &io_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("SileroVAD RunSync failed, ret=%d", ret);
      return 0.0f;
    }

    // Read output - speech probability
    float prob = *reinterpret_cast<float *>(io_.pOutputs[0].pVirAddr);

    // Update state
    std::memcpy(state_.data(), io_.pOutputs[1].pVirAddr,
                state_.size() * sizeof(float));

    // Update context: last context_size_ samples of input audio
    std::memcpy(context_.data(), samples + n - context_size_,
                context_size_ * sizeof(float));

    return prob;
  }

  int32_t WindowSize() const {
    // V5: window_size + overlap = 512 + 64 = 576
    return config_.silero_vad.window_size + context_size_;
  }

  int32_t WindowShift() const {
    return config_.silero_vad.window_size;  // 512
  }

  int32_t MinSilenceDurationSamples() const { return min_silence_samples_; }
  int32_t MinSpeechDurationSamples() const { return min_speech_samples_; }

 private:
  void Init(void *model_data, size_t model_data_length) {
    auto ret = AX_ENGINE_CreateHandle(&handle_, model_data,
                                       model_data_length);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("Failed to create SileroVAD AX handle, ret=%d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    ret = AX_ENGINE_GetIOInfo(handle_, &io_info_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("Failed to get SileroVAD IO info, ret=%d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    PrepareIO(io_info_, &io_, config_.debug);
  }

  void ResetState() {
    state_.assign(2 * 1 * hidden_size_, 0.0f);  // [2, 1, 128]
  }

  VadModelConfig config_;
  AX_ENGINE_HANDLE handle_{};
  AX_ENGINE_IO_INFO_T *io_info_{};
  AX_ENGINE_IO_T io_{};
  AxEngineGuard ax_engine_guard_;
  int32_t sample_rate_ = 16000;
  int32_t context_size_ = 64;
  int32_t hidden_size_ = 128;
  int32_t min_silence_samples_ = 0;
  int32_t min_speech_samples_ = 0;
  std::vector<float> context_;
  std::vector<float> state_;
  mutable std::mutex mutex_;
};

SileroVadModelAxera::SileroVadModelAxera(const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
SileroVadModelAxera::SileroVadModelAxera(Manager *mgr,
                                         const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

SileroVadModelAxera::~SileroVadModelAxera() = default;

void SileroVadModelAxera::Reset() { impl_->Reset(); }
bool SileroVadModelAxera::IsSpeech(const float *samples, int32_t n) {
  return impl_->IsSpeech(samples, n);
}
float SileroVadModelAxera::Compute(const float *samples, int32_t n) {
  return impl_->Compute(samples, n);
}
int32_t SileroVadModelAxera::WindowSize() const {
  return impl_->WindowSize();
}
int32_t SileroVadModelAxera::WindowShift() const {
  return impl_->WindowShift();
}
int32_t SileroVadModelAxera::MinSilenceDurationSamples() const {
  return impl_->MinSilenceDurationSamples();
}
int32_t SileroVadModelAxera::MinSpeechDurationSamples() const {
  return impl_->MinSpeechDurationSamples();
}

#if __ANDROID_API__ >= 9
template SileroVadModelAxera::SileroVadModelAxera(AAssetManager *mgr,
                                                  const VadModelConfig &config);
#endif

#if __OHOS__
template SileroVadModelAxera::SileroVadModelAxera(
    NativeResourceManager *mgr, const VadModelConfig &config);
#endif

}  // namespace sherpa_onnx
