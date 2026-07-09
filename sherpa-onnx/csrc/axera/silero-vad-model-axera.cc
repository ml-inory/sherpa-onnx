// sherpa-onnx/csrc/axera/silero-vad-model-axera.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axera/silero-vad-model-axera.h"

#include <cstring>
#include <memory>
#include <string>
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

class SileroVadModelAxera::Impl {
 public:
  ~Impl() {
    FreeIO(&io_);
    if (handle_) {
      AX_ENGINE_DestroyHandle(handle_);
    }
  }

  explicit Impl(const VadModelConfig &config)
      : config_(config), sample_rate_(config.sample_rate) {
    auto buf = ReadFile(config.silero_vad.model);
    Init(buf.data(), buf.size());

    if (sample_rate_ != 16000) {
      SHERPA_ONNX_LOGE("Expected sample rate 16000. Given: %d",
                       config.sample_rate);
      SHERPA_ONNX_EXIT(-1);
    }

    min_silence_samples_ =
        sample_rate_ * config_.silero_vad.min_silence_duration;

    min_speech_samples_ =
        sample_rate_ * config_.silero_vad.min_speech_duration;
  }

  template <typename Manager>
  Impl(Manager *mgr, const VadModelConfig &config)
      : config_(config), sample_rate_(config.sample_rate) {
    auto buf = ReadFile(mgr, config.silero_vad.model);
    Init(buf.data(), buf.size());

    if (sample_rate_ != 16000) {
      SHERPA_ONNX_LOGE("Expected sample rate 16000. Given: %d",
                       config.sample_rate);
      exit(-1);
    }

    min_silence_samples_ =
        sample_rate_ * config_.silero_vad.min_silence_duration;

    min_speech_samples_ =
        sample_rate_ * config_.silero_vad.min_speech_duration;
  }

  void Reset() {
    for (auto &s : states_) {
      std::fill(s.begin(), s.end(), 0);
    }

    triggered_ = false;
    current_sample_ = 0;
    temp_start_ = 0;
    temp_end_ = 0;
  }

  bool IsSpeech(const float *samples, int32_t n) {
    if (n != WindowSize()) {
      SHERPA_ONNX_LOGE("n: %d != window_size: %d", n, WindowSize());
      SHERPA_ONNX_EXIT(-1);
    }

    float prob = Run(samples, n);

    float threshold = config_.silero_vad.threshold;

    current_sample_ += config_.silero_vad.window_size;

    if (prob > threshold && temp_end_ != 0) {
      temp_end_ = 0;
    }

    if (prob > threshold && temp_start_ == 0) {
      temp_start_ = current_sample_;
      return false;
    }

    if (prob > threshold && temp_start_ != 0 && !triggered_) {
      if (current_sample_ - temp_start_ < min_speech_samples_) {
        return false;
      }

      triggered_ = true;

      return true;
    }

    if ((prob < threshold) && !triggered_) {
      temp_start_ = 0;
      temp_end_ = 0;
      return false;
    }

    if ((prob > threshold - 0.15) && triggered_) {
      return true;
    }

    if ((prob > threshold) && !triggered_) {
      triggered_ = true;

      return true;
    }

    if ((prob < threshold) && triggered_) {
      if (temp_end_ == 0) {
        temp_end_ = current_sample_;
      }

      if (current_sample_ - temp_end_ < min_silence_samples_) {
        return true;
      }
      temp_start_ = 0;
      temp_end_ = 0;
      triggered_ = false;
      return false;
    }

    return false;
  }

  int32_t WindowShift() const { return config_.silero_vad.window_size; }

  int32_t WindowSize() const {
    return config_.silero_vad.window_size + window_overlap_;
  }

  int32_t MinSilenceDurationSamples() const { return min_silence_samples_; }

  int32_t MinSpeechDurationSamples() const { return min_speech_samples_; }

  void SetMinSilenceDuration(float s) {
    min_silence_samples_ = sample_rate_ * s;
  }

  void SetThreshold(float threshold) {
    config_.silero_vad.threshold = threshold;
  }

  float Compute(const float *samples, int32_t n) { return Run(samples, n); }

  float Run(const float *samples, int32_t n) {
    // Input 0: x, shape [1, 512], float32
    std::memcpy(io_.pInputs[0].pVirAddr, samples, n * sizeof(float));

    // Input 1: h, shape [2, 1, 64], float32
    std::memcpy(io_.pInputs[1].pVirAddr, states_[0].data(),
                states_[0].size() * sizeof(float));

    // Input 2: c, shape [2, 1, 64], float32
    std::memcpy(io_.pInputs[2].pVirAddr, states_[1].data(),
                states_[1].size() * sizeof(float));

    auto ret = AX_ENGINE_RunSync(handle_, &io_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync failed, ret = %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    // Output 0: prob, shape [1, 1], float32
    float prob;
    std::memcpy(&prob, io_.pOutputs[0].pVirAddr, sizeof(float));

    // Output 1: next_h, shape [2, 1, 64], float32
    std::memcpy(states_[0].data(), io_.pOutputs[1].pVirAddr,
                states_[0].size() * sizeof(float));

    // Output 2: next_c, shape [2, 1, 64], float32
    float next_h_first;
    std::memcpy(&next_h_first, io_.pOutputs[1].pVirAddr, sizeof(float));
    }
    std::memcpy(states_[1].data(), io_.pOutputs[2].pVirAddr,
                states_[1].size() * sizeof(float));

    return prob;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {

    InitContext(model_data, model_data_length, config_.debug, &handle_);
    InitInputOutputAttrs(handle_, config_.debug, &io_info_);
    PrepareIO(io_info_, &io_, config_.debug);

    // Verify input/output shapes match expectations
    if (config_.silero_vad.window_size != 512) {
      SHERPA_ONNX_LOGE("we require window_size to be 512. Given: %d",
                       config_.silero_vad.window_size);
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t h_size = io_info_->pOutputs[1].pShape[0] *
                     io_info_->pOutputs[1].pShape[1] *
                     io_info_->pOutputs[1].pShape[2];
    int32_t c_size = io_info_->pOutputs[2].pShape[0] *
                     io_info_->pOutputs[2].pShape[1] *
                     io_info_->pOutputs[2].pShape[2];

    states_.resize(2);
    states_[0].resize(h_size);
    states_[1].resize(c_size);

    Reset();
  }

 private:
  AxEngineGuard ax_engine_guard_;
  VadModelConfig config_;
  int64_t sample_rate_;

  AX_ENGINE_HANDLE handle_ = nullptr;
  AX_ENGINE_IO_INFO_T *io_info_ = nullptr;
  AX_ENGINE_IO_T io_;

  std::vector<std::vector<float>> states_;

  int32_t min_silence_samples_;
  int32_t min_speech_samples_;

  bool triggered_ = false;
  int32_t current_sample_ = 0;
  int32_t temp_start_ = 0;
  int32_t temp_end_ = 0;

  int32_t window_overlap_ = 0;
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

void SileroVadModelAxera::SetMinSilenceDuration(float s) {
  impl_->SetMinSilenceDuration(s);
}

void SileroVadModelAxera::SetThreshold(float threshold) {
  impl_->SetThreshold(threshold);
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
