// sherpa-onnx/csrc/axera/silero-vad-model-axera.cc
// Stub implementation - AXERA Silero VAD is currently unsupported

#include "sherpa-onnx/csrc/axera/silero-vad-model-axera.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

class SileroVadModelAxera::Impl {
 public:
  explicit Impl(const VadModelConfig &) {}
  template <typename Manager>
  Impl(Manager *, const VadModelConfig &) {}
  void Reset() {}
  bool IsSpeech(const float *, int32_t) { return false; }
  float Compute(const float *, int32_t) { return 0.0f; }
  int32_t WindowSize() const { return 512; }
  int32_t WindowShift() const { return 256; }
  int32_t MinSilenceDurationSamples() const { return 0; }
  int32_t MinSpeechDurationSamples() const { return 0; }
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
