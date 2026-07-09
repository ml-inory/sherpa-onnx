// sherpa-onnx/csrc/axera/silero-vad-model-axera.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AXERA_SILERO_VAD_MODEL_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_SILERO_VAD_MODEL_AXERA_H_

#include <memory>

#include "sherpa-onnx/csrc/vad-model.h"
#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

class SileroVadModelAxera : public VadModel {
 public:
  explicit SileroVadModelAxera(const VadModelConfig &config);

  template <typename Manager>
  SileroVadModelAxera(Manager *mgr, const VadModelConfig &config);

  ~SileroVadModelAxera() override;

  void Reset() override;

  bool IsSpeech(const float *samples, int32_t n) override;
  float Compute(const float *samples, int32_t n) override;

  int32_t WindowSize() const override;
  int32_t WindowShift() const override;

  int32_t MinSilenceDurationSamples() const override;
  int32_t MinSpeechDurationSamples() const override;

  void SetMinSilenceDuration(float s) override;
  void SetThreshold(float threshold) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_SILERO_VAD_MODEL_AXERA_H_
