// sherpa-onnx/csrc/axera/offline-ced-model-axera.h
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#ifndef SHERPA_ONNX_CSRC_AXERA_OFFLINE_CED_MODEL_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_OFFLINE_CED_MODEL_AXERA_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/audio-tagging-model-config.h"

namespace sherpa_onnx {

class OfflineCEDModelAxera {
 public:
  explicit OfflineCEDModelAxera(const AudioTaggingModelConfig &config);

  template <typename Manager>
  OfflineCEDModelAxera(Manager *mgr, const AudioTaggingModelConfig &config);

  ~OfflineCEDModelAxera();

  std::vector<float> Run(std::vector<float> features) const;

  int32_t NumEventClasses() const { return num_event_classes_; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  int32_t num_event_classes_ = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_OFFLINE_CED_MODEL_AXERA_H_
