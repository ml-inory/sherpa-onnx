// sherpa-onnx/csrc/axera/offline-paraformer-model-axera.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AXERA_OFFLINE_PARAFORMER_MODEL_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_OFFLINE_PARAFORMER_MODEL_AXERA_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineParaformerModelAxera {
 public:
  ~OfflineParaformerModelAxera();

  explicit OfflineParaformerModelAxera(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineParaformerModelAxera(Manager *mgr, const OfflineModelConfig &config);

  std::vector<float> Run(std::vector<float> features) const;

  int32_t VocabSize() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_OFFLINE_PARAFORMER_MODEL_AXERA_H_
