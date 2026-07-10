// sherpa-onnx/csrc/axera/offline-wenet-ctc-model-axera.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXERA_OFFLINE_WENET_CTC_MODEL_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_OFFLINE_WENET_CTC_MODEL_AXERA_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineWenetCtcModelAxera {
 public:
  ~OfflineWenetCtcModelAxera();

  explicit OfflineWenetCtcModelAxera(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineWenetCtcModelAxera(Manager *mgr, const OfflineModelConfig &config);

  /** Run the encoder and return CTC log_probs.
   *
   * @param features  A vector of shape (N*T*C), float32.
   * @param num_frames  Number of feature frames (T).
   * @param feat_dim  Feature dimension (C), typically 80.
   *
   * @return A pair of (log_probs, log_probs_len).
   *   log_probs: flattened float32 array of shape (1*T'*vocab_size)
   *   log_probs_len: int32, output length T'
   */
  std::pair<std::vector<float>, int32_t> Run(
      const float *features, int32_t num_frames, int32_t feat_dim) const;

  int32_t VocabSize() const { return vocab_size_; }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  int32_t vocab_size_ = 4233;
  int32_t subsampling_factor_ = 4;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_OFFLINE_WENET_CTC_MODEL_AXERA_H_
