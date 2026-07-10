// sherpa-onnx/csrc/axera/speaker-embedding-extractor-model-axera.h
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#ifndef SHERPA_ONNX_CSRC_AXERA_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_AXERA_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/speaker-embedding-extractor-impl.h"

namespace sherpa_onnx {

class SpeakerEmbeddingExtractorAxeraImpl : public SpeakerEmbeddingExtractorImpl {
 public:
  explicit SpeakerEmbeddingExtractorAxeraImpl(
      const SpeakerEmbeddingExtractorConfig &config);

  template <typename Manager>
  SpeakerEmbeddingExtractorAxeraImpl(
      Manager *mgr, const SpeakerEmbeddingExtractorConfig &config);

  ~SpeakerEmbeddingExtractorAxeraImpl() override;

  int32_t Dim() const override;

  std::unique_ptr<OnlineStream> CreateStream() const override;

  bool IsReady(OnlineStream *s) const override;

  std::vector<float> Compute(OnlineStream *s) const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  int32_t output_dim_ = 0;
  int32_t feat_dim_ = 80;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_AXERA_H_
