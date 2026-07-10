#include "sherpa-onnx/csrc/axera/offline-wenet-ctc-model-axera.h"
// sherpa-onnx/csrc/axera/offline-recognizer-wenet-ctc-axera-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXERA_OFFLINE_RECOGNIZER_WENET_CTC_AXERA_IMPL_H_
#define SHERPA_ONNX_CSRC_AXERA_OFFLINE_RECOGNIZER_WENET_CTC_AXERA_IMPL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-ctc-decoder.h"
#include "sherpa-onnx/csrc/offline-recognizer-ctc-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/rknn/offline-ctc-greedy-search-decoder-rknn.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

class OfflineRecognizerWenetCtcAxeraImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerWenetCtcAxeraImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineWenetCtcModelAxera>(
            config.model_config)) {
    InitFeatConfig();
  }

  template <typename Manager>
  OfflineRecognizerWenetCtcAxeraImpl(Manager *mgr,
                                     const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineWenetCtcModelAxera>(
            mgr, config.model_config)) {
    InitFeatConfig();
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i < n; ++i) {
      auto s = ss[i];
      auto frames = s->GetFrames();
      int32_t feat_dim = s->FeatureDim();
      int32_t num_frames = static_cast<int32_t>(frames.size()) / feat_dim;

      SHERPA_ONNX_LOGE("[WenetAxeraImpl] frames.size()=%zu feat_dim=%d num_frames=%d",
                       frames.size(), feat_dim, num_frames);

      auto [log_probs_vec, output_len] = model_->Run(
          frames.data(), num_frames, feat_dim);

      int32_t vocab_size = model_->VocabSize();
      if (output_len <= 0 || log_probs_vec.empty()) {
        SHERPA_ONNX_LOGE("Wenet CTC axera encoder returned empty output");
        continue;
      }

      // CTC greedy search decode
      OfflineCtcGreedySearchDecoderRknn decoder(0);  // blank_id=0
      auto ctc_result = decoder.Decode(log_probs_vec.data(),
                                       output_len, vocab_size);

      auto result = Convert(ctc_result, symbol_table_,
                            10,  // frame_shift_ms
                            model_->SubsamplingFactor());
      s->SetResult(result);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void InitFeatConfig() {
    // WeNet CTC models assume input samples are in [-32768, 32767]
    config_.feat_config.normalize_samples = false;
    config_.feat_config.dither = 1;
  }

  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineWenetCtcModelAxera> model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_OFFLINE_RECOGNIZER_WENET_CTC_AXERA_IMPL_H_
