// sherpa-onnx/csrc/axera/audio-tagging-ced-axera-impl.h
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#ifndef SHERPA_ONNX_CSRC_AXERA_AUDIO_TAGGING_CED_AXERA_IMPL_H_
#define SHERPA_ONNX_CSRC_AXERA_AUDIO_TAGGING_CED_AXERA_IMPL_H_

#include <assert.h>

#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/audio-tagging-impl.h"
#include "sherpa-onnx/csrc/audio-tagging-label-file.h"
#include "sherpa-onnx/csrc/audio-tagging.h"
#include "sherpa-onnx/csrc/axera/offline-ced-model-axera.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"

namespace sherpa_onnx {

class AudioTaggingCedAxeraImpl : public AudioTaggingImpl {
 public:
  explicit AudioTaggingCedAxeraImpl(const AudioTaggingConfig &config)
      : config_(config), model_(config.model), labels_(config.labels) {
    if (model_.NumEventClasses() != labels_.NumEventClasses()) {
      SHERPA_ONNX_LOGE("number of classes: %d (model) != %d (label file)",
                       model_.NumEventClasses(), labels_.NumEventClasses());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(CEDTag{});
  }

  std::vector<AudioEvent> Compute(OfflineStream *s,
                                  int32_t top_k = -1) const override {
    if (top_k < 0) {
      top_k = config_.top_k;
    }

    int32_t num_event_classes = model_.NumEventClasses();

    if (top_k > num_event_classes) {
      top_k = num_event_classes;
    }

    int32_t feat_dim = 64;
    std::vector<float> f = s->GetFrames();

    int32_t num_frames = f.size() / feat_dim;
    assert(feat_dim * num_frames == static_cast<int32_t>(f.size()));

    std::vector<float> probs_vec = model_.Run(std::move(f));
    if (probs_vec.empty()) {
      return {};
    }

    const float *p = probs_vec.data();
    std::vector<int32_t> top_k_indexes = TopkIndex(p, num_event_classes, top_k);

    std::vector<AudioEvent> ans(top_k);
    for (int32_t i = 0; i < top_k; ++i) {
      int32_t index = top_k_indexes[i];
      ans[i].name = labels_.GetEventName(index);
      ans[i].index = index;
      ans[i].prob = p[index];
    }

    return ans;
  }

 private:
  AudioTaggingConfig config_;
  mutable OfflineCEDModelAxera model_;
  AudioTaggingLabels labels_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_AUDIO_TAGGING_CED_AXERA_IMPL_H_
