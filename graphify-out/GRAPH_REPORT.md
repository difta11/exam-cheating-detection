# Graph Report - exam-cheating-detection  (2026-11-17)

## Corpus Check
- 8 files · ~4,543 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 1 file(s) not represented in the graph (top: (none) 1)

## Summary
- 88 nodes · 125 edges · 12 communities (6 shown, 6 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Notebook Converter Tool
- Temporal Head Training
- DINOv2 Feature Extraction
- Video Decoding
- Bounded Parallel Map
- CV Window Evaluation
- Test Sequence Linking
- cheat_pose_temporal.py
- train_gru
- Exam Cheating Detection
- CLAUDE.md

## God Nodes (most connected - your core abstractions)
1. `train_gru()` - 6 edges
2. `train_one()` - 5 edges
3. `featurize()` - 5 edges
4. `predict_gru()` - 5 edges
5. `Exam Cheating Detection` - 5 edges
6. `TemporalHead` - 4 edges
7. `hflip()` - 4 edges
8. `PoseGRU` - 4 edges
9. `stats_np()` - 4 edges
10. `convert()` - 4 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (12 total, 6 thin omitted)

### Community 2 - "Temporal Head Training"
Cohesion: 0.29
Nodes (5): gather(), sample_windows(), seed_all(), TemporalHead, train_one()

### Community 3 - "DINOv2 Feature Extraction"
Cohesion: 0.40
Nodes (3): extract(), predict(), prob_from()

### Community 4 - "Video Decoding"
Cohesion: 0.67
Nodes (3): decode_cv2(), decode_ffmpeg(), decode_video()

### Community 8 - "cheat_pose_temporal.py"
Cohesion: 0.11
Nodes (4): bounded_map(), run_pose(), seq_thumbs(), thumb()

### Community 9 - "train_gru"
Cohesion: 0.14
Nodes (10): featurize(), gather(), hflip(), PoseGRU, predict_gru(), prob_from(), seed_all(), stats_np() (+2 more)

### Community 10 - "Exam Cheating Detection"
Cohesion: 0.33
Nodes (5): Data (`dataset/`, also on Kaggle as `diftaradityapratama/exam-cheating-dataset`), Exam Cheating Detection, Findings (EDA), Model (`notebooks/cheat_dinov2_temporal.py`), Run on Kaggle

## Knowledge Gaps
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TemporalHead` connect `Temporal Head Training` to `Notebook Pipeline & Imports`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Should `Notebook Pipeline & Imports` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._
- **Why does `PoseGRU` connect `train_gru` to `cheat_pose_temporal.py`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Should `cheat_pose_temporal.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10526315789473684 - nodes in this community are weakly interconnected._
- **Should `train_gru` be split into smaller, more focused modules?**
  _Cohesion score 0.14166666666666666 - nodes in this community are weakly interconnected._