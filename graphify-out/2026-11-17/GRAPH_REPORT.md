# Graph Report - exam-cheating-detection  (2026-11-17)

## Corpus Check
- Corpus is ~2,200 words - fits in a single context window. You may not need a graph.

## Summary
- 49 nodes · 59 edges · 8 communities (3 shown, 5 thin omitted)
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

## God Nodes (most connected - your core abstractions)
1. `train_one()` - 5 edges
2. `TemporalHead` - 4 edges
3. `convert()` - 4 edges
4. `decode_video()` - 3 edges
5. `extract()` - 3 edges
6. `predict()` - 3 edges
7. `seed_all()` - 2 edges
8. `decode_cv2()` - 2 edges
9. `decode_ffmpeg()` - 2 edges
10. `bounded_map()` - 2 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (8 total, 5 thin omitted)

### Community 2 - "Temporal Head Training"
Cohesion: 0.29
Nodes (5): gather(), sample_windows(), seed_all(), TemporalHead, train_one()

### Community 3 - "DINOv2 Feature Extraction"
Cohesion: 0.40
Nodes (3): extract(), predict(), prob_from()

### Community 4 - "Video Decoding"
Cohesion: 0.67
Nodes (3): decode_cv2(), decode_ffmpeg(), decode_video()

## Knowledge Gaps
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TemporalHead` connect `Temporal Head Training` to `Notebook Pipeline & Imports`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Should `Notebook Pipeline & Imports` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._
- **Why does `extract()` connect `DINOv2 Feature Extraction` to `Notebook Pipeline & Imports`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._