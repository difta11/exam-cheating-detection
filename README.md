# Exam Cheating Detection

Binary classification of 1-second exam-room clips: `0 = normal`, `1 = cheat`.

## Data (`dataset/`, also on Kaggle as `diftaradityapratama/exam-cheating-dataset`)
- `train/{cheat,normal}/subject_{1..26}/{front,side}_<Behavior>.MOV` — 416 videos, 1280x720, ~16 s each.
  - normal: Writing_Answers, Thinking, Looking_Around, Checking_Smartwatch
  - cheat: Looking_Under_the_Desk, Reading_Hidden_Notes, Repeated_Peeking, Using_Smartwatch
  - front camera = HEVC 10-bit HLG HDR @ 30 fps; side camera = HEVC 8-bit SDR @ 29.97 fps.
- `test/test_seq_XXXXX/01..30.jpg` — 1515 sequences (30 frames, 1 s), 742 front / 773 side.
- `sample_submission.csv` — `id,label`.

## Findings (EDA)
- Plain OpenCV decoding of the HDR front videos matches the colour of the test JPGs better than tone-mapping.
- In every video the subject mostly writes; the defining behaviour appears only in short moments, so
  per-window labels are noisy (the ceiling for pure 1-s classification is low).
- Test sequences are shuffled consecutive 1-s windows of 96 source videos. Linking frame 30 -> frame 01
  recovers them (`tools/test_link_chains.py`); angle (pose shoulder width) + clothing clustering
  (`tools/test_subject_groups.py`, side groups hand-checked) gives 6 subjects x 2 angles x 8 videos,
  the same layout as each training subject. Result: `outputs/structure/test_videos.json`.

## Models (CV = 5-fold GroupKFold by subject)
| ver | notebook | idea | CV accuracy (1-s windows) |
|---|---|---|---|
| v1 | `cheat_dinov2_temporal.py` | frozen DINOv2 frames + temporal transformer | 0.559 |
| v2 | `cheat_pose_temporal.py` | YOLO11-pose keypoints, BiGRU + LightGBM | 0.597 |
| v3 | `cheat_video_mil.py` | person crops, frozen V-JEPA 2 / VideoMAE (SSv2) + pose, MIL head | 0.679 pure, 0.863 with ±7 context |
| v4 | `cheat_video_level.py` | MIL on whole re-linked videos; Hungarian behaviour assignment per (subject, angle) | 0.898 per video, **0.981** assignment |

v4 outputs: `submission_pure.csv` (v3 per-window, no test structure), `submission_video.csv` (one label per
re-linked video), `submission_group.csv` (behaviour assignment per subject group). The last two exploit the
test-set structure — check the task rules before submitting them.

## Leaderboard feedback and the per-second hypothesis (v5)
- `submission_group.csv` scored **0.7507** although its calibrated expected per-video accuracy is 0.975
  (front/side takes pair up by synchronised pose motion, 99% on train; behaviours agree for 46/48 pairs).
- Every video is ~half writing, then the behaviour from ~49% of its length (manual onsets of 22 train
  videos, `tools/onset_sheets.py`; detector change points on all videos, `tools/onset_analysis.py`).
  Labelling a whole cheat video as 1 when only its second half is "cheat" gives exactly ~0.75.
- `notebooks/cheat_v5_moments.py`: joint front+side assignment + instance-level MIL per-second scores.
- `tools/onset_model.py`: onset = position prior + detector change point + paired view (77% exact, 100%
  within ±1 s on the annotation). `tools/make_onset_submission.py` -> `outputs/v5_post/submission_onset.csv`
  (+ `_m1` / `_p1` shifted variants).

## Run on Kaggle
```bash
python tools/py2ipynb.py notebooks/<nb>.py kaggle/<slug>/<slug>.ipynb      # v1-v3
python tools/build_v4.py                                                     # v4 (embeds test structure)
uvx --from kaggle --with six kaggle kernels push -p kaggle/<slug>
uvx --from kaggle --with six kaggle kernels status diftaradityapratama/<slug>
uvx --from kaggle --with six kaggle kernels output diftaradityapratama/<slug> -p outputs/<ver>/
```
Slugs: `exam-cheating-dinov2-temporal`, `exam-cheating-pose-temporal`, `exam-cheating-video-mil`,
`exam-cheating-video-level` (v3 reads v2 outputs; v4 reads v2 + v3 outputs as kernel sources).
