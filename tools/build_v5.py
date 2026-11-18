"""Assemble notebooks/cheat_v5_moments.py (v4 data/feature/video-MIL code + v5 sections) and build the Kaggle ipynb.

usage: python tools/build_v5.py <v5_tail.py>     (project root)
The v5 tail is kept in notebooks/_v5_tail.py once assembled; the structure JSON is embedded for Kaggle.
"""
import subprocess, sys
from pathlib import Path

HEAD = '''# %% [markdown]
# # Exam Cheating Detection v5 — joint front+side assignment + per-second cheating moments
#
# Leaderboard feedback: the per-video submission (CV 0.98) scored 0.75. The re-linked test structure is
# confirmed independently (front/side takes pair up by synchronised motion, 99% on train; predicted behaviours
# agree for 46/48 pairs), so the most likely explanation is that test sequences are labelled per second by what
# is visible: a 1-s clip of a cheat video in which the subject is only writing counts as normal.
# v5 adds an instance-level MIL detector that scores every 1-s window and writes:
# * `submission_joint.csv`      - per-video labels, joint front+side behaviour assignment
# * `submission_moment_beh.csv` - inside videos assigned to a cheat behaviour, only windows whose evidence for
#                                 that behaviour beats every normal behaviour are 1
# * `submission_moment_pXX.csv` - inside cheat videos, windows above the XXth percentile of normal-video scores
'''

v4 = Path("notebooks/cheat_video_level.py").read_text(encoding="utf-8").splitlines()
end = next(i for i, l in enumerate(v4) if "## Decisions" in l) - 1          # line of "# %% [markdown]"
body = "\n".join(v4[17:end]).rstrip() + "\n\n"
body = body.replace("import os, json, math, random, time, shutil",
                    "import os, json, math, random, time, shutil\nSMOKE = os.environ.get(\"SMOKE\") == \"1\"", 1)
tail_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("notebooks/_v5_tail.py")
tail = tail_path.read_text(encoding="utf-8")
Path("notebooks/_v5_tail.py").write_text(tail, encoding="utf-8")
src = HEAD + "\n" + body + tail
Path("notebooks/cheat_v5_moments.py").write_text(src, encoding="utf-8")

st = Path("outputs/structure/test_videos.json").read_text(encoding="utf-8")
out = Path("kaggle/exam-cheating-v5-moments")
out.mkdir(parents=True, exist_ok=True)
tmp = out / "_build.py"
tmp.write_text(src.replace("__STRUCTURE__", st), encoding="utf-8")
subprocess.run([sys.executable, "tools/py2ipynb.py", str(tmp), str(out / "exam-cheating-v5-moments.ipynb")], check=True)
tmp.unlink()
print("lines:", len(src.splitlines()))
