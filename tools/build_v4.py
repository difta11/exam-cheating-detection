"""Embed outputs/structure/test_videos.json into the v4 notebook and write the Kaggle .ipynb."""
import subprocess, sys
from pathlib import Path
src = Path("notebooks/cheat_video_level.py").read_text(encoding="utf-8")
st = Path("outputs/structure/test_videos.json").read_text(encoding="utf-8")
tmp = Path("kaggle/exam-cheating-video-level/_build.py")
tmp.parent.mkdir(parents=True, exist_ok=True)
tmp.write_text(src.replace("__STRUCTURE__", st), encoding="utf-8")
subprocess.run([sys.executable, "tools/py2ipynb.py", str(tmp), "kaggle/exam-cheating-video-level/exam-cheating-video-level.ipynb"], check=True)
tmp.unlink()
