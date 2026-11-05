"""Default paper-mechanism entry: actual VLM + CLIP + continuous YAM tasks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_yam_physics import main

if __name__=="__main__":
    if "--paper-store" not in sys.argv:
        sys.argv += ["--paper-store",str(ROOT/"knowledge_base/yam_observed_clip.json")]
    if "--vlm" not in sys.argv:sys.argv.append("--vlm")
    if "--multiple" not in sys.argv:sys.argv.append("--multiple")
    raise SystemExit(main())
