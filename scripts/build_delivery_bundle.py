"""Build a source-only delivery archive with a per-file SHA256 manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = []
    for directory in ("agentic_rag_vlm", "scripts", "tests", "configs", "knowledge_base", "docs"):
        for path in (ROOT / directory).rglob("*"):
            if (path.is_file() and not path.is_symlink()
                    and "__pycache__" not in path.parts
                    and (path.suffix in {".py", ".json", ".md", ".sh"}
                         or directory == "knowledge_base" and path.suffix in {".png", ".jsonl"})
                    and ".local." not in path.name):
                files.append(path)
    files += [ROOT / name for name in ("README.md", "README.zh-CN.md", "DELIVERY_README.md", "LICENSE", "pyproject.toml", "requirements.txt")]
    files += [path for path in [ROOT / "configs/extended_sim_dependencies.txt"] if path.is_file()]
    files += [ROOT/"paper/main_final.tex",ROOT/"paper/README.md"]
    # A repository source archive also retains the final paper dependencies
    # and the small, explicitly labelled demo/verification examples.
    files += [p for name in ("main_final.pdf","main_final.bbl","references.bib","ieeeconf.cls")
              if (p := ROOT/"paper"/name).is_file()]
    files += [p for p in (ROOT/"paper/figures").glob("fig.*.png") if p.is_file()]
    files += [p for p in (ROOT/"examples").rglob("*") if p.is_file()
              and not p.is_symlink() and p.suffix in {".md",".json",".mp4"}]
    # Licensed robot source assets are required to reproduce contact dynamics.
    files += [path for path in (ROOT/"assets/yam_mjlab").rglob("*")
              if path.is_file() and not path.is_symlink()
              and (path.suffix in {".xml", ".stl", ".md"} or path.name == "LICENSE")]
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "x", compression=ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(ROOT)))
        archive.writestr("SOURCE_SHA256.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"archive": str(args.output), "files": len(manifest),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
