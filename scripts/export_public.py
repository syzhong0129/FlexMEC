"""Build a public source ZIP using an explicit whitelist, without publishing it."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = [ROOT / name for name in ["README.md", ".gitignore", "pyproject.toml", "requirements.txt"]]
    patterns = {"flexmec": {".py"}, "scripts": {".py"}, "tests": {".py", ".json"},
                "configs": {".example"}}
    for directory, suffixes in patterns.items():
        selected += [p for p in (ROOT / directory).rglob("*") if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts]
    manifest = {}
    for path in sorted(selected):
        text = path.read_text()
        # Reject machine-specific paths and obvious credentials, rather than
        # relying on .gitignore (which does not remove already tracked files).
        if re.search(r"/(?:data/home|data/YuLab|pool1|mnt/wzw)/", text):
            raise SystemExit(f"Local absolute path in export: {path.relative_to(ROOT)}")
        if re.search(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----|\bghp_[A-Za-z0-9]{30,}", text):
            raise SystemExit(f"Possible secret in export: {path.relative_to(ROOT)}")
        manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            archive.write(path, "flexmec/" + str(path.relative_to(ROOT)))
        archive.writestr("flexmec/PUBLIC_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(selected)} source/documentation files: {args.output}")


if __name__ == "__main__":
    main()
