#!/bin/sh
set -eu
cd "$(dirname "$0")"
source_revision=eafdb809426b13153aa1e4c42d6601844217dfec
data_revision=c86fc3b8293185a6f373677018ff3e37f8391602
if [ ! -d source/.git ]; then
    git init -q source
fi
if ! git -C source rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C source fetch --depth 1 https://github.com/Zxy-MLlab/LIBERO-PRO.git "$source_revision"
    git -C source checkout --detach FETCH_HEAD
fi
if [ "$(git -C source rev-parse HEAD)" != "$source_revision" ]; then
    printf 'Expected LIBERO-PRO source at %s\n' "$source_revision" >&2
    exit 1
fi
uv sync --locked --python 3.11
root="$(pwd -P)"
cd source/libero/libero
while read -r _ path; do
    if [ ! -f "$path" ]; then
        mkdir -p "$(dirname "$path")"
        curl -fsSL --retry 2 "https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro/resolve/$data_revision/$path" -o "$path.download"
        mv "$path.download" "$path"
    fi
done < "$root/assets.sha256"
shasum -a 256 -c "$root/assets.sha256"
cd "$root"
mkdir -p config
.venv/bin/python - <<'PY'
from pathlib import Path
from site import getsitepackages

import yaml

source = Path.cwd() / "source"
(Path(getsitepackages()[0]) / "worldsmith-libero-source.pth").write_text(f"{source}\n")
root = source / "libero/libero"
paths = {
    "benchmark_root": root,
    "bddl_files": root / "bddl_files",
    "init_states": root / "init_files",
    "assets": root / "assets",
}
Path("config/config.yaml").write_text(yaml.safe_dump({key: str(path) for key, path in paths.items()}))
PY
