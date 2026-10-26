#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_SOURCE_ROOT="${REPOSITORY_ROOT}/src"
WEBSHOP_ROOT="${WEBSHOP_ROOT:-${REPOSITORY_ROOT}/external/WebShop}"
WEBSHOP_SOURCE_COMMIT=64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
WEBSHOP_DATA_REVISION=ce990fff5aee388db2706f07820c578ab68e0453
WEBSHOP_DATA_BASE_URL="${WEBSHOP_DATA_BASE_URL:-https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/${WEBSHOP_DATA_REVISION}}"
export PYTHONNOUSERSITE=1

if [[ "${1:-}" == "--print-contract" ]]; then
  PYTHONPATH="${SDK_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python - <<'PY'
import json

from vrlbench.environments.webshop_dependency import webshop_dependency_contract

print(json.dumps(webshop_dependency_contract(), sort_keys=True))
PY
  exit 0
fi

for command_name in git curl python java; do
  command -v "$command_name" >/dev/null
done

python - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        "WebShop requires Python 3.10; "
        f"got {sys.version_info.major}.{sys.version_info.minor}"
    )
if not sys.flags.no_user_site:
    raise SystemExit("WebShop requires PYTHONNOUSERSITE=1")
PY

java_version="$(java -version 2>&1 | head -n 1)"
if [[ ! "$java_version" =~ version\ \"11\. ]]; then
  echo "WebShop requires Java 11; got: $java_version" >&2
  exit 2
fi

python -m pip install \
  -r "${SDK_SOURCE_ROOT}/vrlbench/requirements/webshop-py310.lock.txt"

if [[ -e "$WEBSHOP_ROOT" && ! -d "$WEBSHOP_ROOT/.git" ]]; then
  echo "WEBSHOP_ROOT exists but is not a Git checkout: $WEBSHOP_ROOT" >&2
  exit 2
fi
if [[ ! -d "$WEBSHOP_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$WEBSHOP_ROOT")"
  git clone https://github.com/princeton-nlp/webshop "$WEBSHOP_ROOT"
fi
if [[ -n "$(git -C "$WEBSHOP_ROOT" status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing to change a WebShop checkout with tracked modifications." >&2
  exit 2
fi
if ! git -C "$WEBSHOP_ROOT" cat-file \
  -e "${WEBSHOP_SOURCE_COMMIT}^{commit}" 2>/dev/null; then
  git -C "$WEBSHOP_ROOT" fetch --quiet origin "$WEBSHOP_SOURCE_COMMIT"
fi
git -C "$WEBSHOP_ROOT" checkout --quiet --detach "$WEBSHOP_SOURCE_COMMIT"
if [[ "$(git -C "$WEBSHOP_ROOT" rev-parse HEAD)" != "$WEBSHOP_SOURCE_COMMIT" ]]; then
  echo "WebShop checkout did not resolve to the required commit." >&2
  exit 2
fi

sha256_file() {
  python - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

digest = hashlib.sha256()
with Path(sys.argv[1]).open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

download_data() {
  local filename="$1"
  local expected_sha256="$2"
  local destination="${WEBSHOP_ROOT}/data/${filename}"
  local partial="${destination}.partial"
  local observed_sha256=""

  if [[ -f "$destination" ]]; then
    observed_sha256="$(sha256_file "$destination")"
    if [[ "$observed_sha256" == "$expected_sha256" ]]; then
      return 0
    fi
    echo "WebShop data hash mismatch: $destination" >&2
    exit 2
  fi

  mkdir -p "${WEBSHOP_ROOT}/data"
  if [[ -f "$partial" ]]; then
    observed_sha256="$(sha256_file "$partial")"
    if [[ "$observed_sha256" != "$expected_sha256" ]]; then
      rm -f "$partial"
    fi
  fi
  if [[ ! -f "$partial" ]]; then
    curl --fail --location --retry 5 --continue-at - \
      --output "$partial" \
      "${WEBSHOP_DATA_BASE_URL}/${filename}?download=true"
  fi
  observed_sha256="$(sha256_file "$partial")"
  if [[ "$observed_sha256" != "$expected_sha256" ]]; then
    echo "Downloaded WebShop data hash mismatch: $filename" >&2
    exit 2
  fi
  mv "$partial" "$destination"
}

download_data \
  items_shuffle.json \
  2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8
download_data \
  items_ins_v2.json \
  1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e
download_data \
  items_human_ins.json \
  cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3

PYTHONPATH="${SDK_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m vrlbench.environments.webshop_dependency \
  --root "$WEBSHOP_ROOT" \
  --phase pre-index

index_directory="${WEBSHOP_ROOT}/search_engine/indexes"
index_marker="${index_directory}/vex2_full_index_meta.json"
if [[ -f "$index_marker" ]]; then
  PYTHONPATH="${SDK_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m vrlbench.environments.webshop_dependency \
    --root "$WEBSHOP_ROOT" \
    --phase post-index
else
  if [[ -d "$index_directory" ]] \
    && find "$index_directory" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing an unmarked WebShop index directory: $index_directory" >&2
    exit 2
  fi
  mkdir -p \
    "${WEBSHOP_ROOT}/search_engine/resources" \
    "${WEBSHOP_ROOT}/search_engine/resources_100" \
    "${WEBSHOP_ROOT}/search_engine/resources_1k" \
    "${WEBSHOP_ROOT}/search_engine/resources_100k"
  (
    cd "${WEBSHOP_ROOT}/search_engine"
    PYTHONPATH="$WEBSHOP_ROOT" python - <<'PY'
import runpy
from pathlib import Path

root = Path.cwd().parent
import web_agent_site.utils as webshop_utils
from web_agent_site.engine import engine as webshop_engine

webshop_utils.DEFAULT_FILE_PATH = str(root / "data" / "items_shuffle.json")
webshop_utils.DEFAULT_ATTR_PATH = str(root / "data" / "items_ins_v2.json")
webshop_engine.DEFAULT_ATTR_PATH = webshop_utils.DEFAULT_ATTR_PATH
runpy.run_path("convert_product_file_format.py", run_name="__main__")
PY
    python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input resources \
      --index indexes \
      --generator DefaultLuceneDocumentGenerator \
      --threads 1 \
      --storePositions --storeDocvectors --storeRaw
  )
  converted_documents="$(
    wc -l < "${WEBSHOP_ROOT}/search_engine/resources/documents.jsonl"
  )"
  if [[ "$converted_documents" != "1181430" ]]; then
    echo "WebShop conversion produced $converted_documents documents." >&2
    exit 2
  fi
  PYTHONPATH="${SDK_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python - "$WEBSHOP_ROOT" <<'PY'
import sys
from pathlib import Path

from vrlbench.environments.webshop_dependency import write_webshop_full_index_marker

write_webshop_full_index_marker(Path(sys.argv[1]))
PY
  PYTHONPATH="${SDK_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m vrlbench.environments.webshop_dependency \
    --root "$WEBSHOP_ROOT" \
    --phase post-index
fi

printf 'export WEBSHOP_ROOT=%q\n' "$WEBSHOP_ROOT"
printf 'export PYTHONNOUSERSITE=1\n'
