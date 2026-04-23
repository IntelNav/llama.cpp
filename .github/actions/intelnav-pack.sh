#!/usr/bin/env bash
#
# intelnav-pack.sh <backend-tag>
#
# Package a build of the IntelNav-patched libllama into the tarball
# layout the Rust FFI crate (intelnav-ggml) expects. The FFI's future
# dlopen-based runtime loader reads the INTELNAV_SHA file to key its
# on-disk cache under ~/.cache/intelnav/libllama/<sha>/<backend>/.
#
# Expected cwd: the repo root (where `build/` and `include/` live).

set -euo pipefail

BACKEND="${1:?usage: intelnav-pack.sh <backend-tag>}"

if [[ ! -d build/bin ]]; then
    echo "intelnav-pack: build/bin missing — did cmake --build run?" >&2
    exit 1
fi

SHA="$(git rev-parse HEAD)"
SHORT_SHA="${SHA:0:12}"
DIST="dist"
PKG_NAME="libllama-${BACKEND}-${SHORT_SHA}"
PKG_ROOT="${DIST}/${PKG_NAME}"

rm -rf "${DIST}"
mkdir -p "${PKG_ROOT}/bin" "${PKG_ROOT}/include"

# --- shared libraries ----------------------------------------------------
# Copy every .so / .dylib / .dll in build/bin. ggml ships multiple libs
# per build (base, cpu, backend-specific) and their exact set varies
# across cmake configurations, so we glob rather than name them.
shopt -s nullglob
for f in build/bin/*.so build/bin/*.so.* build/bin/*.dylib build/bin/*.dll; do
    cp -a "$f" "${PKG_ROOT}/bin/"
done
shopt -u nullglob

# --- headers -------------------------------------------------------------
# The FFI crate's shim/intelnav_shim.c #includes "llama.h" which transitively
# pulls ggml.h. Consumers that want to bind ggml directly get the full set.
cp include/llama.h "${PKG_ROOT}/include/"
cp -r ggml/include/. "${PKG_ROOT}/include/"

# --- metadata ------------------------------------------------------------
cp LICENSE "${PKG_ROOT}/"
echo "${SHA}" > "${PKG_ROOT}/INTELNAV_SHA"
cat > "${PKG_ROOT}/README.md" <<EOF
# libllama — IntelNav fork

Prebuilt against commit \`${SHA}\` of
\`https://github.com/IntelNav/llama.cpp\`.

Backend: **${BACKEND}**

Ship as part of the IntelNav runtime. Users should not run these binaries
directly — the \`intelnav-ggml\` Rust crate loads them via \`dlopen\` from
\`~/.cache/intelnav/libllama/<sha>/<backend>/\`. Bundled for convenience:

* \`bin/libllama.so\` — the IntelNav-patched libllama (new functions:
  \`llama_embed_only\`, \`llama_decode_layers\`, \`llama_head_only\`).
* \`bin/libggml*.so\` — the ggml compute backends this build links against.
* \`include/\` — public C headers, matching the above SHA.
* \`INTELNAV_SHA\` — read by the FFI crate to key its cache path.
EOF

# --- tarball -------------------------------------------------------------
tar -czf "${DIST}/${PKG_NAME}.tar.gz" -C "${DIST}" "${PKG_NAME}"
rm -rf "${PKG_ROOT}"

ls -la "${DIST}"
echo "intelnav-pack: wrote ${DIST}/${PKG_NAME}.tar.gz"
