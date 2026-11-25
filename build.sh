#!/usr/bin/env bash

# Portable build for the CUDA/OptiX renderer in facet/renderer/native (output: build/renderer).
# Set OPTIX_DIRECTORY and, if needed, CUDA_DIRECTORY, PYTHON, CUDA_ARCH, and CXX.

set -Eeuo pipefail

PROJECT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RENDERER_DIRECTORY="$PROJECT_DIRECTORY/facet/renderer/native"
OUTPUT_DIRECTORY="${RENDERER_OUTPUT_DIRECTORY:-$PROJECT_DIRECTORY/build/renderer}"

CUDA_DIRECTORY="${CUDA_DIRECTORY:-${CUDA_HOME:-/usr/local/cuda}}"
OPTIX_DIRECTORY="${OPTIX_DIRECTORY:?Set OPTIX_DIRECTORY to the OptiX SDK root}"
PYTHON="${PYTHON:-python}"
CUDA_ARCH="${CUDA_ARCH:-89}"
CXX="${CXX:-g++}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-0}"
HIT_BUFFER_SIZE="${HIT_BUFFER_SIZE:-16}"
# Optional bound on the colour a primitive emits (c_const * c_MLP). Unset
# builds the default renderer, identical to a build without this option.
EMITTED_COLOR_MAX="${EMITTED_COLOR_MAX:-}"
# Optional id-biased hit order for rendering lifted 2D models (see shaders.cu):
# sort key t * (1 + ID_ORDER_DELTA * id / 2^19). Unset builds the default
# (t, id) order. ID_ORDER_SPAN (default 2^19 = 524288) replaces 2^19 in the
# key; it must exceed the largest primitive id, or the resumed trace of a
# later hit-buffer round skips hits (shaders.cu). Only with ID_ORDER_DELTA.
ID_ORDER_DELTA="${ID_ORDER_DELTA:-}"
ID_ORDER_SPAN="${ID_ORDER_SPAN:-}"
# Optional (with ID_ORDER_DELTA): measure the id-order band from the source
# camera at the world origin instead of from the ray origin (shaders.cu).
# Unset keeps the ray-origin key.
ID_ORDER_SOURCE="${ID_ORDER_SOURCE:-}"
# Optional warp-level early exit from the backward hit loop once no ray of
# the warp is active. Unset builds the default renderer.
BACKWARD_EARLY_EXIT="${BACKWARD_EARLY_EXIT:-}"
# Optional: with a zero descriptor (conditioning passed with stride 0) skip
# its 96 inputs in the first decoder layer ("forward"; changes tensor-core
# rounding, not bit-exact), its dW1 columns and, when no descriptor gradient
# is requested, its input gradient ("gradients"), or both ("1"). Unset builds
# the default renderer.
SKIP_ZERO_CONDITIONING="${SKIP_ZERO_CONDITIONING:-}"
# Optional: also instantiate the training kernels for a compact descriptor of
# width 4, 8 or 16 (a contiguous N x width tensor instead of the 96D ABI;
# --fast_descriptor): only those inputs are processed in the first decoder
# layer, dW1 and the descriptor gradient (not bit-exact, like
# SKIP_ZERO_CONDITIONING). Unset builds the default renderer.
COMPACT_CONDITIONING="${COMPACT_CONDITIONING:-}"
# Extra nvcc flags for experiments (e.g. timing ablations); empty by default.
EXTRA_NVCC_FLAGS="${EXTRA_NVCC_FLAGS:-}"

if ! [[ "$HIT_BUFFER_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: HIT_BUFFER_SIZE must be a positive integer" >&2
    exit 1
fi

NVCC="$CUDA_DIRECTORY/bin/nvcc"
COMPUTE="compute_$CUDA_ARCH"
SM="sm_$CUDA_ARCH"

if [[ ! -x "$NVCC" ]]; then
    echo "Error: nvcc was not found at $NVCC" >&2
    exit 1
fi
if [[ ! -f "$OPTIX_DIRECTORY/include/optix.h" ]]; then
    echo "Error: OptiX headers were not found in $OPTIX_DIRECTORY/include" >&2
    exit 1
fi

TORCH_DIRECTORY="$("$PYTHON" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)')"
PYTHON_INCLUDE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
TORCH_ABI="$("$PYTHON" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"

CUDA_INCLUDE="$CUDA_DIRECTORY/include"
if [[ -d "$CUDA_DIRECTORY/lib64" ]]; then
    CUDA_LIBRARY="$CUDA_DIRECTORY/lib64"
else
    CUDA_LIBRARY="$CUDA_DIRECTORY/lib"
fi
if [[ -d "$CUDA_LIBRARY/stubs" ]]; then
    CUDA_STUB_LIBRARY="$CUDA_LIBRARY/stubs"
elif [[ -d "$CUDA_DIRECTORY/targets/x86_64-linux/lib/stubs" ]]; then
    CUDA_STUB_LIBRARY="$CUDA_DIRECTORY/targets/x86_64-linux/lib/stubs"
else
    CUDA_STUB_LIBRARY="$CUDA_LIBRARY"
fi

TORCH_INCLUDE="$TORCH_DIRECTORY/include"
TORCH_API_INCLUDE="$TORCH_INCLUDE/torch/csrc/api/include"
TORCH_LIBRARY="$TORCH_DIRECTORY/lib"
OPTIX_INCLUDE="$OPTIX_DIRECTORY/include"

mkdir -p "$OUTPUT_DIRECTORY"

NVCC_FLAGS=(
    -std=c++17
    -Xcompiler -fPIC
    -arch="$COMPUTE"
    -code="$SM"
    -I"$OPTIX_INCLUDE"
    -I"$CUDA_INCLUDE"
    -DFACET_HIT_BUFFER_SIZE="$HIT_BUFFER_SIZE"
)
if [[ -n "$EMITTED_COLOR_MAX" ]]; then
    NVCC_FLAGS+=(-DFACET_EMITTED_COLOR_MAX="${EMITTED_COLOR_MAX}f")
fi
if [[ -n "$BACKWARD_EARLY_EXIT" ]]; then
    NVCC_FLAGS+=(-DFACET_BACKWARD_EARLY_EXIT)
fi
case "$SKIP_ZERO_CONDITIONING" in
    "") ;;
    1) NVCC_FLAGS+=(-DFACET_SKIP_ZERO_CONDITIONING_FORWARD -DFACET_SKIP_ZERO_CONDITIONING_GRADIENTS) ;;
    forward) NVCC_FLAGS+=(-DFACET_SKIP_ZERO_CONDITIONING_FORWARD) ;;
    gradients) NVCC_FLAGS+=(-DFACET_SKIP_ZERO_CONDITIONING_GRADIENTS) ;;
    *) echo "Error: SKIP_ZERO_CONDITIONING must be 1, forward or gradients" >&2; exit 1 ;;
esac
if [[ -n "$COMPACT_CONDITIONING" ]]; then
    NVCC_FLAGS+=(-DFACET_COMPACT_CONDITIONING)
fi
if [[ -n "$EXTRA_NVCC_FLAGS" ]]; then
    read -r -a EXTRA_NVCC_FLAG_LIST <<< "$EXTRA_NVCC_FLAGS"
    NVCC_FLAGS+=("${EXTRA_NVCC_FLAG_LIST[@]}")
fi
CPP_FLAGS=(
    -std=c++17
    -fPIC
    -D_GLIBCXX_USE_CXX11_ABI="$TORCH_ABI"
    -I"$OPTIX_INCLUDE"
    -I"$CUDA_INCLUDE"
    -I"$TORCH_INCLUDE"
    -I"$TORCH_API_INCLUDE"
    -I"$PYTHON_INCLUDE"
    -DFACET_HIT_BUFFER_SIZE="$HIT_BUFFER_SIZE"
)
if [[ -n "$COMPACT_CONDITIONING" ]]; then
    CPP_FLAGS+=(-DFACET_COMPACT_CONDITIONING)
fi

echo "Building renderer with CUDA at $CUDA_DIRECTORY"
echo "OptiX SDK: $OPTIX_DIRECTORY"
echo "Python: $PYTHON"
echo "CUDA target: $COMPUTE / $SM"
echo "Hit buffer size: $HIT_BUFFER_SIZE"
echo "Emitted colour bound: ${EMITTED_COLOR_MAX:-none}"
echo "Id-biased hit order: ${ID_ORDER_DELTA:-off}${ID_ORDER_SOURCE:+ (source-camera band)}"
echo "Backward early exit: ${BACKWARD_EARLY_EXIT:-off}"
echo "Skip zero descriptor: ${SKIP_ZERO_CONDITIONING:-off}"
echo "Compact descriptor: ${COMPACT_CONDITIONING:-off}"
echo "Extra nvcc flags: ${EXTRA_NVCC_FLAGS:-none}"
PTX_FLAGS=()
if [[ -n "$ID_ORDER_DELTA" ]]; then
    PTX_FLAGS+=(-DFACET_ID_ORDER_DELTA="${ID_ORDER_DELTA}f")
    if [[ -n "$ID_ORDER_SPAN" ]]; then
        PTX_FLAGS+=(-DFACET_ID_ORDER_SPAN="${ID_ORDER_SPAN}.0f")
    fi
    if [[ -n "$ID_ORDER_SOURCE" ]]; then
        PTX_FLAGS+=(-DFACET_ID_ORDER_SOURCE)
    fi
fi

"$NVCC" -std=c++17 -ptx -arch="$COMPUTE" \
    "$RENDERER_DIRECTORY/shaders.cu" \
    -o "$OUTPUT_DIRECTORY/shaders.cu.ptx" \
    -I"$OPTIX_INCLUDE" \
    -DFACET_HIT_BUFFER_SIZE="$HIT_BUFFER_SIZE" "${PTX_FLAGS[@]}"

CUDA_SOURCES=(
    CPyOptiXFACETRenderer_CUDA_Backward
    CPyOptiXFACETRenderer_CUDA_Backward_base
    CPyOptiXFACETRenderer_CUDA_Core
    CPyOptiXFACETRenderer_CUDA_Forward_training
    CPyOptiXFACETRenderer_CUDA_Forward_training_base
    CPyOptiXFACETRenderer_CUDA_Forward_inference
    CPyOptiXFACETRenderer_CUDA_Forward_inference_base
    CPyOptiXFACETRenderer_CUDA_GetMedianDepth
    CPyOptiXFACETRenderer_CUDA_GetMedianDepth_base
    GenerateRays
)

CUDA_OBJECTS=()
for source_name in "${CUDA_SOURCES[@]}"; do
    object="$OUTPUT_DIRECTORY/${source_name}.cu.o"
    "$NVCC" "${NVCC_FLAGS[@]}" \
        -c "$RENDERER_DIRECTORY/${source_name}.cu" \
        -o "$object"
    CUDA_OBJECTS+=("$object")
done

"$NVCC" -std=c++17 -Xcompiler -fPIC -dlink \
    "${CUDA_OBJECTS[@]}" \
    -o "$OUTPUT_DIRECTORY/dlink.o" \
    -arch="$COMPUTE" -code="$SM" \
    -L"$CUDA_STUB_LIBRARY" \
    -L"$CUDA_LIBRARY" \
    -lcuda -lcudart

"$CXX" "${CPP_FLAGS[@]}" \
    -c "$RENDERER_DIRECTORY/GenerateRays.cpp" \
    -o "$OUTPUT_DIRECTORY/GenerateRays.cpp.o"
"$CXX" "${CPP_FLAGS[@]}" \
    -c "$RENDERER_DIRECTORY/CPyOptiXFACETRenderer.cpp" \
    -o "$OUTPUT_DIRECTORY/CPyOptiXFACETRenderer.cpp.o"

"$CXX" "${CPP_FLAGS[@]}" -shared \
    "$RENDERER_DIRECTORY/PyOptiXFACETRenderer.cpp" \
    "${CUDA_OBJECTS[@]}" \
    "$OUTPUT_DIRECTORY/dlink.o" \
    "$OUTPUT_DIRECTORY/GenerateRays.cpp.o" \
    "$OUTPUT_DIRECTORY/CPyOptiXFACETRenderer.cpp.o" \
    -o "$OUTPUT_DIRECTORY/PYOPTIXFACETRENDERER.so" \
    -L"$CUDA_STUB_LIBRARY" \
    -L"$CUDA_LIBRARY" \
    -L"$TORCH_LIBRARY" \
    -lcuda -lcudart -ldl -ltorch_python -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,"$CUDA_LIBRARY" \
    -Wl,-rpath,"$TORCH_LIBRARY"

echo "Build completed: $OUTPUT_DIRECTORY/PYOPTIXFACETRENDERER.so"

if [[ "$RUN_SMOKE_TEST" == "1" ]]; then
    "$PYTHON" "$PROJECT_DIRECTORY/scripts/check_renderer.py"
fi
