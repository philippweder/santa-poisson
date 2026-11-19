#!/usr/bin/env bash
# Build CutFEMx 0.2 with runintgen 0.1 and CutCells 0.4 into the active conda env (design §4.3).
#
# Run it from inside the env, for example
#   conda run -n advent-cut --no-capture-output bash environments/cutfemx/build.sh
# It builds on the conda-forge FEniCSx 0.11 stack of environments/advent-cut.yml, which supplies
# the compilers, cmake, ninja, nanobind and scikit-build-core, so no dolfinx is built from source.
#
# CUTFEMX_SRC  where the three repositories are cloned; default: a directory under the env prefix.
#              A re-run with the clones present checks out the pinned commits again and rebuilds.
#
# The bash is 3.2 of macOS: no mapfile, no associative arrays.

set -eu
set -o pipefail

# The CutCells pin is not the v0.4.0 tag (ed5232e): CutFEMx 0.2.0 uses
# cutcells::output::QuadratureBackend, which the tag lacks, and fails to compile against it. The
# pin is the last CutCells commit before the CutFEMx 0.2.0 commit (14 June 2026); it still reports
# version 0.4.0, which is what CutFEMx's pyproject accepts (cutcells>=0.4.0,<0.5.0).
CUTCELLS_URL=https://github.com/sclaus2/CutCells
CUTCELLS_COMMIT=7f461a7490726768be6e96ca3981efd86819b27c # 12 June 2026, after v0.4.0
RUNINTGEN_URL=https://github.com/sclaus2/runintgen
RUNINTGEN_COMMIT=a4e03f27ee1c6b0f130a25c44e89e508bc658cb1 # v0.1.0
CUTFEMX_URL=https://github.com/sclaus2/CutFEMx
CUTFEMX_COMMIT=1d440aa78accb7820abc63da93bb44a221d25685 # v0.2.0

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "build.sh: CONDA_PREFIX is not set; run it inside the env, e.g." >&2
    echo "  conda run -n advent-cut --no-capture-output bash environments/cutfemx/build.sh" >&2
    exit 1
fi

CUTFEMX_SRC="${CUTFEMX_SRC:-$CONDA_PREFIX/src/cutfemx}"
mkdir -p "$CUTFEMX_SRC"

# Clone <url> to $CUTFEMX_SRC/<name>, then put the work tree on <commit>. The build directories
# of an earlier run are removed: their CMake cache keeps the compiler flags it was configured
# with, so a re-run would otherwise ignore the flags set below. Nothing else is removed.
checkout_pinned() {
    name=$1
    url=$2
    commit=$3
    dir="$CUTFEMX_SRC/$name"
    if [ ! -d "$dir/.git" ]; then
        git clone "$url" "$dir"
    elif ! git -C "$dir" cat-file -e "$commit^{commit}" 2> /dev/null; then
        git -C "$dir" fetch origin
    fi
    git -C "$dir" checkout --detach "$commit"
    for stale in build cpp/build python/build; do
        if [ -d "$dir/$stale" ]; then
            rm -r "${dir:?}/$stale"
        fi
    done
}

checkout_pinned CutCells "$CUTCELLS_URL" "$CUTCELLS_COMMIT"
checkout_pinned runintgen "$RUNINTGEN_URL" "$RUNINTGEN_COMMIT"
checkout_pinned CutFEMx "$CUTFEMX_URL" "$CUTFEMX_COMMIT"

# Tie cmake and python discovery to the env, as the CutFEMx README does.
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export PYTHONNOUSERSITE=1
macos_target=13.4 # the deployment target of the CutFEMx README
cutcells_cmake_args=""
if [ "$(uname -s)" = "Darwin" ]; then
    # The CutFEMx README sets the libc++ macros _LIBCPP_ENABLE_EXPERIMENTAL and
    # _LIBCPP_HAS_NO_EXPERIMENTAL_*. conda-forge's libc++ 21 headers ignore them: they mark
    # std::chrono::time_zone, which dolfinx/common/timing.h pulls in, as introduced in macOS 99.0
    # for a deployment target of 13.4. Disabling the availability markup is the remedy of the
    # conda-forge knowledge base (newer-c-features-with-old-sdk).
    export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_OSX_DEPLOYMENT_TARGET=$macos_target"
    export CXXFLAGS="${CXXFLAGS:-} -D_LIBCPP_DISABLE_AVAILABILITY"
    cutcells_cmake_args="-DCMAKE_OSX_DEPLOYMENT_TARGET=$macos_target"
fi

# runintgen: independent of CutCells.
python -m pip install --no-build-isolation --no-deps --force-reinstall \
    "$CUTFEMX_SRC/runintgen"

# CutCells: the C++ core with CMake and Ninja, installed into the env, then the Python bindings.
# CUTCELLS_WITH_ALGOIM stays off on purpose: CutCells cuts triangles and quadrilaterals with its
# own tables (the smoke checks both). Algoim only adds curved-interface quadrature on quad and hex
# cells, worth having for high-order elements, at the cost of a LAPACKE dependency.
cmake -G Ninja -S "$CUTFEMX_SRC/CutCells/cpp" -B "$CUTFEMX_SRC/CutCells/cpp/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    $cutcells_cmake_args
cmake --build "$CUTFEMX_SRC/CutCells/cpp/build"
cmake --install "$CUTFEMX_SRC/CutCells/cpp/build"
python -m pip install --no-build-isolation --no-deps --force-reinstall \
    "$CUTFEMX_SRC/CutCells/python"

# CutFEMx 0.2.0 builds its extension module without the stable ABI. The conda-forge dolfinx for
# Python 3.12 and later ships its extension module with it (cpp.abi3.so). nanobind gives the two
# kinds of module separate type registries, so a CutFEMx module that is not stable-ABI cannot
# take a dolfinx Function, and every call fails with a TypeError. CutFEMx main detects an abi3
# dolfinx and builds with STABLE_ABI; this applies the same two-line change to the pinned commit.
# find_spec does not import dolfinx, so MPI is not initialised for a directory name.
dolfinx_dir=$(python -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("dolfinx").origin))')
if ls "$dolfinx_dir"/cpp*.abi3.so > /dev/null 2>&1; then
    stable_abi_patch=$(mktemp "${TMPDIR:-/tmp}/cutfemx-stable-abi.XXXXXX")
    cat > "$stable_abi_patch" <<'PATCH'
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -43,7 +43,7 @@ if(_cutfemx_conda_prefix)
     "${_cutfemx_conda_prefix}/Library/lib/cmake")
 endif()
 
-find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
+find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
 message(STATUS "Using Python executable: ${Python_EXECUTABLE}")
 if(_cutfemx_conda_prefix)
   message(STATUS "Using active conda prefix: ${_cutfemx_conda_prefix}")
@@ -299,7 +299,7 @@ if(PETSC4PY_INCLUDE_DIR)
     CUTFEMX_HAS_RUNTIME_PETSC
     HAS_PETSC4PY)
 endif()
-nanobind_add_module(cutfemx_cpp ${WRAPPER_SOURCES} NOMINSIZE)
+nanobind_add_module(cutfemx_cpp STABLE_ABI ${WRAPPER_SOURCES} NOMINSIZE)
 target_compile_definitions(cutfemx_cpp PRIVATE ${CUTFEMX_WRAPPER_DEFINITIONS})
 target_include_directories(cutfemx_cpp BEFORE PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/cpp
PATCH
    # A re-run finds the patch applied already: the checkout above keeps local edits.
    if git -C "$CUTFEMX_SRC/CutFEMx" apply --reverse --check "$stable_abi_patch" 2> /dev/null; then
        echo "build.sh: the stable-ABI change is in CutFEMx already"
    else
        git -C "$CUTFEMX_SRC/CutFEMx" apply "$stable_abi_patch"
    fi
    rm "$stable_abi_patch"
fi

# CutFEMx: needs runintgen and CutCells installed.
python -m pip install --no-build-isolation --no-deps --force-reinstall \
    "$CUTFEMX_SRC/CutFEMx"

python - <<'PY'
import importlib.metadata as metadata

import cutcells
import cutfemx
import dolfinx
import runintgen

for name in ["fenics-dolfinx", "runintgen", "cutcells", "cutfemx"]:
    print(name, metadata.version(name))
PY
