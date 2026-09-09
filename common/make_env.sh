#!/usr/bin/env bash
#
# Shared virtual-environment builder, called by each model's `setup.sh`.
# You should not need to run this directly.
#
# It does what the per-model READMEs tell you to do by hand, but it finds a
# usable interpreter first instead of assuming one exists, and it stops at the
# first failure instead of carrying on into the next step. Both of those are
# real failure modes: hardcoded `python3.12` is absent on many machines, and an
# unchained sequence installs into whatever environment happens to be active.
#
# Usage:  make_env.sh <kernel-name> <kernel-display-name> [required-GB] [pip flags...]
# Run from the directory holding requirements.txt / requirements.lock.

set -euo pipefail

KERNEL_NAME="${1:?kernel name required}"
KERNEL_DISPLAY="${2:?kernel display name required}"
REQUIRED_GB="${3:-0}"
shift 3 || shift $#
PIP_FLAGS=("$@")

VENV_DIR="env"
UV_INSTALL_HINT='curl -LsSf https://astral.sh/uv/install.sh | sh'

# --- keep the big downloads out of $HOME -------------------------------------
# uv puts the interpreters it installs in ~/.local/share/uv, and both uv and pip
# cache wheels under ~/.cache. On a managed cluster $HOME is typically a small,
# quota'd NFS volume, and a per-user quota is invisible to `df` -- it reports
# the filesystem's free space, not what is left of your allowance -- so the
# check further down passes and the install then dies partway through with
# "Disk quota exceeded". Default all three to the repository's own filesystem,
# which is where the multi-GB environment is going anyway. An explicitly set
# value always wins, so this can still be pointed at scratch.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$REPO_ROOT/.uv/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv/cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/.uv/pip-cache}"

# --- what to install ---------------------------------------------------------
# The lock file is the fully resolved set, including transitive dependencies;
# requirements.txt leaves those to the resolver and so drifts over time.
if [ -f requirements.lock ]; then
    REQ_FILE="requirements.lock"
else
    REQ_FILE="requirements.txt"
fi
if [ ! -f "$REQ_FILE" ]; then
    echo "error: no requirements.lock or requirements.txt in $(pwd)." >&2
    echo "       Run this from a model folder, via its own setup.sh." >&2
    exit 1
fi

if [ -e "$VENV_DIR" ]; then
    echo "error: '$VENV_DIR' already exists in $(pwd)." >&2
    echo "       To rebuild it:      rm -rf $VENV_DIR && ./setup.sh" >&2
    echo "       To just use it:     source $VENV_DIR/bin/activate" >&2
    exit 1
fi

# --- free space --------------------------------------------------------------
# Note this is filesystem free space. If your account is under a quota, the
# number below can look fine while your own allowance is already spent -- see
# the $HOME note above, and check with `quota -s` (or `lfs quota -h .` on
# Lustre) if an install fails with "Disk quota exceeded".
if [ "$REQUIRED_GB" != "0" ]; then
    free_gb=$(df -Pk . | awk 'NR==2 {print int($4/1024/1024)}')
    if [ -n "$free_gb" ] && [ "$free_gb" -lt "$REQUIRED_GB" ]; then
        echo "warning: this install needs about ${REQUIRED_GB}GB and $(pwd) has ${free_gb}GB free." >&2
        echo "         Continuing anyway -- press Ctrl-C now if that is a problem." >&2
        sleep 5
    fi
fi

# --- find an interpreter -----------------------------------------------------
# 3.11 and 3.12 are both validated. Anything older fails: several pins require
# 3.11 or newer. Newer than 3.12 has no wheels for some of them.
PY=""
for candidate in python3.12 python3.11 python3 python; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)' 2>/dev/null; then
        PY="$candidate"
        break
    fi
done

USING_UV=0
if [ -z "$PY" ]; then
    echo "No Python 3.11 or 3.12 found on PATH."
    if command -v uv >/dev/null 2>&1; then
        echo "Found uv -- using it to install a private Python 3.12 (no root needed)."
        echo "  interpreter -> $UV_PYTHON_INSTALL_DIR"
        echo "  cache       -> $UV_CACHE_DIR"
        if ! uv python install 3.12; then
            cat >&2 <<EOF

error: uv could not install Python 3.12.

If that failed with "Disk quota exceeded", the target above is on a volume
where you are out of allowance. Point it somewhere with room and re-run:

    export UV_PYTHON_INSTALL_DIR=/path/with/space/uv-python
    export UV_CACHE_DIR=/path/with/space/uv-cache
    ./setup.sh

Check what you have left with \`quota -s\`, and clear any half-extracted
leftovers with \`rm -rf "$UV_PYTHON_INSTALL_DIR/.temp"\`.

EOF
            exit 1
        fi
        uv venv --python 3.12 "$VENV_DIR"
        USING_UV=1
    else
        cat >&2 <<EOF

This environment needs Python 3.11 or 3.12, and neither is on your PATH.

  * If your machine has one under another name, put it on PATH and re-run,
    or create the environment by hand following this folder's README.
  * On a cluster, try:            module avail python
  * Otherwise, uv can install one without root:

        $UV_INSTALL_HINT

    then re-run ./setup.sh -- it will pick uv up automatically.

EOF
        exit 1
    fi
else
    echo "Using $PY ($("$PY" --version 2>&1))"
    if ! "$PY" -m venv "$VENV_DIR" 2>/dev/null; then
        echo "error: '$PY -m venv' failed." >&2
        echo "       On Debian/Ubuntu the venv module ships separately:" >&2
        echo "           sudo apt install python3-venv" >&2
        echo "       Or install uv ($UV_INSTALL_HINT) and re-run ./setup.sh." >&2
        rm -rf "$VENV_DIR"
        exit 1
    fi
fi

# --- install -----------------------------------------------------------------
echo "Installing from $REQ_FILE ..."
if [ "$USING_UV" -eq 1 ]; then
    # A uv-created venv has no pip of its own, so install through uv. The
    # index-strategy flag only makes uv search both indexes the way pip
    # already does -- GraphCast's torch pin lives on the second one.
    uv pip install --python "$VENV_DIR/bin/python" --index-strategy unsafe-best-match \
        ${PIP_FLAGS[@]+"${PIP_FLAGS[@]}"} -r "$REQ_FILE"
else
    "$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
    "$VENV_DIR/bin/python" -m pip install ${PIP_FLAGS[@]+"${PIP_FLAGS[@]}"} -r "$REQ_FILE"
fi

# --- Jupyter kernel ----------------------------------------------------------
"$VENV_DIR/bin/python" -m ipykernel install --user \
    --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"

cat <<EOF

Done. Jupyter kernel "$KERNEL_DISPLAY" is registered.

  Open the notebook in this folder and select that kernel, or work in a shell:

      source $(pwd)/$VENV_DIR/bin/activate

EOF
