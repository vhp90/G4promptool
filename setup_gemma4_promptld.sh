#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Gemma4 PromptLD — Auto Setup for Lightning.ai Studio
# by vhp (Lightning.ai adaptation)
#
# Lightning.ai notes:
#   • Persistent storage lives at /teamspace/studios/this_studio/
#   • Home directory is /home/zeus/
#   • GPU is available, CUDA pre-installed
#   • Python env is pre-configured (use pip directly)
#   • For persistent installs, store in /teamspace/studios/this_studio/
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

AUTO_DOWNLOAD="${G4PROMPTOOL_AUTO_DOWNLOAD:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes|--download)
            AUTO_DOWNLOAD="y"
            shift
            ;;
        --no-download|--skip-download)
            AUTO_DOWNLOAD="n"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--yes|--download] [--no-download|--skip-download]"
            echo "  --yes / --download    Automatically download missing GGUF + mmproj files"
            echo "  --no-download         Skip missing model downloads without prompting"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ── ANSI COLOURS ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

first_match() {
    find "$@" -print -quit 2>/dev/null
}

install_python_packages() {
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python python3 "$@"
    else
        pip install "$@"
    fi
}

download_prebuilt_llama() {
    echo -e "${YELLOW}⚠  llama-server not found. Attempting prebuilt download to ${LLAMA_DIR}...${NC}"
    echo ""
    echo "URL: ${LLAMA_URL}"
    echo ""

    mkdir -p "${LLAMA_DIR}"

    if ! curl -fL --progress-bar -o "${LLAMA_ARCHIVE}" "${LLAMA_URL}"; then
        echo -e "${YELLOW}⚠ Prebuilt download failed or no longer exists at that URL.${NC}"
        return 1
    fi

    echo ""
    echo "Extracting..."

    if ! tar -tzf "${LLAMA_ARCHIVE}" >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠ Downloaded file is not a valid tar.gz archive.${NC}"
        rm -f "${LLAMA_ARCHIVE}"
        return 1
    fi

    if ! tar -xzf "${LLAMA_ARCHIVE}" -C "${LLAMA_DIR}" --strip-components=1 2>/dev/null; then
        if ! tar -xzf "${LLAMA_ARCHIVE}" -C "${LLAMA_DIR}" 2>/dev/null; then
            echo -e "${YELLOW}⚠ Failed to extract the prebuilt archive.${NC}"
            rm -f "${LLAMA_ARCHIVE}"
            return 1
        fi
    fi

    rm -f "${LLAMA_ARCHIVE}"

    if [ ! -f "${LLAMA_EXE}" ]; then
        FOUND_EXE=$(first_match "${LLAMA_DIR}" -type f -name "llama-server" || true)
        if [ -n "${FOUND_EXE}" ] && [ "${FOUND_EXE}" != "${LLAMA_EXE}" ]; then
            mv "${FOUND_EXE}" "${LLAMA_EXE}"
        fi
    fi

    if [ -f "${LLAMA_EXE}" ]; then
        chmod +x "${LLAMA_EXE}"
        echo -e "${GREEN}✅ llama-server installed from prebuilt archive at ${LLAMA_EXE}${NC}"
        return 0
    fi

    echo -e "${YELLOW}⚠ Prebuilt archive extracted, but llama-server was not found inside it.${NC}"
    return 1
}

build_llama_from_source() {
    echo ""
    echo -e "${CYAN}Falling back to building llama.cpp from source with CUDA support...${NC}"

    if ! command -v git >/dev/null 2>&1; then
        echo -e "${RED}❌ git is required to build llama.cpp from source.${NC}"
        exit 1
    fi

    if ! command -v cmake >/dev/null 2>&1; then
        echo -e "${RED}❌ cmake is required to build llama.cpp from source.${NC}"
        exit 1
    fi

    mkdir -p "${STUDIO_BASE}"

    if [ -d "${LLAMA_SRC_DIR}/.git" ]; then
        echo "Updating existing llama.cpp checkout..."
        git -C "${LLAMA_SRC_DIR}" fetch --depth 1 origin master
        git -C "${LLAMA_SRC_DIR}" reset --hard FETCH_HEAD
    else
        echo "Cloning llama.cpp source..."
        rm -rf "${LLAMA_SRC_DIR}"
        git clone --depth 1 "${LLAMA_REPO_URL}" "${LLAMA_SRC_DIR}"
    fi

    CUDA_FLAG="OFF"
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_FLAG="ON"
        echo -e "${GREEN}✅ nvcc detected — building with CUDA support.${NC}"
    else
        echo -e "${YELLOW}⚠ nvcc not found. Building CPU-only llama-server instead.${NC}"
    fi

    cmake -S "${LLAMA_SRC_DIR}" -B "${LLAMA_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA="${CUDA_FLAG}" \
        -DLLAMA_CURL=ON

    cmake --build "${LLAMA_BUILD_DIR}" --config Release --target llama-server -j"$(nproc)"

    if [ ! -f "${LLAMA_BUILD_DIR}/bin/llama-server" ]; then
        echo -e "${RED}❌ Source build finished, but build/bin/llama-server was not created.${NC}"
        exit 1
    fi

    mkdir -p "${LLAMA_DIR}"
    cp -a "${LLAMA_BUILD_DIR}/bin/." "${LLAMA_DIR}/"
    chmod +x "${LLAMA_DIR}/llama-server" 2>/dev/null || true

    echo -e "${GREEN}✅ llama-server built from source and copied to ${LLAMA_DIR}${NC}"
}

echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║       Gemma4 PromptLD — Auto Setup (Lightning.ai)   ║${NC}"
echo -e "${CYAN}${BOLD}║       by      vhp                                   ║${NC}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# ── CONFIG ──────────────────────────────────────────────────────────────────
# Lightning.ai persistent storage base
STUDIO_BASE="/teamspace/studios/this_studio"

LLAMA_DIR="${STUDIO_BASE}/llama"
MODELS_DIR="${STUDIO_BASE}/ComfyUI/models/LLM"
LLAMA_EXE="${LLAMA_DIR}/llama-server"
LLAMA_SRC_DIR="${STUDIO_BASE}/llama.cpp-src"
LLAMA_BUILD_DIR="${LLAMA_SRC_DIR}/build"

# Historical pinned binary URL. If it disappears, fall back to source build.
LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/b8664/llama-b8664-bin-ubuntu-x64-cuda-cu12.4.tar.gz"
LLAMA_ARCHIVE="${LLAMA_DIR}/llama_install.tar.gz"
LLAMA_REPO_URL="https://github.com/ggml-org/llama.cpp.git"

CONFIG_FILE="$(dirname "$0")/model_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ model_config.json not found in $(dirname "$0"). Please create it.${NC}"
    exit 1
fi

HF_REPO=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('hf_repo', ''))")
GGUF_FILE_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('gguf_filename', ''))")
MMPROJ_FILE_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('mmproj_filename', ''))")
BUILD_WORKERS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('build_max_workers', False))")

if [ "$BUILD_WORKERS" = "True" ] || [ "$BUILD_WORKERS" = "true" ]; then
    export MAKEFLAGS="-j$(nproc)"
    export MAX_JOBS=$(nproc)
    export NINJA_MAX_JOBS=$(nproc)
    export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
    echo -e "${CYAN}⚡ Max workers enabled for potential pip/wheel builds ($(nproc) cores)${NC}"
fi

GGUF_URL="https://huggingface.co/${HF_REPO}/resolve/main/${GGUF_FILE_NAME}"
GGUF_FILE="${MODELS_DIR}/${GGUF_FILE_NAME}"
MMPROJ_URL="https://huggingface.co/${HF_REPO}/resolve/main/${MMPROJ_FILE_NAME}"
MMPROJ_FILE="${MODELS_DIR}/${MMPROJ_FILE_NAME}"
# ────────────────────────────────────────────────────────────────────────────

download_hf_asset() {
    local repo="$1"
    local filename="$2"
    local target_dir="$3"
    local target_file="$4"
    local url="$5"
    local label="$6"
    local token_arg="$7"

    if [ -f "${target_file}" ]; then
        echo -e "${GREEN}✅ ${label} already present: $(basename "${target_file}")${NC}"
        return 0
    fi

    if command -v hf >/dev/null 2>&1; then
        if hf download "${repo}" "${filename}" --local-dir "${target_dir}" ${token_arg}; then
            echo -e "${GREEN}✅ ${label} downloaded via High-Speed HF CLI.${NC}"
            return 0
        fi
        echo -e "${YELLOW}⚠ HF CLI failed for ${label}. Falling back to curl...${NC}"
    fi

    if [ -n "${HF_TOKEN:-}" ]; then
        if ! curl -H "Authorization: Bearer ${HF_TOKEN}" -L --progress-bar -o "${target_file}" "${url}"; then
            return 1
        fi
    else
        if ! curl -L --progress-bar -o "${target_file}" "${url}"; then
            return 1
        fi
    fi

    echo -e "${GREEN}✅ ${label} downloaded via fallback.${NC}"
    return 0
}

# ── STEP 1/4: Install llama-server ──────────────────────────────────────────
echo -e "${BOLD}[STEP 1/4] Checking llama-server...${NC}"
echo ""

# Check if already in PATH
if command -v llama-server &>/dev/null; then
    echo -e "${GREEN}✅ llama-server found in PATH — skipping install.${NC}"
elif [ -f "${LLAMA_EXE}" ]; then
    echo -e "${GREEN}✅ llama-server found at ${LLAMA_EXE} — skipping install.${NC}"
else
    if ! download_prebuilt_llama; then
        build_llama_from_source
    fi
fi

# Also make all binaries in the llama dir executable
find "${LLAMA_DIR}" \( -type f -executable -o -type f -name "llama-*" \) 2>/dev/null | while read -r f; do
    chmod +x "$f" 2>/dev/null || true
done

# Add to PATH for this session if not already there
if [[ ":$PATH:" != *":${LLAMA_DIR}:"* ]]; then
    export PATH="${LLAMA_DIR}:$PATH"
    echo "Added ${LLAMA_DIR} to PATH for this session."
    # Persist in .bashrc for future sessions
    if ! grep -q "llama" ~/.bashrc 2>/dev/null; then
        echo "export PATH=\"${LLAMA_DIR}:\$PATH\"" >> ~/.bashrc
        echo "Added to ~/.bashrc for future sessions."
    fi
fi

# ── STEP 2/4: Check/download GGUF model ────────────────────────────────────
echo ""
echo -e "${BOLD}[STEP 2/4] Checking GGUF model...${NC}"
echo ""

mkdir -p "${MODELS_DIR}"

# Check for the configured GGUF specifically, but still report existing models.
GGUF_FOUND=$(first_match "${MODELS_DIR}" -name "*.gguf" ! -iname "*mmproj*" || true)

if [ -f "${GGUF_FILE}" ]; then
    echo -e "${GREEN}✅ Configured GGUF already present in ${MODELS_DIR} — skipping download.${NC}"
    echo "   Found: $(basename "${GGUF_FILE}")"
else
    if [ -n "${GGUF_FOUND}" ]; then
        echo -e "${YELLOW}⚠  A different GGUF already exists in ${MODELS_DIR}, but the configured default is missing.${NC}"
        echo "   Existing: $(basename "${GGUF_FOUND}")"
    else
        echo -e "${YELLOW}⚠  No GGUF found in ${MODELS_DIR}.${NC}"
    fi
    echo ""
    echo "Configured model: ${HF_REPO} -> ${GGUF_FILE_NAME}"
    echo ""
    if [ -z "${AUTO_DOWNLOAD}" ] && [ ! -t 0 ]; then
        AUTO_DOWNLOAD="y"
    fi

    if [ -n "${AUTO_DOWNLOAD}" ]; then
        DOWNLOAD_GGUF="${AUTO_DOWNLOAD}"
        echo "Auto download choice: ${DOWNLOAD_GGUF}"
    else
        read -r -p "Download both now? (y/n): " DOWNLOAD_GGUF
    fi

    if [[ "${DOWNLOAD_GGUF,,}" == "y" ]]; then
        echo ""
        
        # ── Setup HF CLI & Transfer ──
        export HF_HUB_ENABLE_HF_TRANSFER=1
        if ! python3 -c "import huggingface_hub, hf_transfer" 2>/dev/null; then
            echo "Installing hf_transfer for max download speed (Rust)..."
            install_python_packages -q hf_transfer huggingface_hub
        fi
        
        echo ""
        echo "Downloading model GGUF — ~15.7GB utilizing max workers (Rust)..."
        
        # Build token argument if token exists as an env secret
        HF_TOKEN_ARG=""
        if [ -n "${HF_TOKEN:-}" ]; then
            HF_TOKEN_ARG="--token ${HF_TOKEN}"
        fi

        if ! download_hf_asset "${HF_REPO}" "${GGUF_FILE_NAME}" "${MODELS_DIR}" "${GGUF_FILE}" "${GGUF_URL}" "Model GGUF" "${HF_TOKEN_ARG}"; then
            echo -e "${RED}❌ GGUF download failed. Download manually and place in ${MODELS_DIR}${NC}"
            exit 1
        fi

        echo ""
        echo "Downloading mmproj (enables image input)..."
        if ! download_hf_asset "${HF_REPO}" "${MMPROJ_FILE_NAME}" "${MODELS_DIR}" "${MMPROJ_FILE}" "${MMPROJ_URL}" "mmproj" "${HF_TOKEN_ARG}"; then
            echo -e "${YELLOW}❌ mmproj download failed. You can grab it manually from HuggingFace.${NC}"
        fi
    else
        echo ""
        echo "Skipped. Place your GGUF in: ${MODELS_DIR}"
        echo "Then restart ComfyUI — it will appear in the node dropdown."
    fi
fi

if [ -f "${GGUF_FILE}" ] && [ ! -f "${MMPROJ_FILE}" ] && [ -n "${MMPROJ_FILE_NAME}" ]; then
    echo ""
    echo -e "${YELLOW}⚠ Configured vision adapter is missing: ${MMPROJ_FILE_NAME}${NC}"
    echo "Downloading mmproj for the configured default model..."
    HF_TOKEN_ARG=""
    if [ -n "${HF_TOKEN:-}" ]; then
        HF_TOKEN_ARG="--token ${HF_TOKEN}"
    fi
    if ! download_hf_asset "${HF_REPO}" "${MMPROJ_FILE_NAME}" "${MODELS_DIR}" "${MMPROJ_FILE}" "${MMPROJ_URL}" "mmproj" "${HF_TOKEN_ARG}"; then
        echo -e "${YELLOW}❌ mmproj download failed. You can grab it manually from HuggingFace.${NC}"
    fi
fi

# ── STEP 3/4: Install Python dependencies ──────────────────────────────────
echo ""
echo -e "${BOLD}[STEP 3/4] Checking Python dependencies...${NC}"
echo ""

# Install requests if not present
if python3 -c "import requests" 2>/dev/null; then
    echo -e "${GREEN}✅ requests already installed.${NC}"
else
    echo "Installing requests..."
    install_python_packages requests
fi

# Ensure Pillow is available (needed for image encoding)
if python3 -c "from PIL import Image" 2>/dev/null; then
    echo -e "${GREEN}✅ Pillow already installed.${NC}"
else
    echo "Installing Pillow..."
    install_python_packages Pillow
fi

# ── STEP 4/4: Install CUDA dependencies for llama.cpp ──────────────────────
echo ""
echo -e "${BOLD}[STEP 4/4] Checking CUDA libraries...${NC}"
echo ""

# Lightning.ai Studios come with CUDA pre-installed, but verify
if command -v nvidia-smi &>/dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
else
    echo -e "${YELLOW}⚠  nvidia-smi not found. GPU may not be available.${NC}"
    echo "   llama-server will run on CPU (much slower)."
fi

# Verify CUDA libs exist
if ldconfig -p 2>/dev/null | grep -q libcudart; then
    echo -e "${GREEN}✅ CUDA runtime libraries found.${NC}"
else
    echo -e "${YELLOW}⚠  CUDA runtime not found in ldconfig. If llama-server fails,${NC}"
    echo "   you may need to: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
fi

# ── DONE ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║   ✅ Setup Complete!                                 ║${NC}"
echo -e "${CYAN}${BOLD}╠══════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}${BOLD}║                                                      ║${NC}"
echo -e "${CYAN}║   llama-server : ${LLAMA_EXE}${NC}"
echo -e "${CYAN}║   Models folder: ${MODELS_DIR}${NC}"
echo -e "${CYAN}║   Model        : ${GGUF_FILE_NAME}  ${NC}"
echo -e "${CYAN}║   Vision       : ${MMPROJ_FILE_NAME}  ${NC}"
echo -e "${CYAN}${BOLD}║                                                      ║${NC}"
echo -e "${CYAN}${BOLD}║   Next steps:                                        ║${NC}"
echo -e "${CYAN}${BOLD}║   1. Restart ComfyUI                                 ║${NC}"
echo -e "${CYAN}${BOLD}║   2. Add the Gemma4 Prompt Engineer node             ║${NC}"
echo -e "${CYAN}${BOLD}║   3. Hit PREVIEW — node handles the rest             ║${NC}"
echo -e "${CYAN}${BOLD}║                                                      ║${NC}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
