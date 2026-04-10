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

# ── ANSI COLOURS ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

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

# llama.cpp Linux CUDA release — update version as needed
LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/b8664/llama-b8664-bin-ubuntu-x64-cuda-cu12.4.tar.gz"
LLAMA_ARCHIVE="${LLAMA_DIR}/llama_install.tar.gz"

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

# ── STEP 1/4: Install llama-server ──────────────────────────────────────────
echo -e "${BOLD}[STEP 1/4] Checking llama-server...${NC}"
echo ""

# Check if already in PATH
if command -v llama-server &>/dev/null; then
    echo -e "${GREEN}✅ llama-server found in PATH — skipping install.${NC}"
elif [ -f "${LLAMA_EXE}" ]; then
    echo -e "${GREEN}✅ llama-server found at ${LLAMA_EXE} — skipping install.${NC}"
else
    echo -e "${YELLOW}⚠  llama-server not found. Downloading to ${LLAMA_DIR}...${NC}"
    echo ""
    echo "URL: ${LLAMA_URL}"
    echo ""

    mkdir -p "${LLAMA_DIR}"

    # Download
    if ! curl -L --progress-bar -o "${LLAMA_ARCHIVE}" "${LLAMA_URL}"; then
        echo ""
        echo -e "${RED}❌ Download failed. Check your internet connection.${NC}"
        echo "   You can manually download from:"
        echo "   ${LLAMA_URL}"
        echo "   and extract to ${LLAMA_DIR}"
        exit 1
    fi

    echo ""
    echo "Extracting..."

    # Extract tar.gz (Linux release is a tarball, not zip)
    if [[ "${LLAMA_ARCHIVE}" == *.tar.gz ]] || [[ "${LLAMA_ARCHIVE}" == *.tgz ]]; then
        tar -xzf "${LLAMA_ARCHIVE}" -C "${LLAMA_DIR}" --strip-components=1 2>/dev/null || \
        tar -xzf "${LLAMA_ARCHIVE}" -C "${LLAMA_DIR}" 2>/dev/null
    elif [[ "${LLAMA_ARCHIVE}" == *.zip ]]; then
        unzip -o "${LLAMA_ARCHIVE}" -d "${LLAMA_DIR}"
        # Flatten any subfolder
        if [ "$(ls -d "${LLAMA_DIR}"/*/  2>/dev/null | wc -l)" -eq 1 ]; then
            SUBDIR=$(ls -d "${LLAMA_DIR}"/*/ | head -1)
            mv "${SUBDIR}"* "${LLAMA_DIR}/" 2>/dev/null || true
            rmdir "${SUBDIR}" 2>/dev/null || true
        fi
    fi

    rm -f "${LLAMA_ARCHIVE}"

    # Find llama-server in extracted files (may be nested)
    if [ ! -f "${LLAMA_EXE}" ]; then
        FOUND_EXE=$(find "${LLAMA_DIR}" -name "llama-server" -type f 2>/dev/null | head -1)
        if [ -n "${FOUND_EXE}" ]; then
            # Move it to the expected location if it's nested
            if [ "${FOUND_EXE}" != "${LLAMA_EXE}" ]; then
                mv "${FOUND_EXE}" "${LLAMA_EXE}"
            fi
        fi
    fi

    # Make executable
    if [ -f "${LLAMA_EXE}" ]; then
        chmod +x "${LLAMA_EXE}"
        echo -e "${GREEN}✅ llama-server installed at ${LLAMA_EXE}${NC}"
    else
        echo -e "${RED}❌ llama-server not found after extraction.${NC}"
        echo "   Check ${LLAMA_DIR} manually."
        echo "   Available files:"
        ls -la "${LLAMA_DIR}/"
        exit 1
    fi
fi

# Also make all binaries in the llama dir executable
find "${LLAMA_DIR}" -type f -executable -o -name "llama-*" 2>/dev/null | while read f; do
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

# Check if any GGUF already exists (excluding mmproj)
GGUF_FOUND=$(find "${MODELS_DIR}" -name "*.gguf" ! -iname "*mmproj*" 2>/dev/null | head -1)

if [ -n "${GGUF_FOUND}" ]; then
    echo -e "${GREEN}✅ GGUF model already present in ${MODELS_DIR} — skipping download.${NC}"
    echo "   Found: $(basename "${GGUF_FOUND}")"
else
    echo -e "${YELLOW}⚠  No GGUF found in ${MODELS_DIR}.${NC}"
    echo ""
    echo "Configured model: ${HF_REPO} -> ${GGUF_FILE_NAME}"
    echo ""
    read -p "Download both now? (y/n): " DOWNLOAD_GGUF

    if [[ "${DOWNLOAD_GGUF,,}" == "y" ]]; then
        echo ""
        
        # ── Setup HF CLI & Transfer ──
        export HF_HUB_ENABLE_HF_TRANSFER=1
        if ! python3 -c "import huggingface_hub, hf_transfer" 2>/dev/null; then
            echo "Installing hf_transfer for max download speed (Rust)..."
            pip install -q hf_transfer huggingface_hub
        fi
        
        echo ""
        echo "Downloading model GGUF — ~15.7GB utilizing max workers (Rust)..."
        
        # Build token argument if token exists as an env secret
        HF_TOKEN_ARG=""
        if [ -n "${HF_TOKEN:-}" ]; then
            HF_TOKEN_ARG="--token ${HF_TOKEN}"
        fi

        if hf download "${HF_REPO}" "${GGUF_FILE_NAME}" --local-dir "${MODELS_DIR}" $HF_TOKEN_ARG; then
            echo -e "${GREEN}✅ Model downloaded via High-Speed HF CLI.${NC}"
        else
            echo -e "${YELLOW}⚠ HF CLI failed. Falling back to curl...${NC}"
            if [ -n "${HF_TOKEN:-}" ]; then
                if ! curl -H "Authorization: Bearer ${HF_TOKEN}" -L --progress-bar -o "${GGUF_FILE}" "${GGUF_URL}"; then
                    echo -e "${RED}❌ GGUF download failed. Download manually and place in ${MODELS_DIR}${NC}"
                    exit 1
                fi
            else
                if ! curl -L --progress-bar -o "${GGUF_FILE}" "${GGUF_URL}"; then
                    echo -e "${RED}❌ GGUF download failed. Download manually and place in ${MODELS_DIR}${NC}"
                    exit 1
                fi
            fi
            echo -e "${GREEN}✅ Model downloaded via fallback.${NC}"
        fi

        echo ""
        echo "Downloading mmproj (enables image input)..."
        if hf download "${HF_REPO}" "${MMPROJ_FILE_NAME}" --local-dir "${MODELS_DIR}" $HF_TOKEN_ARG; then
            echo -e "${GREEN}✅ mmproj downloaded — vision enabled.${NC}"
        else
            echo -e "${YELLOW}⚠ HF CLI failed. Falling back to curl...${NC}"
            if [ -n "${HF_TOKEN:-}" ]; then
                if ! curl -H "Authorization: Bearer ${HF_TOKEN}" -L --progress-bar -o "${MMPROJ_FILE}" "${MMPROJ_URL}"; then
                    echo -e "${YELLOW}❌ mmproj download failed. You can grab it manually from HuggingFace.${NC}"
                else
                    echo -e "${GREEN}✅ mmproj downloaded — vision enabled.${NC}"
                fi
            else
                if ! curl -L --progress-bar -o "${MMPROJ_FILE}" "${MMPROJ_URL}"; then
                    echo -e "${YELLOW}❌ mmproj download failed. You can grab it manually from HuggingFace.${NC}"
                else
                    echo -e "${GREEN}✅ mmproj downloaded — vision enabled.${NC}"
                fi
            fi
        fi
    else
        echo ""
        echo "Skipped. Place your GGUF in: ${MODELS_DIR}"
        echo "Then restart ComfyUI — it will appear in the node dropdown."
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
    pip install requests
fi

# Ensure Pillow is available (needed for image encoding)
if python3 -c "from PIL import Image" 2>/dev/null; then
    echo -e "${GREEN}✅ Pillow already installed.${NC}"
else
    echo "Installing Pillow..."
    pip install Pillow
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
