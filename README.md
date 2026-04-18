# G4promptool

## How to run this in Lightning.ai Studio

1. Open your Lightning.ai Studio terminal.
2. Navigate to this extension's directory:
   ```bash
   cd /teamspace/studios/this_studio/ComfyUI/custom_nodes/G4promptool
   ```
   *(Note: Adjust the path if you cloned it to a different custom_nodes folder)*
3. Make the setup script executable:
   ```bash
   chmod +x setup_gemma4_promptld.sh
   ```
4. Run the setup script to install `llama-server` and download the model GGUFs:
   ```bash
   ./setup_gemma4_promptld.sh
   ```
   For automation or non-interactive setup, use:
   ```bash
   ./setup_gemma4_promptld.sh --yes
   ```
5. Wait for the setup to finish downloading (auto-detects and uses high-speed `hf transfer` if supported).
6. Restart ComfyUI.
7. Add the **llama.cpp Prompt Engineer** node in ComfyUI, configure your inputs, and run your workflow!

### Configuration
The defaults target [GitMylo/nsfwvision-v5_qwen3.5-9b-gguf](https://huggingface.co/GitMylo/nsfwvision-v5_qwen3.5-9b-gguf), but you can edit `model_config.json` to point the setup script at any llama.cpp-compatible GGUF and matching `mmproj` file.
