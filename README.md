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
5. Wait for the setup to finish downloading (auto-detects and uses high-speed `hf transfer` if supported).
6. Restart ComfyUI.
7. Add the **Gemma4 Prompt Engineer** node in ComfyUI, configure your inputs, and run your workflow!

### Configuration
You can edit the `model_config.json` file to change the model repository or filenames that will be downloaded via the setup script and run by the node.
