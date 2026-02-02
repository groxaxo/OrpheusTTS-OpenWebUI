# OrpheusTTS-OpenWebUI

This is a fork of the [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) project, adding:
- **Gradio WebUI** for easy text-to-speech generation
- **FastAPI Server** with OpenAI-compatible API endpoints
- **LMStudio Integration** for using any Orpheus GGUF model
- **Text Sanitization** for better pronunciation of URLs, numbers, etc.

![image](https://github.com/user-attachments/assets/4b738f1d-23ed-477b-ac84-db0d5b04c76c)

https://github.com/user-attachments/assets/5e441285-b10f-4149-b691-df061c5ddcbb

## ✅ Latest Updates

### 🆕 FastAPI Server with LMStudio Integration
- **OpenAI-Compatible API**: Use the `/v1/audio/speech` endpoint for easy integration
- **Any Orpheus GGUF Model**: Works with any Orpheus model loaded in LMStudio
- **Text Sanitization**: Automatic normalization of URLs, emails, numbers, money, time, and more
- **Web UI**: Simple browser-based interface for testing

### Long-Form Text Processing
- **Tabbed Interface**: The UI now features a dedicated "Long Form Content" tab for processing larger text inputs
- **Smart Text Chunking**: Automatically splits long text into smaller chunks at sentence boundaries
- **Parallel Processing**: Processes multiple chunks simultaneously for faster generation
- **Seamless Audio Stitching**: Combines multiple audio segments into one cohesive output file with crossfading
- **Progress Tracking**: Real-time progress indicators during the generation process

### Technical Improvements
- **Enhanced Logging**: Better error handling and diagnostic information
- **Memory Optimization**: Improved cleanup of temporary files
- **Expanded Parameter Ranges**: Maximum tokens extended to 16384 for longer audio generation
- **Batch Size Control**: Adjust the number of chunks processed in parallel to balance speed and resource usage

## Features

- **Easy-to-use Web Interface**: Simple Gradio UI for text-to-speech generation
- **FastAPI Server**: OpenAI-compatible API for programmatic access
- **LMStudio Support**: Use any Orpheus GGUF model pre-loaded in LMStudio
- **Text Sanitization**: Automatic handling of URLs, numbers, currencies, etc.
- **WSL & CUDA Compatible**: Optimized for Windows Subsystem for Linux with CUDA support
- **Memory Optimized**: Addresses common memory issues on consumer GPUs
- **Voice Selection**: Access to all voices from the original model (8+ languages)
- **Emotive Tags Support**: Full support for all emotion tags

## Quick Start - FastAPI Server (LMStudio)

### 1. Start LMStudio with Orpheus Model

1. Download any Orpheus GGUF model (e.g., `Orpheus-3b-FT-Q8_0.gguf`, `Orpheus-3b-FT-Q4_K_M.gguf`)
2. Load the model in LMStudio
3. Start the local server (default: `http://127.0.0.1:1234`)

### 2. Configure and Start the FastAPI Server

```bash
# Clone the repository
git clone https://github.com/groxaxo/OrpheusTTS-OpenWebUI.git
cd OrpheusTTS-OpenWebUI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env to configure your LMStudio URL if needed

# Start the FastAPI server
python app.py
```

### 3. Use the API

**Web UI**: Open http://localhost:5005 in your browser

**cURL Example**:
```bash
curl -X POST http://localhost:5005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "voice": "tara"}' \
  --output speech.wav
```

**Python Example**:
```python
import requests

response = requests.post(
    "http://localhost:5005/v1/audio/speech",
    json={"input": "Hello, world!", "voice": "tara"}
)
with open("speech.wav", "wb") as f:
    f.write(response.content)
```

## Quick Start - Gradio WebUI (WSL/Linux)

```bash
# Clone the repository
git clone https://github.com/groxaxo/OrpheusTTS-OpenWebUI.git
cd OrpheusTTS-OpenWebUI

# Run the setup script
chmod +x setup_orpheus.sh
./setup_orpheus.sh

# Launch the app
./launch_orpheus.sh
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on RTX 3090 / 4090)
- WSL2 or Linux
- PyTorch 2.6.0 with CUDA
- Hugging Face account with access to the Orpheus TTS models

## Available Voices

The WebUI and FastAPI server provide access to voices in multiple languages:

**English** (in order of conversational realism):
- tara, leah, jess, leo, dan, mia, zac, zoe

**Other Languages**:
- French: pierre, amelie, marie
- German: jana, thomas, max
- Korean: 유나, 준서
- Hindi: ऋतिका
- Mandarin: 长乐, 白芷
- Spanish: javi, sergio, maria
- Italian: pietro, giulia, carlo

## Emotive Tags

Add emotion to your speech with tags:
- `<laugh>`
- `<chuckle>`
- `<sigh>`
- `<cough>`
- `<sniffle>`
- `<groan>`
- `<yawn>`
- `<gasp>`

## Long Form Text Processing

The new Long Form feature lets you generate speech for larger text inputs:

1. **Text Chunking**: Text is automatically split into manageable chunks at sentence boundaries
2. **Parallel Processing**: Process multiple chunks simultaneously based on the batch size setting
3. **Parameter Optimization**: The Long Form tab offers optimized default settings for extended content
4. **Simple Assembly**: All audio chunks are automatically combined into a single cohesive output file with crossfade stitching

This is ideal for:
- Articles and blog posts
- Scripts and dialogues
- Books and stories
- Any text content that exceeds a few paragraphs

## Text Sanitization

The FastAPI server includes automatic text sanitization for better pronunciation:

| Input Type | Example Input | Spoken Output |
|------------|--------------|---------------|
| URLs | `https://example.com` | "https example dot com" |
| Emails | `user@test.com` | "user at test dot com" |
| Money | `$50.30` | "fifty dollars and thirty cents" |
| Numbers | `1035` | "one thousand and thirty-five" |
| Time | `10:35 pm` | "ten thirty-five pm" |
| Years | `1998` | "nineteen ninety-eight" |
| Abbreviations | `Dr. Smith` | "Doctor Smith" |
| Symbols | `@`, `&`, `%` | "at", "and", "percent" |

Sanitization is enabled by default and can be controlled via the API.

## API Endpoints

The FastAPI server provides OpenAI-compatible endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech` | POST | Generate speech (OpenAI-compatible) |
| `/v1/audio/voices` | GET | List available voices |
| `/speak` | POST | Legacy TTS endpoint |
| `/sanitize` | POST | Preview text sanitization |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI documentation |

## Configuration

Configure the server via `.env` file:

```env
# LMStudio API URL
ORPHEUS_API_URL=http://127.0.0.1:1234/v1/completions

# Model name (empty = use loaded model)
ORPHEUS_MODEL_NAME=

# Generation parameters
ORPHEUS_MAX_TOKENS=8192
ORPHEUS_TEMPERATURE=0.6
ORPHEUS_TOP_P=0.9

# Server settings
ORPHEUS_PORT=5005
ORPHEUS_HOST=0.0.0.0
```

## Troubleshooting

If you encounter "KV cache" errors, the setup script should address these automatically. If problems persist, try:
- Reducing `max_model_len` in the `orpheus_wrapper.py` file
- Ensuring your GPU has enough VRAM (recommended 12GB+)
- Setting `gpu_memory_utilization` to a lower value (0.7-0.8)
- For Long Form processing, try reducing the batch size to limit memory usage

---

# Official Orpheus TTS Documentation

## ✨ Upstream Updates

- **[5/2025]** Partnered with [Baseten](https://www.baseten.co/blog/canopy-labs-selects-baseten-as-preferred-inference-provider-for-orpheus-tts-model) for optimized inference at fp8 and fp16. See [deployment guide](/additional_inference_options/baseten_inference_example/README.md).
- **[4/2025]** Released [multilingual models](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba) with [training guide](https://canopylabs.ai/releases/orpheus_can_speak_any_language#training).

## Overview
Orpheus TTS is an open-source text-to-speech system built on the Llama-3b backbone. Orpheus demonstrates the emergent capabilities of using LLMs for speech synthesis. We offer comparisons of the models below to leading closed models like Eleven Labs and PlayHT in our blog post.

[Check out our blog post](https://canopylabs.ai/model-releases)


https://github.com/user-attachments/assets/ce17dd3a-f866-4e67-86e4-0025e6e87b8a


## Abilities

- **Human-Like Speech**: Natural intonation, emotion, and rhythm that is superior to SOTA closed source models
- **Zero-Shot Voice Cloning**: Clone voices without prior fine-tuning
- **Guided Emotion and Intonation**: Control speech and emotion characteristics with simple tags
- **Low Latency**: ~200ms streaming latency for realtime applications, reducible to ~100ms with input streaming

## Models

We provide three models in this release, and additionally we offer the data processing scripts and sample datasets to make it very straightforward to create your own finetune.

1. [**Finetuned Prod**](https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod) – A finetuned model for everyday TTS applications

2. [**Pretrained**](https://huggingface.co/canopylabs/orpheus-tts-0.1-pretrained) – Our base model trained on 100k+ hours of English speech data


### Inference
#### Simple setup on colab
1. [Colab For Tuned Model](https://colab.research.google.com/drive/1KhXT56UePPUHhqitJNUxq63k-pQomz3N?usp=sharing) (not streaming, see below for realtime streaming) – A finetuned model for everyday TTS applications.
2. [Colab For Pretrained Model](https://colab.research.google.com/drive/10v9MIEbZOr_3V8ZcPAIh8MN7q2LjcstS?usp=sharing) – This notebook is set up for conditioned generation but can be extended to a range of tasks.

#### Prompting

1. The `finetune-prod` models: for the primary model, your text prompt is formatted as `{name}: I went to the ...`. The options for name in order of conversational realism (subjective benchmarks) are "tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe". Our python package does this formatting for you, and the notebook also prepends the appropriate string. You can additionally add the following emotive tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`.

2. The pretrained model: you can either generate speech just conditioned on text, or generate speech conditioned on one or more existing text-speech pairs in the prompt. Since this model hasn't been explicitly trained on the zero-shot voice cloning objective, the more text-speech pairs you pass in the prompt, the more reliably it will generate in the correct voice.

Additionally, use regular LLM generation args like `temperature`, `top_p`, etc. as you expect for a regular LLM. `repetition_penalty>=1.1`is required for stable generations. Increasing `repetition_penalty` and `temperature` makes the model speak faster.


## Finetune Model

Here is an overview of how to finetune your model on any text and speech.
This is a very simple process analogous to tuning an LLM using Trainer and Transformers.

You should start to see high quality results after ~50 examples but for best results, aim for 300 examples/speaker.

1. Your dataset should be a huggingface dataset in [this format](https://huggingface.co/datasets/canopylabs/zac-sample-dataset)
2. We prepare the data using [this notebook](https://colab.research.google.com/drive/1wg_CPCA-MzsWtsujwy-1Ovhv-tn8Q1nD?usp=sharing). This pushes an intermediate dataset to your Hugging Face account which you can can feed to the training script in finetune/train.py. Preprocessing should take less than 1 minute/thousand rows.
3. Modify the `finetune/config.yaml` file to include your dataset and training properties, and run the training script. You can additionally run any kind of huggingface compatible process like Lora to tune the model.
   ```bash
    pip install transformers datasets wandb trl flash_attn torch
    huggingface-cli login <enter your HF token>
    wandb login <wandb token>
    accelerate launch train.py
   ```

### Additional Finetuning Resources
- [LoRA Finetuning](finetune/lora.py) - Parameter-efficient finetuning with LoRA
- [PEFT finetuning with Unsloth](https://github.com/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb)

## Pretrain Model

This is a very simple process analogous to training an LLM using Trainer and Transformers.

The base model provided is trained over 100k hours. We recommend not using synthetic data for training as it produces worse results when you try to finetune specific voices, probably because synthetic voices lack diversity and map to the same set of tokens when tokenised (i.e. lead to poor codebook utilisation).

We train the 3b model on sequences of length 8192 - we use the same dataset format for TTS finetuning for the <TTS-dataset> pretraining. We chain input_ids sequences together for more efficient training. The text dataset required is in the form described in this issue [#37](https://github.com/canopyai/Orpheus-TTS/issues/37). 

If you are doing extended training this model, i.e. for another language or style we recommend starting with finetuning only (no text dataset). The main idea behind the text dataset is discussed in the blog post. (tldr; doesn't forget too much semantic/reasoning ability so its able to better understand how to intone/express phrases when spoken, however most of the forgetting would happen very early on in the training i.e. <100000 rows), so unless you are doing very extended finetuning it may not make too much of a difference.

## Additional Inference Options

1. **Watermark your audio**: Use Silent Cipher to watermark your audio generation; see [Watermark Audio Implementation](additional_inference_options/watermark_audio) for details.
2. **No-GPU Inference**: Run Orpheus on CPU using orpheus-cpp; see [No-GPU Guide](additional_inference_options/no_gpu/README.md).
3. **Baseten Deployment**: Production-ready deployment with fp8/fp16 support; see [Baseten Guide](additional_inference_options/baseten_inference_example/README.md).

## Community Implementations

While we can't verify these implementations are completely accurate/bug free, they have been recommended on forums:

1. [A lightweight client for running Orpheus TTS locally using LM Studio API](https://github.com/isaiahbjork/orpheus-tts-local)
2. [Open AI compatible Fast-API implementation](https://github.com/Lex-au/Orpheus-FastAPI)
3. [Gradio WebUI that runs smoothly on WSL and CUDA](https://github.com/Saganaki22/OrpheusTTS-WebUI) (this repository)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

<br>
<br>

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="
      https://api.star-history.com/svg?repos=Saganaki22/OrpheusTTS-WebUI&type=Date&theme=dark
    "
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="
      https://api.star-history.com/svg?repos=Saganaki22/OrpheusTTS-WebUI&type=Date
    "
  />
  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=Saganaki22/OrpheusTTS-WebUI&type=Date"
  />
</picture>
