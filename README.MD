# Mental Health Fine-Tuned LLaMA 2

> A QLoRA fine-tuned LLaMA 2 7B Chat model trained on mental health counseling conversations, served through a FastAPI backend with a clean chat UI — containerized and ready to deploy.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/🤗-ArjunRavi%2Fmental--health--finetune-yellow)](https://huggingface.co/ArjunRavi/mental-health-finetune)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 Overview

This project fine-tunes **LLaMA 2 7B Chat** (`NousResearch/Llama-2-7b-chat-hf`) on the [`Amod/mental_health_counseling_conversations`](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) dataset using **QLoRA** — 4-bit quantization combined with Low-Rank Adaptation — making the full fine-tuning process feasible on a single consumer GPU (or a free Colab T4).

The resulting LoRA adapter weights are published to HuggingFace Hub as [`ArjunRavi/mental-health-finetune`](https://huggingface.co/ArjunRavi/mental-health-finetune) and loaded at inference time alongside the frozen base model. The entire stack is served through a FastAPI application with a minimal chat UI, wrapped in a Docker container.

---

## ✨ Key Capabilities

- **Empathetic response generation** — counseling-style answers to mental health prompts such as anxiety, lack of motivation, emotional distress, and focus issues
- **Efficient deployment** — LoRA adapter is ~16 MB; the base model is loaded once in `bfloat16` and shared
- **Chat UI** — single-page HTML/JS interface that submits prompts and streams responses without page reload
- **Docker-ready** — production deployment with a single `docker build` + `docker run`

---

## 🗂️ Repository Structure

```
mental-health-finetuned-llama/
│
├── research/
│   ├── llama_finetune.ipynb              # QLoRA fine-tuning pipeline (main training notebook)
│   └── llama_mental_health_usecase.ipynb # Inference walkthrough + manual autoregressive decoding
│
├── backend/
│   ├── apps/
│   │   ├── __init__.py
│   │   ├── main.py                       # FastAPI app entry point
│   │   ├── api/
│   │   │   └── v1/
│   │   │       └── routes.py             # GET / and POST /generate endpoints
│   │   ├── core/
│   │   │   ├── config.py                 # Settings via pydantic-settings + YAML
│   │   │   ├── config.yaml               # base_repo + lora_repo pointers
│   │   │   └── logger.py                 # Logging configuration
│   │   └── models/
│   │       └── inference.py              # Model loading + generate_text()
│   ├── static/
│   │   ├── index.html                    # Jinja2 chat template
│   │   ├── script.js                     # Async fetch + DOM injection
│   │   └── style.css                     # Chat UI styles
│   └── requirements.txt
│
├── docker/
│   └── Dockerfile.backend                # Python 3.11-slim, uvicorn entrypoint
│
└── requirements.txt
```

---

## 🔬 Fine-Tuning Pipeline (`research/llama_finetune.ipynb`)

### 1. Dataset

| Property | Value |
|---|---|
| Dataset | [`Amod/mental_health_counseling_conversations`](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) |
| Split | `train` |
| Columns used | `Context` (user input), `Response` (counselor response) |
| Format after mapping | LLaMA 2 Instruct chat template |

Each example is formatted into the LLaMA 2 chat template before training:

```python
def format_dataset(data):
    return {
        'text': f"<s>[INST] {data['Context']} [/INST] {data['Response']} </s>"
    }
```

### 2. Data Cleaning

The dataset contains residual HTML artifacts in some responses. A regex pass strips them out after formatting:

```python
def clean_html(example):
    pattern = r"<.*?\/>|<.*"
    return {'cleaned_text': re.sub(pattern, "", example['text'], flags=re.DOTALL)}
```

Token length distribution is analyzed using IQR to understand outliers before setting the truncation ceiling:

```python
q1 = np.percentile(lengths, 25)
q3 = np.percentile(lengths, 75)
upper_bound = (1.5 * (q3 - q1)) + q3   # IQR-based outlier ceiling
```

All sequences are finally tokenized with `truncation=True, max_length=512`.

### 3. Quantization — BitsAndBytes 4-bit NF4

Full fine-tuning of a 7B parameter model requires ~112 GB VRAM in fp32. 4-bit quantization with NF4 (NormalFloat4) reduces this to ~5–6 GB, fitting comfortably on a Colab T4 or RTX 3090.

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NormalFloat4: optimal for normally distributed weights
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
```

### 4. LoRA Configuration

Only the attention and FFN projection matrices are updated — the rest of the model stays frozen. This is QLoRA: a frozen 4-bit base plus trainable low-rank adapters in fp16.

```python
peft_config = LoraConfig(
    r=8,                    # Rank: adapters are B∈R^{d×r} and A∈R^{r×k}
    lora_alpha=16,          # Effective scaling: alpha/r = 2.0
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # Attention projections
        "gate_proj", "up_proj", "down_proj"         # SwiGLU FFN gates
    ],
    bias="none",
    task_type="CAUSAL_LM"
)
```

All 7 linear projection layers are targeted (attention + FFN), giving the adapter maximum expressiveness while still training far fewer parameters than full fine-tuning.

### 5. Training Arguments

| Hyperparameter | Value | Notes |
|---|---|---|
| Epochs | 1 | Single pass over the dataset |
| Per-device batch size | 1 | VRAM constrained |
| Gradient accumulation | 4 | Effective batch size = 4 |
| Learning rate | 2e-4 | Standard for LoRA |
| LR schedule | cosine | Smooth decay to 0 |
| Optimizer | `paged_adamw_32bit` | Offloads optimizer state pages to CPU RAM |
| Warmup ratio | 0.03 | Linear warmup over 3% of total steps |
| Gradient clipping | 0.3 | Prevents explosive updates |
| Weight decay | 0.001 | Light regularization |
| Logging | TensorBoard | `results/runs/` |

Training is managed by `SFTTrainer` from the `trl` library, which handles LoRA injection, dataset batching, and checkpointing.

### 6. Saving & Publishing

After training, only the LoRA adapter delta weights (~16 MB) are saved and pushed to HuggingFace Hub — the 13 GB base model is loaded separately at runtime:

```python
trainer.model.save_pretrained("Llama-2-7b-chat-hf-finetune")
model.push_to_hub("ArjunRavi/mental-health-finetune")
```

---

## 🔍 Inference Notebook (`research/llama_mental_health_usecase.ipynb`)

This notebook demonstrates two inference modes.

### High-level inference via `model.generate()`

Uses the full LLaMA 2 system prompt format:

```python
prompt = """<s>[INST] <<SYS>>
You are a helpful and empathetic assistant.
<</SYS>>

I am feeling very down, without any reason. Also I am very distracted and cannot find interest in anything.
[/INST]"""

response = lora_model.generate(
    **inputs,
    max_new_tokens=250,
    temperature=0.3,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
```

### Manual autoregressive decoding loop

A pedagogical implementation that bypasses `model.generate()` entirely, doing one forward pass per token to expose exactly how autoregressive decoding works under the hood:

```python
input_ids = input['input_ids']
for i in range(10):
    with torch.no_grad():
        outputs = lora_model(input_ids)
        logits = outputs.logits                                    # [B, S, vocab_size]
    predicted_token = torch.argmax(logits[:, -1, :], dim=-1)      # greedy: pick argmax at last position
    input_ids = torch.cat([input_ids, predicted_token.unsqueeze(1)], dim=-1)  # append and repeat
```

This is exactly what `model.generate()` does internally with `do_sample=False` — seeing it written out makes the KV cache optimization and why it matters immediately intuitive.

---

## 🌐 Serving Stack (`backend/`)

### Architecture

```
Browser
  │
  ├─ GET  /          → renders index.html via Jinja2
  └─ POST /generate  → returns index.html with {prompt, response} injected
           │
           ▼
    FastAPI (Uvicorn, port 8000)
      ├── /static → style.css, script.js
      └── routes.py
            └── generate_text(prompt)
                  ├── tokenizer(prompt)
                  ├── model.generate(...)   ← bfloat16 base + LoRA adapter
                  └── tokenizer.decode(outputs)
```

### Configuration (`core/config.yaml`)

```yaml
base_repo: "NousResearch/Llama-2-7b-chat-hf"
lora_repo: "ArjunRavi/mental-health-finetune"
```

Loaded at startup via `pydantic-settings` with automatic CUDA detection:

```python
class Settings(BaseSettings):
    base_repo: str = CONFIG["base_repo"]
    lora_repo: str = CONFIG["lora_repo"]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

Swapping to a different base model or adapter requires changing only `config.yaml` — no code changes needed.

### Inference (`models/inference.py`)

The base model and LoRA adapter are loaded once at module import time, not per-request. This is the correct production pattern — it avoids the 10–30s cold-start on every call.

```python
# Loaded once at server startup
tokenizer = AutoTokenizer.from_pretrained(settings.base_repo)
base_model = AutoModelForCausalLM.from_pretrained(
    settings.base_repo, device_map="auto", dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base_model, settings.lora_repo)

def generate_text(prompt: str, max_new_tokens: int = 200, temperature: float = 0.5) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(settings.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Renders `index.html` chat interface |
| `POST` | `/generate` | Accepts `prompt` form field, calls `generate_text()`, re-renders template with response |

### Frontend

The UI is a single Jinja2-rendered HTML page. JavaScript intercepts the form submit, POSTs via `fetch()`, parses the returned HTML string to extract the bot response div, and injects it into the live chat box — all without a page reload.

```javascript
const response = await fetch('/generate', { method: 'POST', body: formData });
const html = await response.text();
// Parse returned page HTML and extract just the bot message node
const doc = new DOMParser().parseFromString(html, 'text/html');
const botMsgText = doc.querySelector('.message.bot')?.innerHTML;
```

---

## 🐳 Docker Deployment

### Build

```bash
# Run from repo root
docker build -f docker/Dockerfile.backend -t mental-health-llama .
```

### Run

```bash
# GPU (recommended)
docker run -p 8000:8000 --gpus all mental-health-llama

# CPU only (slow)
docker run -p 8000:8000 mental-health-llama
```

Open `http://localhost:8000` to access the chat interface.

---

## 🚀 Local Setup (Without Docker)

### Prerequisites

- Python 3.11+
- CUDA-capable GPU with ≥ 16 GB VRAM recommended (bfloat16 full model)
- HuggingFace account with access to `NousResearch/Llama-2-7b-chat-hf`

### Install & Run

```bash
git clone https://github.com/arjunravi26/mental-health-finetuned-llama.git
cd mental-health-finetuned-llama

pip install -r requirements.txt

uvicorn backend.apps.main:app --host 0.0.0.0 --port 8000 --reload
```

### Re-run Fine-Tuning

Open `research/llama_finetune.ipynb` in Google Colab (GPU runtime) and install dependencies in the first cell:

```bash
pip install accelerate peft bitsandbytes transformers trl triton
```

---

## 🤗 Use the Model Directly

The LoRA adapter is publicly available on HuggingFace Hub:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base_model, "ArjunRavi/mental-health-finetune")
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

prompt = """<s>[INST] <<SYS>>
You are a helpful and empathetic assistant.
<</SYS>>

I feel overwhelmed and cannot focus on anything.
[/INST]"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.5, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📦 Dependencies

| Package | Role |
|---|---|
| `transformers` | Model loading, tokenizer, `pipeline` |
| `peft` | LoRA adapter injection and loading (`PeftModel`) |
| `bitsandbytes` | 4-bit NF4 quantization (`BitsAndBytesConfig`) |
| `trl` | `SFTTrainer` for supervised fine-tuning |
| `accelerate` | Device mapping and mixed-precision coordination |
| `datasets` | HuggingFace dataset loading and `.map()` preprocessing |
| `fastapi` | API framework |
| `uvicorn` | ASGI server |
| `jinja2` | Server-side HTML template rendering |
| `python-multipart` | Form data parsing for `POST /generate` |
| `pydantic-settings` | Settings class with env override support |
| `torch` | Core tensor ops and CUDA inference |

---

## 🔬 Technical Notes & Reviewer Observations

| Area | Observation |
|---|---|
| **Prompt template mismatch** | `generate_text()` in `inference.py` passes raw user text without the LLaMA 2 `[INST]` wrapper. Wrapping as `<s>[INST] {prompt} [/INST]` before tokenization would match training format and improve response quality noticeably. |
| **`dtype` kwarg** | `inference.py` uses `dtype=torch.bfloat16` but the correct `from_pretrained` kwarg is `torch_dtype`. Passing an unrecognized kwarg may silently fall back to `float32`, doubling VRAM usage. |
| **Dockerfile COPY paths** | `COPY ../backend/requirements.txt .` uses a relative path from the `docker/` build context. Running `docker build -f docker/Dockerfile.backend .` from the repo root will fail. Restructuring to `COPY backend/ backend/` with build context at root resolves this. |
| **No streaming** | `/generate` is synchronous — the browser waits for the full response. `StreamingResponse` with a custom HuggingFace `TextIteratorStreamer` would deliver tokens in real time. |
| **Logger not wired** | `logger.py` exists but is not imported in `inference.py` or `routes.py`. Logging inference latency, errors, and token counts would be valuable in production. |
| **Model warm-up** | The model is loaded on first import, meaning the first request after server start can take 30–60s on slow hardware. A FastAPI `lifespan` startup event with a dummy forward pass would eliminate this cold-start for users. |

---

## 🗺️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
│                                                         │
│  HuggingFace Dataset                                    │
│  (Amod/mental_health_counseling_conversations)          │
│        │                                                │
│        ▼                                                │
│   format_dataset()  →  LLaMA 2 [INST] template         │
│   clean_html()      →  strip HTML artifacts             │
│   tokenize()        →  truncate to 512 tokens           │
│        │                                                │
│        ▼                                                │
│   LLaMA 2 7B Chat (NF4 4-bit quantized)                │
│   + QLoRA (r=8, α=16, 7 projection layers)             │
│   + SFTTrainer (paged_adamw_32bit, cosine LR)           │
│        │                                                │
│        ▼                                                │
│   ArjunRavi/mental-health-finetune  →  HF Hub          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    Serving Stack                         │
│                                                         │
│   Docker (python:3.11-slim, port 8000)                  │
│   └── Uvicorn                                           │
│        └── FastAPI                                      │
│             ├── GET  /      →  index.html (Jinja2)      │
│             ├── POST /generate                          │
│             │    └── generate_text(prompt)              │
│             │         ├── tokenizer(prompt)             │
│             │         ├── model.generate(...)           │
│             │         └── tokenizer.decode(outputs)     │
│             └── /static  →  CSS + JS                   │
└─────────────────────────────────────────────────────────┘
```

---

## 🔗 References

- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — Touvron et al. (2023)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al. (2023)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [PEFT Documentation](https://huggingface.co/docs/peft)

---

## ⚠️ Disclaimer

This model is built for **research and educational purposes only**. It is **not a substitute for professional mental health care, diagnosis, or treatment**. If you or someone you know is experiencing a mental health crisis, please reach out to a qualified mental health professional or contact a crisis helpline in your region.

---

## 👤 Author

**Arjun Ravi**
AI Engineer @ trivoxlabs | specializing in SLMs, local LLM inference, and document extraction pipelines.

- GitHub: [@arjunravi26](https://github.com/arjunravi26)
- HuggingFace: [ArjunRavi](https://huggingface.co/ArjunRavi)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
