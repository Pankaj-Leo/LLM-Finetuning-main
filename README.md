# Complete LLM Fine-Tuning & Deployment Playbook

**Instruction Tuning → PEFT (LoRA / QLoRA) → Preference Alignment (DPO)  
→ Quantization (GPTQ / AWQ / GGUF) → Production Deployment**


---

## Why this project exists (real production problem)

Most organizations experimenting with LLMs face the same failures:

-  Base LLMs don’t follow **domain tone, policy, or structure**
-  **RAG alone** doesn’t fix behavior (JSON schemas, refusals, safety, style)
-  Full fine-tuning is **too expensive** and hard to deploy
-  Even good models fail if **latency and VRAM costs** are ignored

This repository is a **complete, auditable blueprint** for building **deployable LLM systems**, not demos.

---

## What this repository demonstrates 

### 1. Decision intelligence: Finetune vs RAG vs Agents
**Rule of thumb**:
- Use **Fine-Tuning** → when behavior is wrong
- Use **RAG** → when knowledge is missing
- Use **Agents** → when multi-step reasoning/tools are needed

This distinction is explicitly covered in the syllabus and notebooks.

---

### 2. Instruction Fine-Tuning (SFT)
**Goal:** teach the model *how* to respond, not *what* to know.

Supported formats:
- Alpaca (instruction following)
- ShareGPT (multi-turn chat)

Benefits:
- Consistent formatting
- Domain tone enforcement
- Lower prompt complexity
- Reduced hallucination under constraints

---

### 3. Parameter-Efficient Fine-Tuning (LoRA / QLoRA)

**Why PEFT matters**
- Full fine-tuning updates billions of weights → expensive
- LoRA trains small low-rank adapters → efficient and stable

**QLoRA advantage**
- Base model loaded in **4-bit**
- Adapters trained in higher precision
- Enables fine-tuning on consumer GPUs

**Key knobs (from LLaMA Factory configs):**
- `lora_r` → capacity vs memory
- `lora_alpha` → update scaling
- `lora_dropout` → regularization
- `target_modules` → where learning happens
- `gradient_accumulation_steps` → batch simulation
- `cutoff_len` → context length tradeoff

---

### 4. Preference Alignment (DPO)

Instruction tuning answers *correctly*.  
Preference tuning answers *well*.

**DPO (Direct Preference Optimization)**:
- Trains on `(prompt, chosen, rejected)` pairs
- No reward model required
- Simpler and more stable than PPO-based RLHF

**Key concept: `beta`**
- Controls strength of preference enforcement
- Higher beta → stronger alignment, risk of overfitting
- Lower beta → safer but weaker preference shift

---

### 5. Quantization (deployment reality)

Fine-tuned models are useless if they can’t be served cheaply.

#### Quantization methods covered:
- **bitsandbytes** (easy 8-bit / 4-bit inference)
- **GPTQ** (gradient-aware post-training quantization)
- **AWQ** (activation-aware, often higher quality)
- **GGUF / GGML** (CPU & Apple Silicon via llama.cpp)

#### PTQ vs QAT:
- **PTQ**: fast, calibration-based, industry default
- **QAT**: higher fidelity, much more expensive

**Deployment decision matrix:**

| Goal | Best Option |
|---|---|
| Fast GPU inference | GPTQ / AWQ |
| Lowest VRAM | QLoRA / GPTQ |
| Laptop / Edge | GGUF + llama.cpp |
| High-throughput API | vLLM / TensorRT-LLM |

---

### 6. Multimodal Extension (Chapter 9)

This repo includes **Multimodal LLM foundations**:

- Vision encoders (CLIP / ViT)
- Vision-Language alignment
- Captioning, VQA, document understanding
- Cross-modal embedding fusion

This enables:
- PDF understanding
- Image-aware assistants
- Multimodal RAG pipelines

---

## Repository Map (what to review first)

### 🔹 End-to-End Pipeline
- `Final_Finetuning_all_in_one.ipynb`

### 🔹 Alignment
- `Preference_Aligned_Training_DPO_final.ipynb`

### 🔹 Quantization
- GPTQ / AWQ notebooks
- GGUF export workflows

### 🔹 Multimodal
- `Chapter 9 - Multimodal Large Language Models.ipynb`

### 🔹 Classical NLP (foundation credibility)
- BERT fine-tuning
- Embeddings & MTEB
- Knowledge distillation

---

## Evaluation (expected additions)

To make this production-grade, add:

- `results/`
  - before/after prompt comparisons
  - preference win-rate
  - latency & VRAM table (FP16 vs INT8 vs INT4)
  - quantization calibration notes

---

## References

- [Jay Alammar & Maarten Grootendorst, *Hands-On Large Language Models*, O’Reilly](https://www.amazon.com/Hands-Large-Language-Models-Understanding/dp/1098150961)  
- Hugging Face Transformers & PEFT  
- LLaMA Factory Documentation  
- GPTQ: Frantar et al.  
- AWQ: Lin et al.  
- llama.cpp / GGUF format  
- TRL (HF Reinforcement Learning Library)
- [Sunny Savita LLM Finetuning](https://www.youtube.com/playlist?list=PLQxDHpeGU14BX9L83JMSoKHsdIjv98BGn)

---
**Pankaj Somkuwar**  
AI Engineer | AI Product Manager | AI Solutions Architect  

- LinkedIn: https://www.linkedin.com/in/pankaj-somkuwar/  
- GitHub: https://github.com/Pankaj-Leo  
- Website: https://www.pankajsomkuwarai.com  
- Email: pankaj.som1610@gmail.com  

## License
Educational / research use. Verify redistribution rights for third-party datasets.

---