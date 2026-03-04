# Complete LLM Fine-Tuning & Deployment Playbook

Most organizations experimenting with large language models hit the same wall: base models don't follow domain tone, policy, or structure. RAG alone doesn't fix behavior—JSON schemas, refusals, and style remain inconsistent. Full fine-tuning is too expensive, and even good models fail if latency and VRAM costs aren't considered.

---
![](Finetuning.png)

---

## What this repository demonstrates 

### Decision intelligence: Finetune vs RAG vs Agents
**Rule of thumb**:
- Use **Fine-Tuning** → when behavior is wrong
- Use **RAG** → when knowledge is missing
- Use **Agents** → when multi-step reasoning/tools are needed

This distinction is explicitly covered in the notebooks.

---
## 1) Knowledge Distillation
**What:** A large **teacher** model trains a smaller **student** to mimic its outputs (often using soft probabilities).

**Why it matters**
- Large models are too slow/costly to serve.
- A smaller student cuts latency/cost while retaining most quality.

**How**
- Run teacher on representative prompts → collect soft targets.
- Train student to match teacher distributions (often KL loss) + optional task loss.

**Key knobs / gotchas**
- **Temperature (T):** smooths teacher outputs; can stabilize training.
- **Loss mix:** KL vs hard-label weighting.
- **Student size:** too small → quality drop; too big → limited savings.
- Dataset must match production traffic.

---

## 2) Parameter-Efficient Fine-Tuning (LoRA / QLoRA)
**What:** Train tiny **adapters** while freezing the base model.

**Why it matters**
- Full fine-tuning is expensive (updates billions of weights).
- LoRA adapts behavior with far less compute and memory.

**How (QLoRA)**
- Load the base model in **4-bit**.
- Train adapters in higher precision.
- Enables fine-tuning on limited GPUs (setup-dependent).

**Key knobs**
- `lora_r` → capacity vs memory
- `lora_alpha` → update scaling
- `lora_dropout` → regularization
- `target_modules` → where learning happens
- `gradient_accumulation_steps` → batch simulation
- `cutoff_len` → context length vs VRAM

---

## 3) Instruction Fine-Tuning (SFT)
**What:** Supervised training on **instruction → input → response** examples.

**Why it matters**
- Base models don’t reliably follow format or domain tone.
- Prompt-only solutions are brittle and costly (long prompts, retries).

**How**
- Build Alpaca/ShareGPT/custom JSONL datasets.
- Train to predict the target response (supervised next-token objective).
- Use consistent templates so structure is learnable.

**Key knobs / gotchas**
- Data quality dominates: dedup, consistency, schema-correct examples (if JSON).
- `cutoff_len` raises quality coverage but increases VRAM.
- Overtraining can cause “style lock-in” or drift.

---

## 4) Preference Alignment (DPO)
**What:** Train on **(prompt, chosen, rejected)** pairs to improve response quality.

**Why it matters**
- “Correct” answers can still be unhelpful: weak structure, poor tone, messy refusals.
- DPO improves preference alignment without a separate reward model.

**How**
- Collect preference pairs (human review or ranked candidates).
- Optimize the model to prefer chosen over rejected.

- `beta` controls preference strength:
- higher → stronger shift, higher overfit risk v/s lower → weaker shift, safer retention


---

## 5) Quantization
**What:** Reduce weight precision (8-bit/4-bit) to cut VRAM and speed inference.

**Why it matters**
- Deployment is constrained by latency + VRAM + cost.
- Enables CPU/edge deployment for some scenarios.

**How (common paths)**
- **bitsandbytes:** quick 8-bit/4-bit inference path (runtime-dependent).
- **GPTQ / AWQ:** GPU-focused post-training quantization (kernel/runtime support matters).
- **GGUF:** CPU/Apple Silicon via **llama.cpp**.

---

## 6) Domain-Specific Fine-Tuning (continued pretraining)
**What:** Continue pretraining on domain text to internalize terminology and patterns.

**Why it matters**
- General models often miss domain conventions and jargon.
- Improves downstream SFT generalization with fewer examples.

**How**
- Clean and dedup domain corpus (docs, tickets, manuals).
- Continue pretraining (next-token objective), then run SFT for instruction style.

**Key knobs / gotchas**
- Too much domain pretraining can cause catastrophic forgetting.
- This is not “loading PDFs” (that’s RAG) — it changes model priors.

---

## 7) Hugging Face (workflow unifier)
**What:** A standard ecosystem for datasets, training, adapters, and distribution.

**Why it matters**
- Reduces workflow fragmentation and improves reproducibility.
- Makes it easy to version models/adapters and share training configs.

**How**
- `datasets` for data loading/versioning
- `transformers` for training/inference
- `peft` for LoRA/QLoRA
- `trl` for DPO
- Hub (or internal registry) to version artifacts

---

## Repository Highlights

- **End-to-end:** `Final_Finetuning_all_in_one.ipynb`
- **Knowledge distillation:** `Knowledge_DIstillation_in_Deep_Learning.ipynb`
- **PEFT & Hugging Face:** `huggingface_solution.ipynb`, `huggingface_crash_course.ipynb`
- **Alignment:** `Preference_Aligned_Training_DPO_final.ipynb`
- **Quantization:** GPTQ, AWQ, GGUF notebooks
- **Domain-specific:** Instruction tuning on PDF and pharma datasets

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
