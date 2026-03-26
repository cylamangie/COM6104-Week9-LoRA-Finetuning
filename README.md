# COM6104-Week9-LoRA-Finetuning
COM6104 Week 9 lab — Fine‑tuning Qwen3.5‑0.8B with LoRA using HuggingFace Transformers, PEFT, and Stable Baselines. Includes dataset preparation, training, and evaluation.

## 📌 Overview
This project demonstrates **Low‑Rank Adaptation (LoRA)** fine‑tuning of the **Qwen3.5‑0.8B** language model using HuggingFace Transformers and PEFT.  
It includes dataset preparation, LoRA configuration, training with HuggingFace `Trainer`, and evaluation of the fine‑tuned model.

> 📝 **Note**: This is the **nineth lab assignment** for **COM6104 – Topics in Data Science and Artificial Intelligence**.

---

## 🎯 Motivation
Large Language Models (LLMs) are powerful but expensive to fine‑tune.  
LoRA provides an efficient way to adapt models by training only a small fraction of parameters.  
This lab highlights:
- How to configure LoRA adapters for attention and MLP layers.  
- How to prepare custom dialogue datasets.  
- How to train with HuggingFace `Trainer`.  
- How to evaluate fine‑tuned models for new knowledge and catastrophic forgetting.

---

## ⚙️ Files
- **`lora.ipynb`** → Notebook with full workflow.  
---

## 📊 Key Steps
1. **Install dependencies**: `transformers`, `peft`, `bitsandbytes`, `accelerate`, `loralib`.  
2. **Load base model**: `Qwen/Qwen3.5-0.8B`.  
3. **Configure LoRA**: Target modules include attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP layers.  
4. **Prepare dataset**: Encode dialogues with `tokenizer.apply_chat_template`.  
5. **Train**: Run for 100 epochs with HuggingFace `Trainer`.  
6. **Evaluate**: Test fine‑tuned model on new prompts.  
7. **Save & reload**: Save adapter weights and reload with `PeftModel`.

---

## 🚀 How to Run
Install dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Course Context
Completed as part of COM6104 – Topics in Data Science and Artificial Intelligence at The Hang Seng University of Hong Kong.

## 💡 Reflection
This lab helped me understand how LoRA adapters enable efficient fine‑tuning of LLMs.
I learned how to configure target modules, prepare datasets, and evaluate whether the model learned new knowledge without catastrophic forgetting.
It reinforced the importance of careful dataset design and hyperparameter tuning in LLM adaptation.

## 📚 Acknowledgements
Parts of this code were adapted from COM6104 lab materials provided by the instructor.
This repository is licensed under the MIT License, which permits reuse and modification with proper attribution.
