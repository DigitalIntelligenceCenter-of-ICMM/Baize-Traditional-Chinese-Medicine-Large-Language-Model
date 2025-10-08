# 白泽中医药大语言模型  
**Baize-Traditional-Chinese-Medicine-Large-Language-Model**

🩺 专注中医药领域的开源大语言模型（TCM-LLM）  
📚 支持多版本语料训练 | 🧠 包含 0.6B 与 8B 系列模型 | 🔧 支持 LoRA 微调与 4bit/16bit 推理

> 🐉 “白泽”取自中国古代通晓万物的神兽，寓意“通晓中医，智启未来”。本项目致力于构建面向中医药领域的智能问答与知识理解大模型。

---

## 📚 项目简介

本项目发布一系列基于开源大模型微调的**中医药领域专用语言模型**，涵盖从小参数到大模型、从量化到全精度的多种配置，支持研究、教学与非商业应用。模型基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 框架进行高效微调，使用自研的 **Baize-TCM-Corpus** 系列语料库训练，具备良好的中医知识理解与问答能力。

---

## 🏗️ 模型概览

| 模型名称 | 基础模型 | 参数量 | 训练语料 | 量化方式 | 推理精度 | 下载链接 |
|--------|----------|--------|----------|----------|----------|----------|
| `Baize-Traditional-Chinese-Medicine-Large-Language-Model` | Qwen3-0.6B | ~0.6B | V3 | 未量化 | FP16 | [Hugging Face](https://huggingface.co/DigitalIntelligenceCenter-of-ICMM/Baize-Traditional-Chinese-Medicine-Large-Language-Model) |
| `[Baize-Traditional-Chinese-Medicine-Large-Language-Model-V1-4bit]` | Qwen3-8B | ~8B | V1 | 4bit (NF4) | Int4 推理 | [Hugging Face](https://huggingface.co/DigitalIntelligenceCenter-of-ICMM/Baize-Traditional-Chinese-Medicine-Large-Language-Model-V1-4bit) |
| `Baize-Traditional-Chinese-Medicine-Large-Language-Model-V1-16bit` | Qwen3-8B  | ~8B | V1 | 未量化 | FP16 | [Hugging Face](https://huggingface.co/DigitalIntelligenceCenter-of-ICMM/Baize-Traditional-Chinese-Medicine-Large-Language-Model-V1-16bit)) |
| `Baize-Traditional-Chinese-Medicine-Large-Language-Model-V2-4bit` | Qwen3-8B  | ~8B | V2 | 4bit (NF4) | Int4 推理 | [Hugging Face](https://huggingface.co/DigitalIntelligenceCenter-of-ICMM/Baize-Traditional-Chinese-Medicine-Large-Language-Model-V2-4bit)) |
| `Baize-Traditional-Chinese-Medicine-Large-Language-Model-V2-16bit` | Qwen3-8B  | ~8B | V2 | 未量化 | FP16 | [Hugging Face](https://huggingface.co/DigitalIntelligenceCenter-of-ICMM/Baize-Traditional-Chinese-Medicine-Large-Language-Model-V2-16bit) |
| `Baize-Traditional-Chinese-Medicine-Large-Language-Model-V3-4bit` | Qwen3-8B  | ~8B | V3 | 4bit (NF4) | Int4 推理 | [Hugging Face](https://huggingface.co/DigitalIntelligenceCenter-of-ICMM/Baize-Traditional-Chinese-Medicine-Large-Language-Model-V3-16bit) |
| `Baize-Traditional-Chinese-Medicine-Large-Language-Model-V3-16bit` | Qwen3-8B  | ~8B | V3 | 未量化 | FP16 | [Hugging Face](https://huggingface.co/DigitalIntelligenceCenter-of-ICMM/Baize-Traditional-Chinese-Medicine-Large-Language-Model-V3-16bit) |

> ✅ 所有 8B 模型均采用 **LoRA 微调**，原始主干冻结，适配器权重独立发布。

---

## 🧾 训练详情

- **微调框架**：[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- **训练方式**：LoRA（`r=64, lora_alpha=128, target_modules=q_proj,v_proj`）
- **上下文长度**：2048 tokens
- **语料版本说明**：
  - **V1**：初版语料，4,000+ QA 对
  - **V2**：扩展至 10,578 条，优化术语一致性
  - **V3**：进一步清洗与扩展(157,438对)，覆盖更多临床与经典文献场景

---

## 🚀 快速使用

### 1. 加载 4bit 量化模型（低显存）

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "your-username/baize-tcm-8b-v3-4bit",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("your-username/baize-tcm-8b-v3-4bit", trust_remote_code=True)


### 2. 加载 16bit 模型（高精度推理）
python
model = AutoModelForCausalLM.from_pretrained(
    "your-username/baize-tcm-8b-v3-16bit",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("your-username/baize-tcm-8b-v3-16bit", trust_remote_code=True)
