# 🐯 Korean Folk Painting (Minhwa) Style LoRA
> **vision-ai-project**: Generating Traditional Korean Art with Stable Diffusion & LoRA

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Project Overview (프로젝트 개요)
이 프로젝트는 한국의 전통 예술인 **'민화(Minhwa)'**의 화풍을 학습한 생성형 AI 모델입니다.
Foundation Model인 **Stable Diffusion**에 **LoRA(Low-Rank Adaptation)** 기술을 적용하여, 적은 데이터로도 고유의 붓터치와 색감을 재현하는 것을 목표로 합니다.

### 🎯 Motivation (설정 이유 및 목표)
* **문제 의식:** 기존의 Text-to-Image 모델들은 서양화풍에는 강하지만, 한국적인 화풍(특히 민화의 질감, 오방색 등)을 정확히 구현하는 데 한계가 있습니다.
* **목표:** 공공 데이터를 활용하여 저작권 문제없는 한국형 이미지 생성 모델을 구축하고, 누구나 쉽게 한국적 디자인을 창작하도록 돕습니다.

---

## 🤖 Model Details (모델 상세 - System Card)

### Model Description
* **Base Model:** [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
* **Architecture:** LoRA (Low-Rank Adaptation) applied to UNet
* **Training Method:** Fine-tuning with DreamBooth / Kohya_ss
* **Developed by:** (본인의 이름 또는 팀명)
* **Shared on:** GitHub & Hugging Face

### Intended Use (사용 목적)
* 한국 전통 스타일의 일러스트레이션 제작
* 교육용 자료 및 디자인 소스 생성
* **Trigger Word:** 프롬프트에 `minhwa style`을 입력하여 스타일을 적용합니다.

---

## 💾 Dataset Preparation (데이터 제작 과정) - **Core Part**
본 프로젝트는 **고품질의 자체 데이터셋 구축**에 중점을 두었습니다.

### 1. Data Collection (수집)
* **Source:** [e-뮤지엄](https://www.emuseum.go.kr/), [국립중앙박물관](https://www.museum.go.kr/)
* **Copyright:** **공공누리 제1유형 (출처표시, 상업적 이용 가능, 변경 가능)** 데이터만 엄선하여 사용했습니다.
* **Quantity:** 고해상도 민화 이미지 (약 50~100장)

### 2. Preprocessing (전처리)
* 모든 이미지를 학습에 최적화된 `512x512` 픽셀로 Center Cropping 및 Resizing.
* RGB 채널 정규화(Normalization) 수행.

### 3. Captioning (캡션 제작)
단순 이미지 수집을 넘어, 정교한 스타일 학습을 위해 **(Image, Text) Pair** 데이터를 직접 제작했습니다.
1.  **Auto-Captioning:** `BLIP` 모델을 사용하여 기초 캡션 생성
2.  **Human Refinement:** 생성된 캡션에 `minhwa style`, `tiger`, `magpie`, `pine tree` 등 세부 객체와 스타일 태그를 수작업으로 보강.

---

## ⚙️ Training Procedure (학습 과정)

* **Environment:** Google Colab (T4 GPU)
* **Library:** Hugging Face `diffusers`, `peft`
* **Hyperparameters:**
    * `learning_rate`: 1e-4
    * `train_batch_size`: 1
    * `num_train_epochs`: (예: 50)
    * `lora_rank`: 4

---

## 📊 Evaluation & Results (결과 및 평가)

### Qualitative Analysis (정성 평가)
| Prompt | Base Model (SD 1.5) | Ours (Minhwa LoRA) |
| :---: | :---: | :---: |
| "A tiger sitting under a pine tree" | (기본 모델 생성 이미지) | (LoRA 적용 이미지) |
| "A cat in minhwa style" | (기본 모델 생성 이미지) | (LoRA 적용 이미지) |

> *결과 분석: 기본 모델은 실사 같은 호랑이를 그리지만, 본 LoRA 모델은 민화 특유의 해학적인 표정과 붓터치를 반영함.*

### Limitations & Bias (한계점)
* **Data Bias:** 학습 데이터가 '까치호랑이(Jakho-do)'에 편향되어 있어, 인물화나 산수화 생성 시 성능이 다소 떨어질 수 있음.
* **Resolution:** 512x512 해상도로 학습되어, 그 이상의 고해상도 생성 시 디테일이 뭉개질 수 있음.

---

## 🚀 How to Run (실행 방법)

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

# Load LoRA weights
pipe.load_lora_weights("./path/to/your/lora_weights")

prompt = "minhwa style, a cute dog sitting on a rock"
image = pipe(prompt).images[0]
image.save("result.png")
