#  Living Minhwa: Generative AI for Korean Folk Painting
### "살아있는 민화: 한국 전통 예술의 생성형 AI 복원 및 미디어 아트화 프로젝트"

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat&logo=pytorch&logoColor=white)
![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow?style=flat&logo=huggingface&logoColor=white)
![ComfyUI](https://img.shields.io/badge/Tool-ComfyUI-purple?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

##  Project Overview

이 프로젝트는 **Stable Diffusion**과 **LoRA(Low-Rank Adaptation)** 기술을 활용하여 한국의 전통 '민화' 스타일을 학습하고, Stable Video Diffusion (SVD)를 통해 정적인 민화를 동적인 Media Art로 확장하는 **Multimodal Generative AI** 프로젝트입니다.

단순한 이미지 생성을 넘어, 텍스트 프롬프트로 민화를 창작하고, 이를 다시 영상으로 변환함으로써 잊혀져 가는 전통 예술에 새로운 디지털 가치를 부여하는 것을 목표로 합니다.

###  Objective & Motivation
* **Problem:** 기존 생성 모델(SD 1.5)은 서양 화풍에 편향되어 있어, '한국 호랑이'나 '소나무'를 그릴 때 민화 특유의 해학적 느낌과 붓터치 질감을 살리지 못함.
* **Solution:** 공공 데이터셋을 활용하여 저작권 문제없는 '민화 전용 LoRA'를 제작하고, 최신 SVD 기술로 생동감을 부여.
* **Key Tech:** `Stable Diffusion v1.5`, `LoRA`, `Stable Video Diffusion (SVD)`, `High-quality Captioning`.

---

##  Workflow & Pipeline

본 프로젝트는 데이터 수집부터 영상 생성까지 총 4단계의 파이프라인으로 구성되어 있습니다.

```mermaid
graph LR
    A[Data Collection<br>e-Museum] --> B[Preprocessing<br>Crop & Captioning]
    B --> C[LoRA Fine-tuning<br>Stable Diffusion]
    C --> D[Inference<br>Text-to-Image]
    D --> E[Image-to-Video<br>SVD via ComfyUI]
