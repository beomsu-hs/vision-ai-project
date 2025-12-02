# 🐯 Living Minhwa: Generative AI for Korean Folk Painting
### "살아있는 민화: 한국 전통 예술의 생성형 AI 복원 및 미디어 아트화 프로젝트"

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c) ![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow) ![ComfyUI](https://img.shields.io/badge/Tool-ComfyUI-purple)

## 📖 Project Overview (프로젝트 개요)

이 프로젝트는 **Stable Diffusion**과 **LoRA(Low-Rank Adaptation)** 기술을 활용하여 한국의 전통 '민화(Minhwa)' 스타일을 학습하고, **Stable Video Diffusion (SVD)**를 통해 정적인 민화를 동적인 영상(Media Art)으로 확장하는 멀티모달 생성 AI 프로젝트입니다.

단순한 이미지 분류를 넘어, **텍스트 프롬프트로 민화를 생성(Text-to-Image)**하고, 이를 **영상으로 변환(Image-to-Video)**함으로써 전통 예술에 새로운 디지털 가치를 부여하는 것을 목표로 합니다.

### 🎯 Objective & Motivation
- **문제 의식:** 전통 예술 데이터는 디지털화되어 있으나, 이를 현대적인 콘텐츠로 재생산할 수 있는 AI 모델은 부족함.
- **해결 방안:** 공공 데이터를 활용해 '민화 전용 LoRA'를 제작하고, 최신 SVD 기술로 생동감을 불어넣음.
- **핵심 기술:** Foundation Model (SD 1.5, SVD), LoRA Fine-tuning, High-quality Captioning.

---

## 🚀 Workflow & Pipeline

본 프로젝트는 총 5단계의 파이프라인으로 구성되어 있습니다.

```mermaid
graph LR
    A[Data Collection<br>(e-Museum)] --> B[Preprocessing<br>(Crop & Captioning)]
    B --> C[LoRA Fine-tuning<br>(Stable Diffusion)]
    C --> D[Inference<br>(Text-to-Image)]
    D --> E[Image-to-Video<br>(SVD via ComfyUI)]
1. Data Collection & PreprocessingSource: 국립중앙박물관 e-뮤지엄 (공공누리 1유형 및 저작권 만료 데이터 엄선)Selection: '까치호랑이', '화조도' 등 민화의 특징이 뚜렷한 고해상도 이미지 70장 선별.Preprocessing: - 512x512 / 768x768 Center Crop.RGB Convert 및 Normalize.2. Data Creation: High-Quality Captioning (핵심 과정)단순한 이미지 수집을 넘어, 모델이 스타일을 정확히 학습하도록 정교한 캡션 데이터를 직접 제작했습니다.Trigger Word: minhwa style (스타일 발현을 위한 핵심 키워드)Process: BLIP 모델을 이용해 초안을 생성한 후, 민화적 요소(소나무, 까치, 털의 질감 등)를 수동으로 보정.Example:Before: A tiger and a bird on a tree.After: minhwa style, a fierce tiger with detailed fur sitting under an old pine tree, a magpie looking down, traditional korean painting paper texture.3. Model Training (Fine-tuning)Base Model: Stable Diffusion v1.5Method: LoRA (Low-Rank Adaptation)Environment: NVIDIA RTX 4060 Laptop (8GB VRAM)Hyperparameters:Rank (dim): 32Alpha: 16Learning Rate: 1e-4Batch Size: 1 (Gradient Accumulation 활용)4. Multimodal Expansion: Image-to-Video생성된 정적 이미지를 ComfyUI 환경에서 SVD (Stable Video Diffusion) 모델에 입력하여 움직이는 민화로 변환합니다.Optimization: 8GB VRAM 환경에서의 구동을 위해 WebUI 대신 메모리 효율이 높은 ComfyUI 사용.Settings: 1024x576 Resolution, 25 Frames, Motion Bucket ID 127.💻 How to Run (실행 방법)PrerequisitesPython 3.10+PyTorch with CUDA supportComfyUI (for SVD)InstallationBashgit clone [https://github.com/your-username/vision-ai-project.git](https://github.com/your-username/vision-ai-project.git)
cd vision-ai-project
pip install -r requirements.txt
Inference (Python Script)Pythonfrom diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# Load LoRA
pipe.load_lora_weights("./lora_weights/minhwa_lora.safetensors")

prompt = "minhwa style, a cute cat playing with a butterfly, pine tree background"
image = pipe(prompt).images[0]
image.save("result.png")
🎨 Results ShowcaseText-to-Image (LoRA)Prompt: "A tiger smoking a pipe"Prompt: "A modern city landscape"(Note: 민화 스타일이 적용된 결과물)Image-to-Video (SVD)Motion: "Blinking eyes & Moving branches"(Click to watch full video)📋 Hugging Face System CardModel DetailsModel Name: Minhwa-Style-LoRA-v1Architecture: Stable Diffusion v1.5 based LoRALicense: CreativeML Open RAIL-MIntended Use한국 전통 디자인 패턴 생성교육용 자료 및 미디어 아트 전시비상업적 용도 권장 (학습 데이터의 저작권은 만료되었으나, 생성물의 윤리적 활용 필요)Limitations & Biases데이터 편향: '까치호랑이' 위주의 데이터로 학습되어, 인물화나 산수화 생성 시 호랑이의 특징(털 질감 등)이 섞여 나올 수 있음.해상도 한계: SD 1.5 기반이므로 텍스트 묘사가 뭉개지는 현상 발생 가능.🛠 Tech Stack & ToolsFramework: PyTorch, DiffusersTraining: Kohya_ss / DreamboothInference & Workflow: ComfyUIHardware: NVIDIA GeForce RTX 4060 Laptop GPU
