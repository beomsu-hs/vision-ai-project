import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import os
import random # 랜덤 시드용

# ================= 설정값 =================
# 이미지 경로
input_image_path = "result_pro_Eiffel Tower.png" 

# 결과물 파일 이름
output_video_name = "video_tower1.mp4"

# 움직임 강도 (1 ~ 255)
motion_bucket_id = 70 

noise_aug_strength = 0.15
# ============================================

print("SVD(비디오) 모델 로드 중... (처음엔 다운로드 때문에 오래 걸립니다)")

# SVD 모델 사용 - 14프레임 생성 가능
model_id = "stabilityai/stable-video-diffusion-img2vid"

pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    variant="fp16"
)

# VRAM 최적화
pipe.enable_model_cpu_offload()

# 입력 이미지 로드 및 리사이징
# SVD는 특정 해상도(1024x576 등)를 선호합니다. 민화(세로) 비율에 맞춰 조정합니다.
print(f"이미지 읽는 중: {input_image_path}")
image = load_image(input_image_path)
image = image.resize((576, 1024)) # 세로형 영상 표준 해상도

print("영상 생성 시작 (약 2~3분 소요)")

seed = random.randint(0, 100000)
generator = torch.manual_seed(seed)

frames = pipe(
    image, 
    decode_chunk_size=2,    # VRAM 절약용
    generator=generator,
    motion_bucket_id=motion_bucket_id,
    noise_aug_strength=noise_aug_strength,
    num_inference_steps=12, # 12번만 그려서 속도 확보
    num_frames=14,         # 생성할 프레임 수 (약 2~3초 분량)
    fps=3
).frames[0]

# 영상 저장
print("MP4 파일로 저장 중...")
export_to_video(frames, output_video_name, fps=7)

print(f"영상 완성! 확인해보세요: {output_video_name}")