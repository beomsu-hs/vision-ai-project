import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os

# ================= 설정값 =================
lora_path = "./model/minhwa_v1.safetensors" 
negative_prompt = "low quality, worst quality, blurry, distorted, ugly, watermark, text, human face, realistic photo"

# 테스트할 프롬프트 리스트
test_prompts = [
    # 1. [퓨전] 민화 스타일의 스포츠카
    ("sports_car", "minhwa style, a red sports car driving on a mountain road, pine trees, stylized clouds, traditional ink painting"),
    
    # 2. [퓨전] 민화 스타일의 아이언맨 (로봇)
    ("iron_man", "minhwa style, a mechanical armor hero standing on a scholar's rock, red and gold metal armor, tiger skin pattern cape, fierce pose"),
    
    # 3. [동물] 용 (Dragon) - 호랑이 친구
    ("dragon", "minhwa style, a mystical blue dragon flying in the clouds, holding a magic pearl, traditional korean art"),
    
    # 4. [사물] 스타벅스 커피 (찻잔)
    ("coffee", "minhwa style, a cup of coffee on a wooden table, peony flowers background, steam rising, vintage paper texture"),
    
    # 5. [풍경] 서울 N타워
    ("seoul_tower", "minhwa style, N Seoul Tower on top of Namsan mountain, pine trees, cherry blossoms, flat perspective")
]
# ==========================================

print(" 모델 로드 중...")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

print(f" LoRA 로드 중: {lora_path}")
pipe.load_lora_weights(lora_path)

print(f" 총 {len(test_prompts)}장의 이미지를 생성합니다.\n")

for name, prompt in test_prompts:
    print(f" 생성 중: {name}...")
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        cross_attention_kwargs={"scale": 1.0} # 강도 조절
    ).images[0]
    
    save_name = f"test_{name}.png"
    image.save(save_name)
    print(f"  저장 완료: {save_name}")

print("\n 모든 테스트가 완료되었습니다!")