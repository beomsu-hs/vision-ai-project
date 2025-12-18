import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from deep_translator import GoogleTranslator

# ================= 설정값 =================
lora_path = "./model/minhwa_v1.safetensors"

# 1. 퀄리티를 깎아먹는 요소들을 강력하게 차단하는 네거티브 프롬프트
negative_prompt = (
    "low quality, worst quality, sketch, blurry, ugly, distorted, "
    "photorealistic, 3d render, cgi, modern city, smooth, shiny metal, "
    "text, watermark, human face, flesh, mutations, deformed, bad anatomy"
)

# 2. 테스트할 프롬프트 
user_inputs = [
    ("dog1", "소나무 아래 앉아있는 강아지")

]

# 3. 퀄리티 강제 향상 (자동으로 뒤에 붙음)
cheat_code_suffix = ", minhwa style, (masterpiece, best quality:1.4), (intricate details:1.2), (traditional korean ink painting:1.3), (hanji paper texture:1.3), rough brush strokes, flat perspective, vivid colors"
# =================================================

print("모델 로드 중...")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

#  스케줄러 업그레이드 (DPM++ 2M SDE Karras)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="sde-dpmsolver++"
)
pipe.to("cuda")

print(f" LoRA 로드 중: {lora_path}")
pipe.load_lora_weights(lora_path)

translator = GoogleTranslator(source='ko', target='en')

print(f" 총 {len(user_inputs)}장의 '고화질' 이미지를 생성합니다.\n")

for name, korean_prompt in user_inputs:
    
    print(f" 번역 중: '{korean_prompt}'")
    try:
        translated_prompt = translator.translate(korean_prompt)
    except Exception as e:
        print(" 번역 실패! 기본 영어 프롬프트를 사용합니다.")
        translated_prompt = "minhwa art"
        
    print(f"   ㄴ 영어: '{translated_prompt}'")

    final_prompt = translated_prompt + cheat_code_suffix
    
    print(f"▶ 그리는 중... (해상도: 512x768)")

    image = pipe(
        final_prompt,
        negative_prompt=negative_prompt,
        width=512,              # 너비
        height=768,             # 높이 (세로로 길게 -> 민화 구도 최적화)
        num_inference_steps=40, # 붓질 횟수 증가 (30 -> 40)
        guidance_scale=8.0,     # 프롬프트 충실도 상향
        cross_attention_kwargs={"scale": 0.9} # LoRA 강도 (너무 세면 깨져서 0.9로 미세 조정)
    ).images[0]
    
    save_name = f"result_pro_{name}.png"
    image.save(save_name)
    print(f" 저장 완료: {save_name}\n")

print("고화질 작업 완료!")
