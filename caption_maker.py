import os

base_path = "C:/Users/MSI/OneDrive/Desktop/project_minhwa/image"

# 1. 호랑이 폴더 (60_minhwa style)
tiger_folder = os.path.join(base_path, "60_minhwa style")
tiger_caption = "minhwa style, a fierce tiger with detailed fur, traditional korean painting, texture of hanji paper"

# 2. 나머지 폴더 (20_minhwa style)
others_folder = os.path.join(base_path, "20_minhwa style")
others_caption = "minhwa style, traditional korean painting, texture of hanji paper, flat perspective"

def create_captions(folder_path, default_caption):
    if not os.path.exists(folder_path):
        print(f"폴더를 찾을 수 없습니다: {folder_path}")
        return

    files = os.listdir(folder_path)
    count = 0
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            file_name = os.path.splitext(file)[0]
            txt_path = os.path.join(folder_path, f"{file_name}.txt")
            
            # 이미 텍스트 파일이 있으면 건너뜀 (덮어쓰기 방지)
            if not os.path.exists(txt_path):
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(default_caption)
                count += 1
    
    print(f" {folder_path} : {count}개의 캡션 파일 생성 완료!")

# 실행
print("--- 캡션 생성 시작 ---")
create_captions(tiger_folder, tiger_caption)
create_captions(others_folder, others_caption)
print("--- 작업 완료. 이제 텍스트 내용을 수정하세요! ---")