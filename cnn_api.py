from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms


def adjust_hls(hls_image, lightness_scale=1.0, saturation_scale=1.0):
    h, l, s = cv2.split(hls_image)  # HLS 이미지를 각 채널로 분리
    l = np.clip(l * lightness_scale, 0, 255).astype(np.uint8)  # 밝기 조절
    s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)  # 채도 조절
    adjusted_hls = cv2.merge([h, l, s])  # 조정된 채널을 다시 합침
    return adjusted_hls


def classify_image(image_path):
    
    # 데이터셋을 사용하기 위한 transform 정의 , RESNET-50(224*224를 입력으로 받음)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 이미지 크기 조정
        transforms.ToTensor(),  # PIL 이미지를 PyTorch Tensor로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 이미지 정규화
    ])
    
    # 이미지 로드 및 전처리 코드
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)[:, :, ::-1] # RGB to BGR
    hls_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HLS)
    adjusted_hls_image = adjust_hls(hls_image, lightness_scale=1.2, saturation_scale=0.9)
    
    # 이미지 정규화 (픽셀 값의 범위를 0 ~ 1 사이로 조정)
    normalized_image = adjusted_hls_image / 255.0

    # 모델 로드 및 추론 코드
    image = Image.fromarray((normalized_image * 255).astype(np.uint8))
    image = transform(image)
    
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load("resnet50_binary_classification_model.pth"))
    input_tensor = image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        preds = torch.sigmoid(outputs) >= 0.5

    if preds == True:
        return "합성입니다."
    else:
        return "합성이 아닙니다."

app = FastAPI()

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # 업로드된 파일을 로컬 파일 시스템에 저장합니다.
    image_path = f"path/to/save/{file.filename}"
    with open(image_path, "wb") as buffer:
        contents = await file.read()
        buffer.write(contents)
    
    # 이미지 분류 함수 호출
    result = classify_image(image_path)
    
    # 결과 반환
    return {"result": result}