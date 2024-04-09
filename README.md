## [SW중심대학 공동 AI 경진대회 2023](https://dacon.io/competitions/official/236092/overview/description)
- 위성 이미지의 건물 영역 분할(Image Segmentation)을 수행하는 AI모델을 개발
- SW중심대학 | 알고리즘 | 비전 | 객체분할 | DiceScore
- 2023.07.03 ~ 2023.07.28

## 데이터셋

<details>
<summary>
<b>train_img</b>
</summary>

    - TRAIN_0000.png ~ TRAIN_7139.png
    - 1024 x 1024
</details>

<details>
<summary>
<b>train.csv</b>
</summary>

    - TEST_00000.png ~ TEST_60639.png
    - 224 x 224
</details>

<details>
<summary>
<b>test_img</b>
</summary>

    - img_id : 학습 위성 이미지 샘플 ID
    - img_path : 학습 위성 이미지 경로 (상대 경로)
    - mask_rle : RLE 인코딩된 이진마스크(0 : 배경, 1 : 건물) 정보
        - 학습 위성 이미지에는 반드시 건물이 포함되어 있습니다.
        - 그러나 추론 위성 이미지에는 건물이 포함되어 있지 않을 수 있습니다.
        - 학습 위성 이미지의 촬영 해상도는 0.5m/픽셀이며, 추론 위성 이미지의 촬영 해상도는 공개하지 않습니다.
</details>

<details>
<summary>
<b>test.csv</b>
</summary>

    - img_id : 추론 위성 이미지 샘플 ID
    - img_path : 추론 위성 이미지 경로 (상대 경로)
</details>

<details>
<summary>
<b>sample_submission.csv</b>
</summary>

    - img_id : 추론 위성 이미지 샘플 ID
    - mask_rle : RLE 인코딩된 예측 이진마스크(0: 배경, 1 : 건물) 정보
        - 단, 예측 결과에 건물이 없는 경우 반드시 -1 처리
</details>

</details>
<br>

# [상위 25%] U-Net + ResNet50 + Contour

## 1. 데이터 처리
### 1-a. train 이미지 분할
- 1024 x 1024 -> 224 x 224 크기 이미지로 분할

### 1-b. Contour Outlier 처리

- 후처리 단계에서, 검출된 건물 영역 크기 기준 이상치 처리

## 2. 학습 이미지 변형
```
# Augmentation
aug_transform = Compose([
        Resize(224, 224),
        HorizontalFlip(p=0.5), # 50% 확률로 이미지를 수평으로 뒤집음
        A.VerticalFlip(p=0.5), # 50% 확률로 이미지를 수직으로 뒤집음
        A.Rotate(limit=30), # -30도에서 30도 사이의 각도로 이미지를 무작위로 회전
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5), # 컬러 변형
        Normalize(), # 이미지를 정규화
        ToTensorV2() # PyTorch tensor로 변환
    ]
)
    
```
## 3. U-Net + ResNet50
```
model = UNet(backbone_name='ResNet50', encoder_weights='imagenet').to(device)
```
