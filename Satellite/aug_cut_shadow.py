import os
import cv2
import pandas as pd
import numpy as np
import random
import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from efficientnet_pytorch import EfficientNet
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_PATH = "./train_img_cut"
CSV_PATH = "./train_patch.csv"
BATCH_SIZE = 16
NUM_WORKERS = 6
EPOCH = 100
BACKBONE = ResNet50
MODEL_NAME = "UNet_ResNet50_shadow_100"
# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 그림자 명도 높이기 함수
def remove_shadow_dark_area(image_path):

    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(v)

    threshold=1.34*min_val+20.72

    # 수정된 부분: 이진화 임계값 조정
    threshold_value = int(0.01 * max_val)
    # 이미지 불러오기
    src = cv2.imread(image_path)

    # 이미지를 HSV로 변환
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 명도가 특정 임계값보다 낮은 영역을 검정색으로 채우기
    v[v < threshold] = v[v < threshold]*1.3

    # 명도가 수정된 이미지를 사용하여 원래 이미지를 변환
    hsv_modified = cv2.merge((h, s, v))
    modified_img = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)

    return modified_img

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
class SatelliteDataset(Dataset):
    def __init__(self, img_folder, csv_file, num_samples=14000, transform=None, infer=False):
        self.all_img_paths = sorted(glob.glob(os.path.join(img_folder, '*')))
        self.csv_file = csv_file
        self.img_folder = img_folder
        self.num_samples = min(num_samples, len(self.all_img_paths))
        self.update_samples()
        self.transform = transform
        self.infer = infer

        # transforms
        self.default_transform = Compose([
            Resize(224, 224),
            Normalize(),
            ToTensorV2()
        ])

    def update_samples(self):
        df = pd.read_csv(self.csv_file)
        df = df.sample(n=self.num_samples)
        self.img_paths = [os.path.join(self.img_folder, f"{img_id}.png") for img_id in df['img_id']]
        self.mask_rle_list = df['mask_rle'].tolist()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        image = cv2.imread(img_path)
        image = remove_shadow_dark_area(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            else:
                image = self.default_transform(image=image)['image']
            return image

        mask_rle = self.mask_rle_list[idx]
        if(mask_rle==""):
            mask_rle=-1
        mask = rle_decode(mask_rle, shape=(224, 224))
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            augmented = self.default_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
dataset = SatelliteDataset(img_folder = IMAGE_PATH, csv_file = CSV_PATH, transform=aug_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# U-Net의 기본 구성 요소인 Double Convolution Block을 정의합니다.
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )
class UNet(nn.Module):
    def __init__(self, backbone_name=BACKBONE, classes=1, encoder_weights='imagenet'):
        super(UNet, self).__init__()

        #self.backbone = EfficientNet.from_pretrained(backbone_name, weights_path=None)
        self.backbone = BACKBONE

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, classes, 1)

    def forward(self, x):
        # Backbone feature extraction
        #backbone_features = self.backbone.extract_features(x)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

from tqdm import tqdm

model = UNet(backbone_name=BACKBONE, encoder_weights='imagenet').to(device)

# loss function과 optimizer 정의
criterion = torch.nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# training loop
for epoch in range(EPOCH):
    dataset.update_samples()
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader)
    for idx, (images, masks) in enumerate(progress_bar):
        images = images.float().to(device)
        masks = masks.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.unsqueeze(1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_description(f'Epoch {epoch+1}, Loss: {epoch_loss/(idx+1)}')

    model_name = MODEL_NAME + str(epoch) + ".pth"
    torch.save(model.state_dict(), "./" + model_name)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')
## **픽셀의 숫자가 특정 개수 이하면 -1**
model_name = MODEL_NAME + ".pth"
torch.save(model.state_dict(), "./" + model_name)

# 컨투어 검출하기
def contoursDelet(mask):
  mask = np.array(mask)
  contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # cv2.drawContours(img_copy, contours, -1, (0, 0, 255), 1)

  # Threshold를 지정하세요.
  threshold = 100

  # Create an empty mask to store the pixels to be removed
  remove_mask = np.zeros_like(mask)

  # 각 컨투어를 순회하며 면적 계산
  for idx, contour in enumerate(contours):
      # 현재 컨투어가 최상위 컨투어이고(부모가 없고), 하위 컨투어(자식이)가 없으면 단일 구조
      if hierarchy[0][idx][3] == -1 and hierarchy[0][idx][2] == -1:
          area = cv2.contourArea(contour)
          #print(area)
          if area < threshold:
              cv2.drawContours(remove_mask, [contour], -1, 255, thickness=cv2.FILLED)

      # 현재 컨투어가 최상위 컨투어이고, 하위 컨투어(자식이)가 있으면 뚫린 구조
      elif hierarchy[0][idx][3] == -1 and hierarchy[0][idx][2] != -1:
          outer_area = cv2.contourArea(contour)
          inner_area = cv2.contourArea(contours[hierarchy[0][idx][2]])
          #print(outer_area - inner_area)
          if (outer_area - inner_area) < threshold:
              cv2.drawContours(remove_mask, [contour], -1, 255, thickness=cv2.FILLED)

  # Remove the small areas from the original mask
  final_mask = cv2.bitwise_and(mask, cv2.bitwise_not(remove_mask))

  return final_mask

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

        # transforms
        self.transform = transform
        self.default_transform = Compose([
            Resize(224, 224),
            Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]

        image = cv2.imread(img_path)
        image = remove_shadow_dark_area(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            else:
                image = self.default_transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            augmented = self.default_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

transform = A.Compose(
    [
        A.Resize(112, 112),
        A.Normalize(),
        ToTensorV2()
    ]
)

test_dataset = SatelliteDataset(csv_file='./test.csv',infer=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
model.load_state_dict(torch.load('./' + MODEL_NAME + '.pth', map_location=torch.device('cpu')))
model = model.to(device)
with torch.no_grad():
    model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to(device)

        outputs = model(images)
        masks = torch.sigmoid(outputs).cpu().numpy()
        masks = np.squeeze(masks, axis=1)
        masks = (masks > 0.2).astype(np.uint8) # Threshold = 0.35

        for i in range(len(images)):
            mask_resized = cv2.resize(masks[i], (224, 224))
            mask_deleted = contoursDelet(mask_resized)
            mask_rle = rle_encode(mask_deleted)
            if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                result.append(-1)
            else:
                result.append(mask_rle)
submit = pd.read_csv('./sample_submission.csv')
submit['mask_rle'] = result
submit.to_csv('./submit.csv', index=False)