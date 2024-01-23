# %%
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import albumentations as at
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)
image = cv2.cvtColor(cv2.imread('./train_img/TRAIN_0001.png'), cv2.COLOR_BGR2RGB)
#plt.imshow(image)
# 이미지 배열을 입력받아 5개 출력하는 함수
def show_images(images, labels):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(25, 25))
    for i in range(2):
        axs[i].imshow(images[i])
        axs[i].set_title(labels[i])

# 원본 이미지를 입력받아 4개의 augmentation 적용하여 시각화
def aug_apply(image, label, aug):
    image_list = [image]
    label_list = ['origin']

    for i in range(4):
        aug_image = aug(image=image)['image']
        image_list.append(aug_image)
        label_list.append(label)

    show_images(image_list, label_list)

aug_multi = at.Compose([
    #at.ChannelShuffle(p=0.5), # RGB 채널 랜덤
    #at.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=5, num_shadows_upper=7, shadow_dimension=5, always_apply=True, p=1), #랜덤하게 그림자 넣기
    #at.CLAHE(p=1), #선명하게
    #at.Sharpen(p=1)
])

aug_apply(image=image, label='Multi', aug=aug_multi)
# %%
