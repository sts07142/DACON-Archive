# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[::2], mask_rle[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(shape)

f = open('./submit.csv','r')
rdr = csv.reader(f)
path = './test_img/'
num=-1

for line in rdr:
    num=num+1
    if num==20:
        break
    elif num==0:
        continue

    IMAGE_PATH = path + line[0] + '.png'
    MASK_RLE = line[1]

    #print(IMAGE_PATH)
    
    img = cv2.imread(IMAGE_PATH)
    img_copy = img.copy()
    mask = rle_decode(MASK_RLE.split(" "), shape=(224,224))

    # 컨투어 검출하기
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_copy, contours, -1, (0, 0, 255), 1)

    # dst = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)

    # plot으로 보여주기
    plt.figure(figsize=(20,8))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.subplot(1, 3, 3)
    plt.imshow(img_copy)
 
f.close()
# %%
