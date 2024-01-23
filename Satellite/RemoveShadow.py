# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import albumentations as A
# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[::2], mask_rle[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(shape)

def remove_shadow_dark_area(image_path):

    img = cv2.imread(image_path)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(l)
    
    #threshold=1.34*min_val+20.72
    threshold=np.mean(l)
    thm=np.median(l)
    th1= int(threshold/3)
    th2= int(threshold/3 * 2)
    th3= int(threshold)
    #print(min_val , max_val, thm, th1, th2, th3)
    # 수정된 부분: 이진화 임계값 조정
    threshold_value = int(0.01 * max_val)
    # 이미지 불러오기
    src = cv2.imread(image_path)

    # print("original")
    # print(l[l<threshold])
    # 명도가 특정 임계값보다 낮은 영역을 검정색으로 채우기
    l[l < th2] = l[l < th2] * 1.5
    # l[th1< l < th2] = l[th1< l < th2] * 1.5
    # l[th2< l < th3] = l[th2< l < th3] * 1.2
    # l[l < th3] = l[l < th3] * 1.2
    # l[l < th2] = l[l < th2] / 1.2 * 1.5
    # l[l < th1] = l[l < th1] / 1.5 * 2.2
    #print("change")
    #print(v[v<threshold])
    # 명도가 수정된 이미지를 사용하여 원래 이미지를 변환
    hls_modified = cv2.merge((h, l, s))
    modified_img = cv2.cvtColor(hls_modified, cv2.COLOR_HLS2BGR)

    return modified_img

f = open('./submit.csv','r')
rdr = csv.reader(f)
path = './test_img/'
num=-1

for line in rdr:
    num=num+1
    if num==15:
        break
    elif num==0:
        continue

    IMAGE_PATH = path + line[0] + '.png'
    MASK_RLE = line[1]

    #print(IMAGE_PATH)
    
    img = cv2.imread(IMAGE_PATH)
    img_copy=remove_shadow_dark_area(IMAGE_PATH)
    # plot으로 보여주기
    plt.figure(figsize=(20,8))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    #plt.subplot(1, 3, 2)
    #plt.imshow(mask)
    plt.subplot(1, 2, 2)
    plt.imshow(img_copy)
 
f.close()

# %%
