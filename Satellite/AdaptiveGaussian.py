# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 컨투어 검출하기
def contoursDelet(mask):
  mask = np.array(mask)
  contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # cv2.drawContours(img_copy, contours, -1, (0, 0, 255), 1)

  # Threshold를 지정하세요.
  threshold = 8

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

original = cv2.imread('./train_img/TRAIN_0001.png')
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
original = cv2.resize(original,(224,224))

img = cv2.imread('./train_img/TRAIN_0001.png', 0)
img = cv2.resize(img,(224,224))
img = cv2.medianBlur(img, 5) # 잡음제거

#th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11, 5)

th3 = cv2.adaptiveThreshold(img, 1000, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,5, 4)

th4 = contoursDelet(th3)

titles = ['Original Image', 'Simple Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [original, th4]
plt.figure(figsize=(50,50))
for i in range(2):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
# %%
