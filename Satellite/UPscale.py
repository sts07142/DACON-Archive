# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
def upscale(img_path):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel('./EDSR_x4.pb')
    sr.setModel('edsr', 4)

    #이미지 로드하기
    img = cv2.imread(img_path)
    #이미지 추론하기 ( 해당 함수는 전처리와 후처리를 함꺼번에 해준다)
    result = sr.upsample(img)

    return result

# image=upscale('./test_img/TEST_00001.png')
# plt.imshow(image)
# cv2.imwrite('./test_img_up/TEST_00001.png',image)

f = open('./submit.csv','r')
rdr = csv.reader(f)
getPath = './test_img/'
putPath = './test_img_up/'
num=-1
for line in rdr:
    num = num+1
    if num==0:
        continue
    IMAGE_PATH = getPath + line[0] + '.png'
    RESULT_PATH = putPath + line[0] + '.png'
    
    img = upscale(IMAGE_PATH)
    cv2.imwrite(RESULT_PATH,img)
    print(line[0] + " upscale success")
 
f.close()


# %%
