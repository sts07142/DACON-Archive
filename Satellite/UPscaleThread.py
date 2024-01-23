# %%
from threading import Thread
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

def saveUP(join, end):
    num=-2
    f = open('./submit.csv','r')
    rdr = csv.reader(f)

    for line in rdr:
        num = num+1
        if num==-1:
            continue
        elif num<join:
            continue
        elif num==end:
            break

        IMAGE_PATH = getPath + line[0] + '.png'
        RESULT_PATH = putPath + line[0] + '.png'
        
        img = upscale(IMAGE_PATH)
        cv2.imwrite(RESULT_PATH,img)
        print(line[0] + " upscale success")
    f.close()

if __name__ == "__main__":
    getPath = './test_img/'
    putPath = './test_img_up/'
    START, END = 0, 60639
    s1=0
    e1=1515
    r=1516
    th1 = Thread(target=saveUP, args=(s1,e1))
    th2 = Thread(target=saveUP, args=(s1+r,e1+r))
    th3 = Thread(target=saveUP, args=(s1+2*r,e1+2*r))
    th4 = Thread(target=saveUP, args=(s1+3*r,e1+3*r))
    th5 = Thread(target=saveUP, args=(s1+4*r,e1+4*r))
    th6 = Thread(target=saveUP, args=(s1+5*r,e1+5*r))
    th7 = Thread(target=saveUP, args=(s1+6*r,e1+6*r))
    th8 = Thread(target=saveUP, args=(s1+7*r,e1+7*r))
    th9 = Thread(target=saveUP, args=(s1+8*r,e1+8*r))
    th10 = Thread(target=saveUP, args=(s1+9*r,e1+9*r))
    th11 = Thread(target=saveUP, args=(s1+10*r,e1+10*r))
    th12 = Thread(target=saveUP, args=(s1+11*r,e1+11*r))
    th13 = Thread(target=saveUP, args=(s1+12*r,e1+12*r))
    th14 = Thread(target=saveUP, args=(s1+13*r,e1+13*r))
    th15 = Thread(target=saveUP, args=(s1+14*r,e1+14*r))
    th16 = Thread(target=saveUP, args=(s1+15*r,e1+15*r))
    th17 = Thread(target=saveUP, args=(s1+16*r,e1+16*r))
    th18 = Thread(target=saveUP, args=(s1+17*r,e1+17*r))
    th19 = Thread(target=saveUP, args=(s1+18*r,e1+18*r))
    th20 = Thread(target=saveUP, args=(s1+29*r,e1+19*r))
    th21 = Thread(target=saveUP, args=(s1+20*r,e1+20*r))
    th22 = Thread(target=saveUP, args=(s1+21*r,e1+21*r))
    th23 = Thread(target=saveUP, args=(s1+22*r,e1+22*r))
    th24 = Thread(target=saveUP, args=(s1+23*r,e1+23*r))
    th25 = Thread(target=saveUP, args=(s1+24*r,e1+24*r))
    th26 = Thread(target=saveUP, args=(s1+25*r,e1+25*r))
    th27 = Thread(target=saveUP, args=(s1+26*r,e1+26*r))
    th28 = Thread(target=saveUP, args=(s1+27*r,e1+27*r))
    th29 = Thread(target=saveUP, args=(s1+28*r,e1+28*r))
    th30 = Thread(target=saveUP, args=(s1+29*r,e1+29*r))
    th31 = Thread(target=saveUP, args=(s1+30*r,e1+30*r))
    th32 = Thread(target=saveUP, args=(s1+31*r,e1+31*r))
    th33 = Thread(target=saveUP, args=(s1+32*r,e1+32*r))
    th34 = Thread(target=saveUP, args=(s1+33*r,e1+33*r))
    th35 = Thread(target=saveUP, args=(s1+34*r,e1+34*r))
    th36 = Thread(target=saveUP, args=(s1+35*r,e1+35*r))
    th37 = Thread(target=saveUP, args=(s1+36*r,e1+36*r))
    th38 = Thread(target=saveUP, args=(s1+37*r,e1+37*r))
    th39 = Thread(target=saveUP, args=(s1+38*r,e1+38*r))
    th40 = Thread(target=saveUP, args=(s1+39*r,e1+39*r))
    
    th1.start()
    th2.start()
    th3.start()
    th4.start()
    th5.start()
    th6.start()
    th7.start()
    th8.start()
    th9.start()
    th10.start()
    th11.start()
    th12.start()
    th13.start()
    th14.start()
    th15.start()
    th16.start()
    th17.start()
    th18.start()
    th19.start()
    th20.start()
    th21.start()
    th22.start()
    th23.start()
    th24.start()
    th25.start()
    th27.start()
    th28.start()
    th29.start()
    th30.start()
    th31.start()
    th32.start()
    th33.start()
    th34.start()
    th35.start()
    th36.start()
    th37.start()
    th38.start()
    th39.start()
    th40.start()

    th1.join()
    th2.join()
    th3.join()
    th4.join()
    th5.join()
    th6.join()
    th7.join()
    th8.join()
    th9.join()
    th10.join()
    th11.join()
    th12.join()
    th13.join()
    th14.join()
    th15.join()
    th16.join()
    th17.join()
    th18.join()
    th19.join()
    th20.join()
    th21.join()
    th22.join()
    th23.join()
    th24.join()
    th25.join()
    th27.join()
    th28.join()
    th29.join()
    th30.join()
    th31.join()
    th32.join()
    th33.join()
    th34.join()
    th35.join()
    th36.join()
    th37.join()
    th38.join()
    th39.join()
    th40.join()




# %%
