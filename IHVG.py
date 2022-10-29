# -*- coding: utf-8 -*-

###Image Visibility Graph

import numpy as np
import pandas as pd
from PIL import Image
import os
import time
from matplotlib import pyplot as plt
path=r'C:\Users\dxz\Desktop'

###给图片添加噪声
def get_NSR(X,NSR):
    m,n=X.shape
    #先求原始图像的能量
    ps=(np.std(X))**2
    #噪声的能量
    pn=ps*pow(10,NSR/10.)-ps
    #生成期望的噪声
    expect_noise = np.random.normal(loc=0, scale=np.sqrt(pn), size=(m, n))
    X=X+expect_noise
    return X

def  display_time(func):
    def wrapper(*args,**kwargs):
        start=time.perf_counter()
        res=func(*args,**kwargs)
        end=time.perf_counter()
        print('程序运行时间:{}'.format(end-start))
        return res
    return wrapper

def imageVisibilityGraph(I,criterion,lattice):
    I=I.astype('float16')
    m,n=I.shape
    if m!=n:
        print('Input image must be square')
        return

    if (criterion!='HVG')+(criterion!='VG')>1:
        print('Criterion string must be <HVG> or <VG>')
        return

    if type(lattice)!=bool:
        print('lattice must be logical: TRUE to save lattice structure, FALSE otherwise')
        return

    ei=[]
    ej=[]

    ###IHVG算法的实现
    if criterion=='HVG':
        for i in range(n):
            print(i)
            for j in range(n):
                ###lattice
                if lattice:
                    #1.右边
                    if j < n-1:
                        ei.append(n * i + j)
                        ej.append(n * i + j + 1)

                    #2.右下角
                    if (i < n-1)  & (j < n-1):
                        ei.append(n * i + j)
                        ej.append(n * (i + 1) + j + 1)

                    #3.下边
                    if i < n-1:
                        ei.append(n * i + j)
                        ej.append(n * (i + 1) + j)

                    #4. 左下角
                    if (i < n-1) & (j > 0):
                        ei.append(n * i + j)
                        ej.append(n * (i + 1) + j - 1)

                #1. 右边
                if j < n - 2:
                    max_value=I[i,j+1]
                    for k in range(j+2,n):
                        max_value=max(max_value,I[i,k-1])
                        if max_value<min(I[i,j],I[i,k]):
                            ei.append(n * i + j)
                            ej.append(n * i + k)
                        if max_value>=I[i,j]:
                            break

                # 2. 右下角
                if (j<n-2)&(i<n-2):
                    diag_length=min(n-i,n-j)
                    max_value=I[i+1,j+1]
                    for k in range(2,diag_length):
                        max_value=max(max_value,I[i+k-1,j+k-1])
                        if max_value<min(I[i,j],I[i+k,j+k]):
                            ei.append(n * i + j)
                            ej.append(n * (i+k) + j+k)
                        if max_value>=I[i,j]:
                            break

                #3. 下边
                if (i<n-2):
                    max_value = I[i+1, j]
                    for k in range(i+2, n):
                        max_value = max(max_value, I[k-1, j])
                        if max_value < min(I[i, j], I[k, j]):
                            ei.append(n * i + j)
                            ej.append(n * k + j)
                        if max_value >= I[i, j]:
                            break

                #4. 左下角
                if (j > 1)  & (i < n - 2):
                    diag_length = min(n-i, j+1)
                    max_value = I[i + 1, j - 1]
                    for k in range(2, diag_length):
                        max_value = max(max_value, I[i + k - 1, j - k + 1])
                        if max_value < min(I[i, j], I[i + k, j - k]):
                            ei.append(n * i + j)
                            ej.append(n * (i + k) + j - k)
                        if max_value >= I[i, j]:
                            break
    ei=np.array(ei).reshape(-1,1)
    ej=np.array(ej).reshape(-1,1)
    e=np.append(ei,ej,axis=1)
    return e

def compute_Q(e_original,e_noise,N):
    Q_min=4*(N-1)*N+4*(N-1)**2
    e_concat=np.append(e_original,e_noise,axis=0)
    e_count_df=pd.DataFrame(e_concat)
    sum_1=np.sum(e_count_df.duplicated())*2
    sum_2=len(e_original)*2
    return (sum_1-Q_min)/(sum_2-Q_min)

@display_time
def main():
    img_path = os.path.join(path, 'Lena.png')
    img = Image.open(img_path)
    img = np.array(img)
    N = len(img)
    e_original = imageVisibilityGraph(I=img, criterion='HVG', lattice=True)
    # image_white_noise=get_NSR(img, 100)
    # e_white_noise=imageVisibilityGraph(I=image_white_noise, criterion='HVG', lattice=True)
    # Q_infinity=compute_Q(e_original, e_white_noise, N)
    # s = []
    # for i in range(10):
    #     print(i)
    #     img_noise = get_NSR(img, i)
    #     e_noise = imageVisibilityGraph(I=img_noise, criterion='HVG', lattice=True)
    #     Q=compute_Q(e_original, e_noise, N)
    #     s.append((Q-Q_infinity)/(1.-Q_infinity))
    # plt.semilogy(range(10),s)
    # plt.show(block=True)

if __name__ == '__main__':
    main()