# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 22:25:09 2018

@author: 10066
"""

import os  
import cv2
import numpy as np
  
'''def file_name(file_dir):   
    for root, dirs, files in os.walk(file_dir):  
        print(root) #当前目录路径  
        print(dirs) #当前路径下所有子目录  
        print(files) #当前路径下所有非目录子文件  
    image_path = 
    return files'''
  
    
def contrast_brightness_image(src1, a, g):  
    h, w, ch = src1.shape#获取shape的数值，height和width、通道  
  
    #新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)  
    src2 = np.zeros([h, w, ch], src1.dtype)  
    dst = cv2.addWeighted(src1, a, src2, 1-a, g)#addWeighted函数说明如下  
    return dst 

image_path = []

for label in ['68']:
        
    file_dir = "C:/Users/10066/Desktop/dachuang/datasets/ourtrain/" + label
    for root, dirs, files in os.walk(file_dir):  
        image_path.append(files)
    
    kernel_size = (5, 5);
    sigma = 2;


    fl=open("C:/Users/10066/Desktop/dachuang/datasets/ourtrain/" + label + "/new_txt.txt","w")
    
    '''for imgName in image_path[0][0:-1]:
        
        use_path = "C:/Users/10066/Desktop/dachuang/datasets/ourtrain/" + label + "/" + imgName
        img = cv2.imread(use_path);
        img = cv2.GaussianBlur(img, kernel_size, sigma);
        new_imgName = "C:/Users/10066/Desktop/dachuang/datasets/ourtrain/" + label + "/new/" + "New_1_" + str(kernel_size[0]) + "_" + str(sigma) + "_" + imgName;
        cv2.imwrite(new_imgName, img);
        new_name = "New_1_" + str(kernel_size[0]) + "_" + str(sigma) + "_" + imgName + ' ' + label + '\n'
        fl.write(new_name)
        
    
    
    for imgName in image_path[0][0:-1]:
        
        use_path = "C:/Users/10066/Desktop/dachuang/datasets/ourtrain/" + label + "/" + imgName
        img = cv2.imread(use_path);
        img = contrast_brightness_image(img, 1.2, 10)
        new_imgName = "C:/Users/10066/Desktop/dachuang/datasets/ourtrain/" + label + "/new/" + "New_2_" + str(kernel_size[0]) + "_" + str(sigma) + "_" + imgName;
        cv2.imwrite(new_imgName, img);
        new_name = "New_2_" + str(kernel_size[0]) + "_" + str(sigma) + "_" + imgName + ' ' + label + '\n'
        fl.write(new_name)'''
        
    
    for imgName in image_path[0][0:-1]:
        
        use_path = "C:/Users/10066/Desktop/dachuang/datasets/ourtrain/" + label + "/" + imgName
        img = cv2.imread(use_path);        
        img = contrast_brightness_image(img, 0.8, 0.8)
        new_imgName = "C:/Users/10066/Desktop/dachuang/datasets/ourtrain/" + label + "/new/" + "New_3_" + str(kernel_size[0]) + "_" + str(sigma) + "_" + imgName;
        cv2.imwrite(new_imgName, img);
        new_name = "New_3_" + str(kernel_size[0]) + "_" + str(sigma) + "_" + imgName + ' ' + label + '\n'
        fl.write(new_name)
    
    fl.close()






