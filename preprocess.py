# -*- coding: utf-8 -*-

import cv2, glob, numpy, os
import pandas as pd

scale=300
image_size = 540
source_dir='/media/carlos_bologna/WD_NTFS/Diabetic_Retinopathy/original_images/'
dest_dir='data/'

def scaleRadius(img,scale):
    x=img[img.shape[0]/2,:,:].sum(1)
    r=(x>x.mean()/10).sum() / 2
    s=scale * 1.0 / r
    return cv2.resize(img,(0,0),fx=s,fy=s)

def calcRadius(img):
    x=img[img.shape[0]/2,:,:].sum(1)
    r=(x>x.mean()/10).sum() / 2
    return r

df_train = pd.read_csv('data/trainLabels.csv')
df_test = pd.read_csv('data/testLabels.csv') # Já temos os targets da base de teste, então vamos usar.

df_train['folder'] = 'train'
df_test['folder'] = 'test'
df = pd.concat([df_train.loc[:, ['folder', 'image', 'level']], df_test.loc[:, ['folder', 'image', 'level']]])

tot_lote = len(df) / 5000

for index, row in df.iterrows():
    try:
        a=cv2.imread(os.path.join(source_dir, row.folder, row.image + '.jpeg'))
        #scale img to a given radius
        a=scaleRadius(a,scale)
        #subtract local mean color
        a=cv2.addWeighted(a, 4, cv2.GaussianBlur(a,(0,0),scale/30), -4, 128)
        #remove outer 10%
        b=numpy.zeros(a.shape)
        cv2.circle(b,(a.shape[1]/2,a.shape[0]/2), int(scale * 0.9),(1,1,1), -1,8,0)
        #a= a * b + 128 * (1 - b)
        a = a * b
        # Crop image removing black border
        #r = calcRadius(a)
        r = image_size / 2
        x_half = a.shape[0] / 2
        y_half = a.shape[1] / 2
        a = a[x_half - r:x_half + r, y_half - r:y_half + r, :]
        cv2.imwrite(os.path.join(dest_dir, str(row.level), row.folder + '_' + row.image + '.jpeg'), a)
        if (index % 5000 == 0):
            print 'Lote: ' + str(index) + '/' + str(tot_lote)
    except:
        print 'Error in ' + row.image
