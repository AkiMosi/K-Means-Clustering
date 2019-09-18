#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:34:10 2019

@author: akimosi
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import math as m
import time
from mpl_toolkits.mplot3d import Axes3D
start = time.time()

im = Image.open("img.jpg")
im = im.resize([int(im.size[0]/8),int(im.size[1]/8)])
im.save("actual.jpg")
im_new = Image.new(im.mode,im.size)
pixel_new = im_new.load()
pixel = im.load()
r,g,b = [],[],[]
for i in range(im.size[0]):
    for j in range(im.size[1]):
        r.append(pixel[i,j][0])
        g.append(pixel[i,j][1])
        b.append(pixel[i,j][2])
r_new = list(r)
g_new = list(g)
b_new = list(b)

def cluster(k,cent):
    count = 0
    while(True):
        dist = []
        clas = []
        cent1 = []
        for i in range(k):
            dist.append(list())
            clas.append(list())
            cent1.append(list())
        for i in range(len(r)):
            index = 0
            mini = 10000000000
            for j in range(k):
                temp = m.sqrt((cent[j][0][0] - r[i])**2 + (cent[j][0][1] - g[i])**2 + (cent[j][0][2] - b[i])**2)
                dist[j].append(temp)
                if(temp < mini):
                    mini = temp
                    index = j
            clas[index].append([r[i],g[i],b[i]])
        for i in range(k):
            clas[i] = np.array(clas[i])
        clas = np.array(clas)
        for i in range(k):
            if(len(clas[i])==0):
                cent1[i].append([0,0,0])
            else:
                cent1[i].append([int(round(clas[i][:,0].sum()/len(clas[i]))),
                                 int(round(clas[i][:,1].sum()/len(clas[i]))),
                                 int(round(clas[i][:,2].sum()/len(clas[i])))])
        count += 1
        if(cent == cent1):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("R")
            ax.set_ylabel("G")
            ax.set_zlabel("B")
            for i in range(k):
                if(len(clas[i])==0):
                    continue
                ax.scatter3D(clas[i][:,0],clas[i][:,1],clas[i][:,2])
            plt.show()
            return cent
        else :
            cent = cent1

f_cent = []
K = 2
while(K<128):
    rndpts = []
    dist = []
    for i in range(K):
        dist.append([])
        temp = rnd.randint(0,len(r)-1)
        rndpts.append([(r[temp],g[temp],b[temp])])
    cent = cluster(K,rndpts)
    f_cent = (cluster(K,cent))
    print("Done for K = ",K)
    print("Centroids : ",f_cent)
    print("Time taken : ",format((time.time()-start),".3f"),"s")
    for i in range(len(r)):
        index = 0
        mini = 10000000000
        for j in range(K):
            temp = m.sqrt((f_cent[j][0][0] - r[i])**2 + (f_cent[j][0][1] - g[i])**2 + (f_cent[j][0][2] - b[i])**2)
            dist[j].append(temp)
            if(temp < mini):
                mini = temp
                index = j
        r_new[i],g_new[i],b_new[i] = f_cent[index][0][0],f_cent[index][0][1],f_cent[index][0][2]
    count = 0
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            pixel_new[i,j] = (r_new[count],g_new[count],b_new[count])
            count+=1
    stri = "k_"+str(K)+".jpg"
    im_new.save(stri)
    K*=2
