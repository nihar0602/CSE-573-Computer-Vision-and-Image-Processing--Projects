#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:01:23 2020

@author: nihar
"""
import numpy as np
import cv2 as cv
import glob
import time
# import imp
import sys
from scipy.spatial import distance
import json
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank =comm.Get_rank()
    cores = comm.Get_size()
    parallel = True
except:
    print("\nMPI installation not detected. Process will run as serial\n")
    parallel =False





def build_feature(img_width,img_height):
    '''


    Parameters
    ----------
    img_width : INT
        WIDTH OF THE IMAGE.
    img_height : INT
        HEIGHT OF THE IMAGE.

    Returns
    -------
    features : ARRAY
        GIVES ARRAY OF TUPLES FOR ALL TYPES OF HAAR FEATURES WHICH ARE CREATED.

    '''

    features = []

    '''type 1
       +++---        A   B   C
       +++---        D   E   F
       feature = (ABED) - (BCEF) = -F+2E-D+A-2B+C = 2E+A+C-F-D-2B
    '''
    feature1 = []
    for x in range(0,img_width):
        for y in range(0,img_height):
            for w in range(1,img_width-x,2):
                for h in range(1,img_height-y):
                    A=(x,y)
                    B=(x+(w//2),y)
                    C=(x+w,y)
                    D=(x,y+h)
                    E=(x+(w//2),y+h)
                    F=(x+w,y+h)
                    add = [E,E,A,C]
                    sub = [F,D,B,B]
                    add = np.asarray(add)
                    sub = np.asarray(sub)
                    sum_ = ((add),(sub))
                    feature1.append(sum_)

    features.extend(feature1)
    print("Feature 1 created:",len(feature1))

    '''type 2
        ------      A    B
        ------      D    C
        ++++++
        ++++++      E    F
        feature = (DCFE)-(ABCD) = F-2C+B+2D-A-E= F+B+2D-2C-A-E
    '''
    feature2 = []
    for x in range(0,img_width):
        for y in range(0,img_height):
            for w in range(1,img_width-x):
                for h in range(1,img_height-y,2):
                    A=(x,y)
                    B=(x+w,y)
                    C=(x+w,y+(h//2))
                    D=(x,y+(h//2))
                    E=(x,y+h)
                    F=(x+w,y+h)
                    add = [F,B,D,D]
                    sub = [C,C,A,E]
                    add = np.asarray(add)
                    sub = np.asarray(sub)
                    sum_ = ((add),(sub))
                    feature2.append(sum_)
    features.extend(feature2)
    print("Feature 2 created:",len(feature2))

    '''type 3
        --++--        A  B  E  F
        --++--        D  C  H  G
        feature = (HEBC)-[(ABCD)+(EFGH)] = -G+2H-2C+D-A+2B-2E+F
    '''
    feature3 = []
    for x in range(0,img_width):
        for y in range(0,img_height):
            for w in range(1,img_width-x,3):
                for h in range(1,img_height-y):
                    A=(x,y)
                    B=(x+w//2,y)
                    C=(x+w//3,y+h)
                    D=(x,y+h)
                    E=(x+2*(w//3),y)
                    F=(x+w,y)
                    H=(x+2*(w//3),y+h)
                    G=(x+w,y+h)
                    add = [H,H,D,B,B,F]
                    sub = [G,C,C,A,E,E]
                    add = np.asarray(add)
                    sub = np.asarray(sub)
                    sum_ = ((add),(sub))
                    feature3.append(sum_)
    features.extend(feature3)
    print("Feature 3 created:",len(feature3))

    '''type 4
        ++++++      A   B
        ++++++
        ------      D   C
        ------      E   F
        ++++++
        ++++++      H   G
        feature = [(ABCD)+(EFGH)]-(DCFE) = G-2F+2C-B+A-2D+2E-H
    '''
    feature4 = []
    for x in range(0,img_width):
        for y in range(0,img_height):
            for w in range(1,img_width-x):
                for h in range(1,img_height-y,3):
                    A=(x,y)
                    B=(x+w,y)
                    C=(x+w,y+(h//3))
                    D=(x,y+(h//3))
                    E=(x,y+2*(h//3))
                    F=(x+w,y+2*(h//3))
                    G=(x+w,y+h)
                    H=(x,y+h)
                    add = [G,C,C,A,E,E]
                    sub = [F,F,B,D,D,H]
                    add = np.asarray(add)
                    sub = np.asarray(sub)
                    sum_ = ((add),(sub))
                    feature4.append(sum_)
    features.extend(feature4)
    print("Feature 4 created:",len(feature4))

    '''type 5
        +++---       IA  HB  GC
        +++---
        ---+++       FD  EE  DF
        ---+++       CG  BH  AI
        feature = (ABED)+(EFIH)-(DEHG)-(BCFE) = I-2H+G-2F+4E-2D+C-2B+A
    '''
    feature5 = []
    for x in range(0,img_width):
        for y in range(0,img_height):
            for w in range(1,img_width-x,2):
                for h in range(1,img_height-y,2):
                    A=(x,y)
                    B=(x+(w//2),y)
                    C=(x+w,y)
                    D=(x,y+(h//2))
                    E=(x+w//2,y+(h//2))
                    F=(x+w,y+(h//2))
                    G=(x,y+h)
                    H=(x+(w//2),y+h)
                    I=(x+w,y+h)
                    add = [I,G,E,E,E,E,C,A]
                    sub = [H,H,F,F,D,D,B,B]
                    add = np.asarray(add)
                    sub = np.asarray(sub)
                    sum_ = ((add),(sub))
                    feature5.append(sum_)
    features.extend(feature5)
    print("Feature 5 created:",len(feature5))
    print('----------------------------------------')
    print(len(features))
    x = np.array(features)
    np.save("features",x)
    return features


def feature_extraction(img,feature_locs):
    '''


    Parameters
    ----------
    img : INTEGRAL IMAGE
        TAKES THE INTEGRAL IMAGE
    feature_locs : ARRAY(90640x2)
        CONTAINS AN ARRAY OF CREATED FEATURES

    Returns
    -------
    FEATURE VALUES
        THIS FUCTION EXTRACTS THE FEATURE VALUE ON THE APPLIED INTERGRAL IMAGE

    '''
    features = np.ones([feature_locs.shape[0],1],dtype=np.float64)
    for i in range(0,feature_locs.shape[0]):
        add = feature_locs[i,0]
        sub = feature_locs[i,1]
        features[i] = np.sum(img[add[:,0],add[:,1]]) - np.sum(img[[sub[:,0]],[sub[:,1]]])

    return np.reshape(features,(len(features),))

def integral_image(img):
    '''


    Parameters
    ----------
    img : NORMAL IMAGE
        SIMPLE IMAGE OR NORMALIZED IMAGE

    Returns
    -------
    sum_img : INTEGRAL IMAGE
        GIVES THE OUTPUT OF THE INTEGRAL IMAGE

    '''
    sum_img = np.zeros(img.shape)
    ii_img = np.zeros(img.shape)
    ii_img[:,0] = img[:,0]
    for x in range(1,img.shape[1]):
        ii_img[:,x] = ii_img[:,x-1] + img[:,x]
    sum_img[0,:] = ii_img[0,:]
    for y in range(1,img.shape[0]):
        sum_img[y,:] = sum_img[y-1,:] + ii_img[y,:]
    return sum_img

def train(fvals,index,max_iter):
    '''


    Parameters
    ----------
    fvals : ARRAY
        THE FEATURE VALUE MATRIX (90640x14000)
    index : INT
        PARTITION OF FACES AND NON-FACE IN THE FEATURE VALUE MATRIX
    max_iter : INT
        TOTAL NO. OF BEST WEAK CLASSIFIERS

    Returns
    -------
    strongclass :  LIST
        GIVES THE INDEX OF FEATURES WHICH ARE BEST
    alpha : LIST
        ALPHA VALUES OF THE CORRESPONDING FEATURES WHICH ARE BEST

    '''
    truth = np.hstack((np.ones([1,index]),np.zeros([1,fvals.shape[1]-index])))
    # print('LLLLAA',fvals.shape[1])
    # print(truth.shape)
    weights = np.zeros(truth.shape)
    weights[:,:index] = 0.5/(index)
    weights[:,index:]= 0.5/(truth.shape[1]-index)
    m1=np.mean(fvals[:,:index],axis=1)
    # print(m1,'m1\n')
    m2=np.mean(fvals[:,index:],axis=1)
    # print(m2,'m2\n')
    threshold =0.5*(m1+m2)
    threshold = np.reshape(threshold,(len(threshold),1))
    threshold[m2<m1] = -1*threshold[m2<m1]
    fvals[m2<m1,:] = -1*fvals[m2<m1,:]
    predictions = np.zeros(fvals.shape)
    predictions[fvals<threshold] = 1
    # print(predictions==0)
    strongclass = []
    alpha = []
    iter = 0
    while(iter< max_iter):
        # print(weights,'before normalized\n')
        weights = np.divide(weights,np.sum(weights))
        acterror = np.sum(weights*np.absolute(truth-predictions),axis=1)
        # print(acterror)
        print(np.min(acterror),'acterror\n')
        minind = np.argmin(acterror)
        et = acterror[minind]
        print(minind,'minind\t',iter,'iter\n')
        exponents = 1-np.absolute(np.subtract(truth,predictions[minind,:]))
        beta = et/(1-et)
        print(beta,'beta\n')
        print('min error', et)
        weights = weights*np.power(beta,exponents)
        strongclass.append(minind)
        alpha.append(np.log10(1/beta))
        iter+=1
        if iter%1000==0:
            np.save("strongfeats"+str(iter),strongclass)
            np.save("alpha"+str(iter),alpha)
    return strongclass, alpha

def faces(img,features,alpha,thresholds,polarity):
    '''


    Parameters
    ----------
    img : SUB-WINDOW OF INTEGRAL IMAGE
        LOAD AN SUB-WINDOW PATCH OF INTEGRAL IMAGE.
    features : ARRAY
        BUILD FEATURES.
    alpha : INT
        ALPHA VALUES FOR THE CORRESPONDING FEATURES
    thresholds : ARRAY (nx1)
        THRESHOLD VALUE OF ALL THE FEATURES
    polarity : ARRAY (nx1)
        POLARITY OF ALL THE FEATURES

    Returns
    -------
    check : INT
        RETURNS 1 OR 0. 1: IF FACE IS DETECTED, 0: IF NO FACE IS DETECTED

    '''
    fvals = feature_extraction(img,features)
    fvals = np.reshape(fvals,(len(fvals),1))
    fvals = np.multiply(fvals,polarity)
    weakclass = (fvals<thresholds)
    weakclass = np.reshape(weakclass,alpha.shape)
    test = np.multiply(weakclass,alpha)
    check = np.sum(test)>=0.5*np.sum(alpha)
    # print(np.sum(test),np.sum(alpha))
    return check


#------------------------------------------------------------------------------------------------------------

def main():



    if len(sys.argv)>2 and sys.argv[2] == 1:
#------------------------------- BUILD FEATURES -------------------------------------------------

        build_feature(21,21)

#--------------------------------EXTRACT FEATURE VALUES -----------------------------------------

        if rank ==0:
            feature_locs = np.load('features.npy',allow_pickle='False')
            imgs = glob.glob('Model_Files/faces/*.ppm')
            imgs2 = glob.glob('Model_Files/non-faces/*.jpg')
            print(len(imgs2),len(imgs2))
            comm.send(imgs2,dest=1,tag =98)
            comm.send(imgs2,dest=2,tag=95)
            comm.send(imgs2,dest=3,tag=96)
            feat1 = np.ones([len(feature_locs),round(0.25*len(imgs))],dtype = np.float64)
            intimg =[]
            for i in range(0,round(0.25*len(imgs))):
                print(i,'i\t','core',rank,'\n')
                image = imgs[i]
                img = cv.imread(image,0)
                img = cv.resize(img,(21,21))
                ii_img = integral_image(img)
                intimg.append([ii_img])
                feat1[:,i] = feature_extraction(ii_img,feature_locs)
            feat2 = comm.recv(source=1,tag = 56)
            feat3 = comm.recv(source=2,tag = 57)
            feat4 = comm.recv(source=3,tag =58)
            int2 = comm.recv(source=1,tag = 5)
            int3 = comm.recv(source=2,tag = 6)
            int4 = comm.recv(source=3,tag =7)
            final_feat = np.hstack((feat1,feat2,feat3,feat4))
            intimg.append(int2)
            intimg.append(int3)
            intimg.append(int4)
            # print(len(feat1)+len(feat2),'faces\n')
            # print(len(feat3)+len(feat4),"nonfaces\n")
            print(len(intimg),'lenintimg')
            print(len(final_feat))
            print(final_feat[:10,:])
            np.save("integral_image",intimg)
            np.save("calc_features14k",final_feat)


        if rank ==1:
            feature_locs = np.load('features.npy',allow_pickle='False')
            imgs = comm.recv(source=0,tag=98)
            feat2 = np.ones([len(feature_locs),round(0.25*len(imgs))],dtype = np.float64)
            intimg =[]
            for i in range(0,round(0.25*len(imgs))):
                print(i,'i\t','core',rank,'\n')
                image = imgs[i+round(0.25*len(imgs))]
                img = cv.imread(image,0)
                img = cv.resize(img,(21,21))
                ii_img = integral_image(img)
                intimg.append([ii_img])
                feat2[:,i] = feature_extraction(ii_img,feature_locs)
            comm.send(feat2,dest=0,tag =56)
            comm.send(intimg,dest=0,tag=5)

        if rank ==2:
            feature_locs = np.load('features.npy',allow_pickle='False')
            imgs = comm.recv(source=0,tag=95)
            feat3 = np.ones([len(feature_locs),round(0.25*len(imgs))],dtype = np.float64)
            intimg =[]
            for i in range(0,round(0.25*len(imgs))):
                print(i,'i\t','core',rank,'\n')
                image = imgs[i + round(0.5*len(imgs))]
                img = cv.imread(image,0)
                img = cv.resize(img,(21,21))
                ii_img = integral_image(img)
                intimg.append([ii_img])
                feat3[:,i] = feature_extraction(ii_img,feature_locs)
            comm.send(feat3,dest=0,tag =57)
            comm.send(intimg,dest=0,tag=6)

        if rank ==3:
            feature_locs = np.load('features.npy',allow_pickle='False')
            imgs = comm.recv(source=0,tag=96)
            feat4 = np.ones([len(feature_locs),round(0.25*len(imgs))],dtype=np.float64)
            intimg =[]
            for i in range(0,round(0.25*len(imgs))):
                print(i,'i\t','core',rank,'\n')
                image = imgs[i+round(0.75*len(imgs))]
                img = cv.imread(image,0)
                img = cv.resize(img,(21,21))
                ii_img = integral_image(img)
                intimg.append([ii_img])
                feat4[:,i] = feature_extraction(ii_img,feature_locs)
            comm.send(feat4,dest=0,tag =58)
            comm.send(intimg,dest=0,tag=7)

#---------------------------------------ADABOOST-------------------------------------------------------

        features = np.load('features.npy',allow_pickle='False')
        fvals = np.load('calc_features14k.npy',allow_pickle='False')
        print(features.shape)
        index = 7000
        max_iter = 10000
        indices,alpha = train(fvals,index,max_iter)
        print('Indices:',indices,'Alpha:',alpha)
        final_cls = features[indices,:]
        np.save("finalclassifier",final_cls)
        np.save('alpha',alpha)

#----------------------------------------------------------------------------------------------------------

    else:

        t1 = time.time()
        test_imgs = glob.glob(sys.argv[1]+'/*.jpg')
        print(sys.argv)
        # print(test_imgs)

        test_imgs.sort()

        featureinds = np.load('Model_Files/strongfeats6000.npy',allow_pickle='False')
        features = np.load('Model_Files/features.npy',allow_pickle="False")
        features = features[featureinds,:]

        thresholds = np.load('Model_Files/thresholds.npy',allow_pickle='False')
        # print(thresholds.shape)
        polarity = np.load('Model_Files/polarity.npy',allow_pickle="False")
        # print(polarity.shape)

        alpha = np.load("Model_Files/alpha6000.npy",allow_pickle='False')
        # error =0
        thresholds = thresholds[featureinds]
        polarity = polarity[featureinds,]

        result=[]
        for xx in range(len(test_imgs)):
            points  = [(0,0)]
            xlen=[]
            ylen=[]
            count =[]
            counter = 0
            io = test_imgs[xx].split("/")
            # print(io)
            iname = io[1]
            print(iname)
            img1 = cv.imread(test_imgs[xx],0)
            # print(img.shape)
            sp = [4]
            for scale_percent in sp:
                #scale_percent = 5 # percent of original size
                sf = 100/scale_percent
                width = int(img1.shape[1]/sf)
                height = int(img1.shape[0]/sf)
                dim = (height,width)
                img = cv.resize(img1,dim,interpolation = cv.INTER_AREA)
                ii_img = integral_image(img)
                # print(img.shape)
                for ii in range(0,img.shape[0]-21 , int(0.1*height)):
                    for jj in range(0,img.shape[1]-21, int(0.1*width)):
                        sub_win = ii_img[ii:ii+21,jj:jj+21]
                        check1 = faces(sub_win,features[:2000],alpha[:2000],thresholds[:2000],polarity[:2000])
                        if check1 ==True:
                            check = faces(sub_win,features,alpha,thresholds,polarity)
                            if check == True:
                                # print("index",ii,jj)
                                if (len(xlen)>0 and np.sqrt(((ii*sf)-points[-1][0])**2+((jj*sf)-points[-1][1])**2)<2*21*sf):
                                    #print("DD")
                                    count[-1]+=1
                                    points[-1]= (min(round(ii*sf),points[-1][0]),min(round(jj*sf),points[-1][1]))
                                    xlen[-1]=min(round(abs(ii*sf-points[-1][0])*0.5+xlen[-1]),img1.shape[0]-points[-1][0])
                                    ylen[-1]=min(round(abs(jj*sf-points[-1][1])*0.5+ylen[-1]),img1.shape[1]-points[-1][1])
                                else:
                                    points.append((round(ii*sf),round(jj*sf)))
                                    #print("appended")
                                    count.append(0)
                                    xlen.append(round(21*sf))
                                    ylen.append(round(21*sf))
            points.pop(0)
            xlen = np.asarray(xlen)
            ylen = np.asarray(ylen)

            count = np.asarray(count)
            points = np.asarray(points)

            if len(xlen>0):
                dists = distance.cdist(points,points,'euclidean')
                tmp = dists< 0.05*img1.shape[1]
                set = np.arange(0,tmp.shape[0])
                points2 =[]
                xlen2=[]
                ylen2=[]
                count2=[]
                area = np.multiply(xlen,ylen)
                tier = count/area
                while len(set)>0:
                    vec = tmp[set[0],:]
                    newind = np.argmin(count[vec])
                    points2.append(points[newind,:])
                    xlen2.append(xlen[newind])
                    ylen2.append(ylen[newind])
                    count2.append(count[newind])
                    jk = np.nonzero(vec==True)
                    set = np.setdiff1d(set,jk)
                xlen = np.asarray(xlen2)
                ylen =np.asarray(ylen2)
                points = np.asarray(points2)
                count = np.asarray(count2)
                xcents = points[:,0]+xlen*0.5
                ycents = points[:,1]+ylen*0.5
                xlen = np.round(xlen*0.7)
                ylen = np.round(ylen*0.7)
                xlen = xlen.astype(int)
                ylen = ylen.astype(int)
                points[:,0] = np.round(xcents-xlen*0.5)
                points[:,1] = np.round(ycents-ylen*0.5)

            for i in range(len(xlen)):
                # print(points[i][0])
                # print(xlen,ylen,"xylen")
                jk_img = img1[points[i][0]:points[i][0]+xlen[i],points[i][1]:points[i][1]+ylen[i]]
                bbox = [int(points[i][0]),int(points[i][1]),int(xlen[i]),int(ylen[i])]
                # print('bbox',bbox)
                result.append({"iname": iname, "bbox": bbox})
                counter = counter + 1
        output_json = "results.json"
        #dump json_list to result.json
        t2= time.time()
        print('Total time : {}'.format(t2-t1))
        with open(output_json, 'w') as f:
            json.dump(result, f)


if __name__== "__main__":
    main()
