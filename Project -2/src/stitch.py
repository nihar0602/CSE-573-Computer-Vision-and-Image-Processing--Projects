# -*- coding: utf-8 -*-
"""
CVIP: PA-2 IMAGE STITCHING (PANORAMA)

@author: PATEL NIHAR DILIPBHAI
UBIT: 50318506
"""

#DATA - SAMPLE PIAZZA IMAGES
#ub DATA - BONUS TRIAL ( 5 Images)
#extra 1 - Piazza Forum (3 Hotel Images)
#extra 2 - Piazza Forum (4 Hotel Images)
#extra 3 - ITALY IMAGES (3 Images)
#extra 4 - Mountain Sample (2 Images)
#extra 5 - CMU Campus (5  Images)

import cv2
import sys
import numpy as np 
import os
import glob
import scipy.spatial.distance as sp 
import matplotlib.pyplot as plt 
import copy
import random as rand
from PIL import Image

def match_keypoints(kp1,kp2,des1,des2,src_img,des_img):
    
    """
    MATCH THE KEYPOINTS OF LEFT IMAGE AND RIGHT IMAGE
    INPUT: KEYPOINTS OF LEFT AND RIGHT IMAGE, DESCRIPTORS OF LEFT AND RIGHT IMAGE, MATCH KEYOINTS ON! 

    OUTPUT : SHOWS THE MATCH KEYPOINTS IN IMAGE 1 (LEFT ) AND IMAGE 2 (RIGHT). 

    """


    
# --------------------  SMALLEST DISTANCE MEASURED USING SSD -----------------------

    des_length1 = des1.shape[0]
    des_length2 = des2.shape[0]
    print("DES1",des_length1)
    print("DES2",des_length2)

    match_points = []
    for ix,iv in enumerate(des1):
        dist = np.linalg.norm(des1[ix] - des2[0])
        best_dist = dist
        sec_best = dist
        dist_best_idx = [ix,0]
        # print(dist_best_idx)
        for jx,jv in enumerate(des2):
            dist = np.linalg.norm(des1[ix]- des2[jx])
            if dist < best_dist:
                sec_best = best_dist
                best_dist = dist
                dist_best_idx = [ix, jx]
            elif dist < sec_best:
                sec_best = dist
        if (best_dist/sec_best) < 0.8:
            match_points.append(dist_best_idx)
    
    # print(match_points)
    match_points = np.array(match_points)

    match_points_img1 = match_points[:,0]
    match_points_img2 = match_points[:,1]
    # print(match_points_img1)
    # print(match_points_img2)

    keypoint_coord_img_1 = []
    keypoint_coord_img_2 = []

    
    for ii_idx,ii_val in enumerate(match_points_img1):
        keypoint_coord_img_1.append(kp1[ii_val].pt)

    for jj_idx,jj_val in enumerate(match_points_img2):
        keypoint_coord_img_2.append(kp2[jj_val].pt)


# ----------------------- MERGING ALL THE MATCHED KEYPOINTS -----------------------------------------
    keypoints_match = np.concatenate((np.array(keypoint_coord_img_1),np.array(keypoint_coord_img_2)),axis= 1 )
    # print(keypoints_match)


    keypoint_coord_img_1_x = []
    keypoint_coord_img_1_y = []
    
    keypoint_coord_img_2_x = []
    keypoint_coord_img_2_y = []
    
    for xx,yy in keypoint_coord_img_1:
        keypoint_coord_img_1_x.append(xx)
        keypoint_coord_img_1_y.append(yy)

    for xx1,yy1 in keypoint_coord_img_2:
        keypoint_coord_img_2_x.append(xx1)
        keypoint_coord_img_2_y.append(yy1)

    keypoint_coord_img_1_x  = np.asarray(keypoint_coord_img_1_x)
    keypoint_coord_img_1_y  = np.asarray(keypoint_coord_img_1_y)

    keypoint_coord_img_2_x  = np.asarray(keypoint_coord_img_2_x)
    keypoint_coord_img_2_y  = np.asarray(keypoint_coord_img_2_y)    

    
    # print(type(keypoint_coord_img_1_x))


    # plt.figure(figsize = (16,14))
    # plt.subplot(2,1,1)
    # plt.imshow(src_img)
    # plt.scatter(keypoint_coord_img_1_x,keypoint_coord_img_1_y)
    # plt.subplot(2,1,2)
    # plt.imshow(des_img)
    # plt.scatter(keypoint_coord_img_2_x,keypoint_coord_img_2_y)
    # plt.show()
    
    print("Press CTRL+W TO CLOSE THE IF THE IMAGE WINDOW SHOWING KEYPOINTS DISPLAY!! ")




    return keypoints_match
    

def RANSAC(match_keypoints,iterations):
    """
    GETS THE HOMOGRAPHY MATRIX IN ORDER TO REMOVE THE OUTLIERS! 
    INPUT: MATCHKEYPOINTS (N x 4) MATRIX, ITERATIONS
    OUTPUT: HOMOGRAPHY MATRIX (3X3)
    """

    no_of_keypoints = match_keypoints.shape[0]
    ones_ = np.ones((no_of_keypoints,1))
    a_ = match_keypoints
    best_H = []
    highest_inlier_count = 0
    corresponding_points = np.zeros((no_of_keypoints,2))
    
    
    point_1_ = np.append(match_keypoints[:,0:2],ones_,axis = 1)
    point_2_ = match_keypoints[:,2:4]

    for iteration in range(iterations):

        a = rand.choice(match_keypoints)
        b = rand.choice(match_keypoints)
        c = rand.choice(match_keypoints)
        d = rand.choice(match_keypoints)

        random_matchpairs = np.concatenate(([a],[b],[c],[d]),axis= 0)
        # print('HI',random_matchpairs)

        point_1 = np.float32(random_matchpairs[:,0:2])
        point_2 = np.float32(random_matchpairs[:,2:4])

        H = cv2.getPerspectiveTransform(point_1,point_2)

        current_inlier_count = 0

        # --- Solve for Ah = b ------- #
        xx_index = list()
        good_match = list()

        for i in range(no_of_keypoints):
            check_ = np.matmul(H,point_1_[i])
            corresponding_points[i] = (check_/check_[-1])[0:2]
            check = np.square(np.linalg.norm(corresponding_points[i] - point_2_[i]))
            if check < 0.6:
                current_inlier_count = current_inlier_count + 1
                xx_index.append(i)
                good_match.append([ a_[i][0], a_[i][1], a_[i][2], a_[i][3]])


        if current_inlier_count > highest_inlier_count:
            highest_inlier_count = current_inlier_count
            best_match = good_match
            print('Best matching.....',len(best_match))

            best_H = H
            
    
    print("Best matchin ..... DONE")
    print("RANSAC DONE!")
    return best_H

def main():
    all_images = []
    all_images.extend(glob.glob('sys.argv[1]' + '*.jpg'))
    all_images.sort()
    print(all_images)

    inp_img = []
    folder = sys.argv[1]
    for filename in os.listdir(folder):
        if filename == "panorama.jpg":
            print('Panorama Image already found, ignoring that image for stitching')
            continue
        img = cv2.imread(os.path.join(folder, filename))
        print(filename)
        dimension = (500,400)
        n = cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)
        inp_img.append(n)

    # inp_img.sort()
    # print(inp_img)
    

    print(len(inp_img))

    sift = cv2.xfeatures2d.SIFT_create()


    # ---------------- DETECT AND DRAW THE KEYPOINTS USING SIFT --------------------
    img_nos = list()
    for i in range(len(inp_img)):
        img = kps, des = sift.detectAndCompute(inp_img[i],None)
        img_nos.append(img)
    
    img_match_coor = []

    print("IMAGE ORDERING......")
    #------------ CHECK FOR THE CORRESPONDING KEYPOINTS MATCH WITH RESPECT TO ONE IMAGE --------------------------
    for k in range(len(inp_img) - 1):
        for l in range(k+1, len(inp_img)):
            a01 = match_keypoints(img_nos[k][0],img_nos[l][0],img_nos[k][1],img_nos[l][1],inp_img[k],inp_img[l])
            if a01.shape == (1,4):
                print("N0 Points Match")
                continue
            print("KP1", len(img_nos[k][0]))
            print(a01.shape)
            print((k,l),a01.shape[0]/len(img_nos[k][0]))
            x = a01.shape[0]/len(img_nos[k][0])
            y = a01.shape[0]/len(img_nos[l][0])
            if (x > 0.04 and x < 0.9) and (y > 0.04 and y < 0.9):
                print("Selected")
                img_match_coor.append((k,l))
    print(img_match_coor)    

    print('Matching Image Pairs Achieved.')

    #------------------ ONCE WE GOT UNIQUE POINTS, CHECK FOR ENDPOINTS FROM THAT --------------------------------

    all_index = list()
    for a in range(len(img_match_coor)):
        for b in range(2):
            # print((a,b), "CHECK", img_match_coor[a][b])  #TO CHECK THE INDEX OF EACH ELEMENT OF THE TUPLE
            all_index.append(img_match_coor[a][b])

    # print(all_index) 
    all_index = set(all_index)
    print(all_index) #GIVES YOU UNIQUE ELEMENT OF THE TUPLE
    
    #------------------- CHECK FOR RESPECTIVE END POINTS POSSIBILITY--------------------------
    print("Checking for endpoints ......")
    end_points = list()
    for c in all_index:
        count = 0
        for x in range(len(img_match_coor)):
            if c in img_match_coor[x]:
                print('Current Point:',c)
                count = count + 1 
                print('Count',count)
                if count > 1:
                    print("More than required, remove: ", c)
                    end_points.remove(c)
                    continue
                else: 
                    print('check for end points.....',c)
                    end_points.append(c)
                
    print(end_points)
    print("Endpoints Acquired")

    final_order_list = []
    print("Arranging Image pairs in acatual(panoramic) order .................")
    for x in end_points:
        final_order_list.append(x)
        # if len(img_match_coor) == 1:
        #     print("NIKAL LAVDE")
        #     break

        for y in range(len(img_match_coor)):
            if len(img_match_coor) == 1:
                break
            for z in range(len(img_match_coor[y])):
                # print((y,z))
                # if len(img_match_coor) == 1:
                #     print("CONTROLL")
                #     break
                if x in img_match_coor[y]:
                    # print("CHECK", img_match_coor[y])
                    if x != img_match_coor[y][z]:
                        x_cores = img_match_coor[y][z]
                        final_order_list.append(x_cores)
                        print(x,"CHRCK",img_match_coor[y],"YEHS HAI X CORES", x_cores)
                        img_match_coor.remove(img_match_coor[y])
                        print(img_match_coor)
                        for io_ in range(len(img_match_coor)):
                            if len(img_match_coor) == 1:
                                break
                            for iz in range(len(img_match_coor[io_])):
                                if x_cores in img_match_coor[io_]:
                                    if x_cores != img_match_coor[io_][iz]:
                                        x_cores_up = img_match_coor[io_][iz]
                                        final_order_list.append(x_cores_up)
                                        print(x_cores,"CHECK NEW",img_match_coor[io_],"YEH HAI UPDATED",x_cores_up)
                                        img_match_coor.remove(img_match_coor[io_])
                                        print(img_match_coor)
                                        for io1_ in range(len(img_match_coor)):
                                            if len(img_match_coor) == 1:
                                                break
                                            for iz1 in range(len(img_match_coor[io1_])):
                                                if x_cores_up in img_match_coor[io1_]:
                                                    if x_cores_up != img_match_coor[io1_][iz1]:
                                                        x_cores_up_up = img_match_coor[io1_][iz1]
                                                        final_order_list.append(x_cores_up_up)
                                                        print(x_cores_up,"CHECK EKDUM NEW",img_match_coor[io1_],"YEH HAI UDDDD",x_cores_up_up)
                                                        img_match_coor.remove(img_match_coor[io1_])
                                                        print(img_match_coor)
                                        if len(img_match_coor) == 1:
                                            break
                        print("LEN", len(img_match_coor))
                        if len(img_match_coor) == 1:
                            break
                        
    
    print(final_order_list)
    homography_list = list()

    print("Arranging Image pairs in acatual(panoramic) order ................. In process")

    

    len_ord_fin = len(final_order_list)

    tkp1 = img_nos[final_order_list[0]][0]
    tdes1 = img_nos[final_order_list[0]][1]
    tkp2 = img_nos[final_order_list[1]][0]
    tdes2 = img_nos[final_order_list[1]][1]

    x1 = match_keypoints(tkp1,tkp2,tdes1,tdes2,inp_img[final_order_list[0]],inp_img[final_order_list[1]])
    y1 = (RANSAC(x1,400))
    check1_  = np.matmul(y1, np.array([0, 0, 1]))
    if check1_[0]/check1_[2] < 0:
        print(check1_[0]/check1_[2])
    else:
        final_order_list.reverse()
        print(check1_[0]/check1_[2])

    # print(f'This is final list now: {final_order_list}')
    print("This is final list now:", final_order_list)
    # t_match = match_keypoints()
    print("Image ORDERING ....... COMPLETED")

    ########################------------------------- MAIN STICHING PROGRAM STARTS HERE ---------------------------------------------------##########################

    print("Image Stitching ..... Started")
    def panorama_(i1, i2):
        """
        THIS FUNCTION DOES THE STITCHING OF TWO IMAGES
        INPUT: TWO IMAGE
        OUTPUT: RESULTED IMAGE
        """

        kp1, des1 = sift.detectAndCompute(i1,None)
        kp2, des2 = sift.detectAndCompute(i2,None)

        correspondence_ = match_keypoints(kp1,kp2,des1,des2,i1,i2)
        H  = RANSAC(correspondence_,400)
        H = np.linalg.inv(H)
        stitched_ = cv2.warpPerspective(i2,H,(i1.shape[1]+ i2.shape[1],i1.shape[0]))
        stitched_[0:i1.shape[0],0:i1.shape[1]] = i1
        # plt.figure(figsize=(16,14)) 
        # plt.imshow(stitched_)
        # plt.show()        

        return stitched_

    final_order_list.reverse()
    panorama = inp_img[final_order_list[0]]
    
    img_width = 0
    img_height = 0

    for index_,image_ in enumerate(final_order_list[:-1]):
        kp_last_image,des_last_image = sift.detectAndCompute(panorama,None)
        previous_image = inp_img[final_order_list[index_+1]]
        kp_previous_image, descriptor_previous_image  = sift.detectAndCompute(previous_image,None)
        xx = match_keypoints(kp_last_image,kp_previous_image,des_last_image,descriptor_previous_image,panorama,previous_image)
        H = RANSAC(xx,500)
        # print(h)
        img_width += panorama.shape[1] + previous_image.shape[1]
        img_height = previous_image.shape[0]
        panorama = cv2.warpPerspective(panorama,(H),(img_width,img_height))
        panorama[0:previous_image.shape[0], 0:previous_image.shape[1]] = previous_image 

        # plt.figure()
        # plt.title("JOILE")
        # plt.imshow(panorama)
        # print("LE MARO")


    plt.figure()
    plt.title("Magnifincento!")
    plt.imshow(panorama)
    plt.show()

    # cv2.imwrite("FINAL",panorama)
    # panorama = cv2.cvtColor(panorama,cv2.COLOR_GRAY2RGB)
    # pan_save = Image.fromarray(panorama)
    # pan_save.save('PANORAMA.png')


    img = np.asarray(panorama, dtype=np.uint8)
    cv2.imwrite(os.path.join(folder , 'panorama.jpg'), img)
    cv2.waitKey(0)

    print('Image Stitching Completed! ')
    print(f"Please check in source folder: {folder}")
    
if __name__ == "__main__":
        main()
    