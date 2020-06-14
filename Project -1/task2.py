"""
Character Detection

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import math

import utils
from task1 import *   # you could modify this line


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="data/proj1-task2-png.png",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="data/c.png",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def size_of_image(kernel):
    '''
    Gives the sizew of the image/kernel! 

    '''
    rows_ = len(kernel)
    col_ = len(kernel[0])

    return rows_,col_

def mean_(image):
    '''
    Calculates the mean of the image or kernel wherever applied!

    '''
    img_sum = 0
    for img_r in image:
        for img_c in img_r:
            img_sum = img_sum + img_c
    img_mean = img_sum/(len(image)*len(image[0]))

    image = img_mean
    return image

def edge_detection(image):
    '''
    Finds the Gradient direction using edge detection from the Task1.py file

    '''
    image_edge_x = detect_edges(image,sobel_x,True)
    image_edge_y = detect_edges(image,sobel_y,True)

    image_edges = copy.deepcopy(image_edge_x)

    for ii, row in enumerate(image_edge_y):
        for jj, col in enumerate(row):
            image_edges[ii][jj] = math.atan2(image_edge_y[ii][jj],image_edge_x[ii][jj])

    return image_edges   

  


def detect(img, template):
    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    # TODO: implement this function.
    # raise NotImplementedError


    temp_rowc = len(template)
    temp_colc = len(template[0])

    # template_edges = edge_detection(template)
    template_edges = template
    template_mean_ = mean_(template_edges)


    for temp_index,temp_row in enumerate(template_edges):
        for temp_col_index,temp_col in enumerate(temp_row):
            template_edges[temp_index][temp_col_index] = (template_edges[temp_index][temp_col_index] - template_mean_)
    
    template_1_sqre = copy.deepcopy(template_edges)
    for numbers_i,numbers_v in enumerate(template_edges):
        for numbers_j,numbers_w in enumerate(numbers_v):
            template_1_sqre[numbers_i][numbers_j] = (template_edges[numbers_i][numbers_j])**2

    sum_ = 0
    for xx, xx_val in enumerate(template_1_sqre):
        for yy,yy_val in enumerate(xx_val):
            sum_ = sum_ + yy_val

    template_1_sum = sum_
    template_1_sqrt = template_1_sum ** 0.5
    print(template_1_sqrt)

    coordinates = []

    for img_row_index,img_row in enumerate(img[:-temp_rowc]):
        for img_col_index,img_col in enumerate(img_row[:-temp_colc]):
            cropped_edge_image = utils.crop(img,img_row_index,(img_row_index+temp_rowc),img_col_index,(img_col_index+temp_colc))
            # cropped_edge_image = edge_detection(cropped_image)                     
            
            cropped_mean = mean_(cropped_edge_image)

            cropped_image_1 = copy.deepcopy(cropped_edge_image)

            # for crpped_index,crpped_val in enumerate(cropped_image_1):
            #     for crpped_col_index,crpped_cval in enumerate(crpped_val):
            #         cropped_image_1[crpped_index][crpped_col_index] = (cropped_image_1[crpped_index][crpped_col_index] - cropped_mean)

            cropped_image_1 = (cropped_image_1 - cropped_mean)
            
            # cropped_image_1_sqre = copy.deepcopy(cropped_image_1)
            # for crpped_sqr_index,crpped_sqr_val in enumerate(cropped_image_1):
            #     for crpped_sqr_col_index,crpped_sqr_col in enumerate(crpped_sqr_val):
            #         cropped_image_1_sqre[crpped_sqr_index][ crpped_sqr_col_index] = (cropped_image_1_sqre[crpped_sqr_index][crpped_sqr_col_index])**2

            # cropped_image_1_sqre = (sum(sum((cropped_image_1)**2)))


            # crpped_sum_ = 0
            # for crpped_sum_index,crpped_sum_val in enumerate(cropped_image_1_sqre):
            #     for crpped_yy,crpped_val in enumerate(crpped_sum_val):
            #         crpped_sum_ = crpped_sum_ + crpped_val
            
            # crpped_1_sum = crpped_sum_
            crpped_1_sum = (sum(sum((cropped_image_1)**2)))          
            crpped_1_sqrt = (crpped_1_sum)**0.5

            ncc_num_mul = utils.elementwise_mul(template_edges,cropped_image_1)
            
            ncc_num_sum = 0
            for i,i_val in enumerate(ncc_num_mul):
                for j,j_val in enumerate(i_val):
                    ncc_num_sum = ncc_num_sum + j_val

            ncc_final_num = ncc_num_sum
            # print(template_1_sqrt,crpped_1_sqrt)

            ncc_main = ncc_final_num/(template_1_sqrt * crpped_1_sqrt)
            # print(ncc_main)
            if ncc_main > 0.785:
                print(ncc_main)
                coordinates.append([img_row_index,img_col_index])


    return coordinates


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    template = read_image(args.template_path)
    print(size_of_image(img))
    print(size_of_image(template))

    coordinates = detect(img,template)
    print(coordinates)
    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()

