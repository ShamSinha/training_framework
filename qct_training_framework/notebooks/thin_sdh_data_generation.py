import torch
import pandas as pd
import random

import numpy as np
import SimpleITK as sitk
import torch
from loguru import logger



import h5py
from tqdm.auto import tqdm

import os
import torch
from qer.utils.db import get_mongo_db
db = get_mongo_db()
from qer.utils.imageoperations.resampler import load_raw_sitk_img
import copy
import skimage.segmentation as skimg_segm
import sqlite3
from qer.ai.predictor.get_predictions import load_and_run_model
from skimage.morphology import skeletonize
from skimage.filters import median

import cv2
import skimage
from skimage.morphology import ball , disk , binary_erosion , binary_dilation, binary_closing, binary_opening
from scipy import stats
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import ball, dilation, flood_fill


def get_mls_icv(sitk_img):
    output = load_and_run_model("mls_helper_icv_quant",sitk_img)
    icv_mask = output['results']['heatmaps']['ICV']
    icv_mask_arr = sitk.GetArrayFromImage(icv_mask)
    midline_output = load_and_run_model("mls_quantification" ,sitk_img)
    mls_arr = sitk.GetArrayFromImage(midline_output['results']['heatmaps']['MLS'])
    return mls_arr , icv_mask_arr


def get_thick_contour(icv_mask_arr , thickness):
    eroded_icv_mask_arr = icv_mask_arr.copy()
    for i in range(eroded_icv_mask_arr.shape[0]):
        eroded_icv_mask_arr[i] = binary_erosion(icv_mask_arr[i], disk(thickness))
    contour = icv_mask_arr-eroded_icv_mask_arr
    return contour


def get_hemi_contour(req_image, req_mls_arr, req_contour) : 
    ml_coordinates = np.argwhere(req_mls_arr==1)
    (x1,y1) = ml_coordinates[0]
    (x2,y2) = ml_coordinates[-1]

    # Calculate the slope (m) and y-intercept (b)
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    # Create an empty mask
    seperator_mask = np.zeros(req_contour.shape, dtype=np.uint8)

    for x in range(req_contour.shape[0]):
        for y in range(req_image.shape[1]):
            if y >= m * x + b:
                seperator_mask[x, y] = 2
            else :
                seperator_mask[x, y] = 1

    final_image_to_work = seperator_mask*req_contour
    return final_image_to_work , seperator_mask


def get_icv_edge(req_icv_mask) : 
    icv_edges = req_icv_mask.copy()
    edge_mask = cv2.Canny(icv_edges.astype(np.uint8), 0, 1)  # Using Canny edge detection

    # Dilate the edge pixels to make the edges thicker
    kernel_size = 2  # Adjust this to control the thickness of the edges
    dilated_edge_mask = cv2.dilate(edge_mask, np.ones((kernel_size, kernel_size), np.uint8))

    # Set the intensity of the thickened edge pixels to a new value (3)
    thick_edge_intensity = 3
    icv_edges[dilated_edge_mask > 0] = thick_edge_intensity
    icv_edges = (icv_edges == 3)*1
    return icv_edges


# Function to calculate distance between two points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_coordinates(final_image_to_work , icv_edges ,side = "left" , section = "frontal", num_segments = 3):
    if side == "left" :
        hemisphere_strip = (final_image_to_work == 1)*1
    if side == "right" :
        hemisphere_strip = (final_image_to_work == 2)*1
        
    # Find indices of non-zero elements (curve pixels)
    curve_indices = np.argwhere(hemisphere_strip == 1)

    # Define the ratios for the three sections
    ratio_section1 = 0.32
    ratio_section2 = 0.36
    # Calculate total number of curve pixels
    total_pixels = len(curve_indices)

    # Calculate lengths of the three sections
    length_section1 = int(total_pixels * ratio_section1)
    length_section2 = int(total_pixels * ratio_section2)

    # Divide the curve indices into three sections
    section1_indices = curve_indices[:length_section1]
    section2_indices = curve_indices[length_section1:length_section1+length_section2]
    section3_indices = curve_indices[length_section1+length_section2:]
   
    section_mask = np.zeros_like(hemisphere_strip)

    for idx in section1_indices:
        section_mask[idx[0], idx[1]] = 1

    for idx in section2_indices:
        section_mask[idx[0], idx[1]] = 2

    for idx in section3_indices:
        section_mask[idx[0], idx[1]] = 3
    
    if section == "frontal" :
        req_coordinates = np.argwhere((section_mask == 1)*1*icv_edges )
        
    if section == "temporal" :
        req_coordinates = np.argwhere((section_mask == 2)*1*icv_edges )
    
    if section == "occipital" :
        req_coordinates = np.argwhere((section_mask == 3)*1*icv_edges )
        

    # Calculate the step size based on the number of segments
    step_size = len(req_coordinates) // num_segments

    # Calculate the indices for the segmented points
    indices = [step_size * i for i in range(num_segments)] + [len(req_coordinates) - 1]

    # Get the coordinates of the segmented points
    segmented_points = [req_coordinates[indices[i]:indices[i+1]+1] for i in range(num_segments)]

    max_distance_coordinates = []

    # Find the pair of coordinates with maximum distance in each segment
    for segment in segmented_points:
        pairs = combinations(segment, 2)
        max_distance_pair = max(pairs, key=lambda pair: distance(pair[0], pair[1]))
        max_distance_coordinates.append(max_distance_pair)
        
    return max_distance_coordinates



def get_all_section(hemisphere_strip) : 
    curve_indices_left = np.argwhere(hemisphere_strip == 1)
    curve_indices_right = np.argwhere(hemisphere_strip == 2)

    # Define the ratios for the three sections
    ratio_section1 = 0.32
    ratio_section2 = 0.36

    # Calculate total number of curve pixels
    total_pixels_left = len(curve_indices_left)
    total_pixels_right = len(curve_indices_right)

    # Calculate lengths of the three sections
    length_section1_left = int(total_pixels_left * ratio_section1)
    length_section2_left = int(total_pixels_left * ratio_section2)
    
    length_section1_right = int(total_pixels_right * ratio_section1)
    length_section2_right = int(total_pixels_right * ratio_section2)

    # Divide the curve indices into three sections
    section1_indices_left = curve_indices_left[:length_section1_left]
    section2_indices_left = curve_indices_left[length_section1_left:length_section1_left+length_section2_left]
    section3_indices_left = curve_indices_left[length_section1_left+length_section2_left:]
    
    # Divide the curve indices into three sections
    section1_indices_right = curve_indices_right[:length_section1_right]
    section2_indices_right = curve_indices_right[length_section1_right:length_section1_right+length_section2_right]
    section3_indices_right = curve_indices_right[length_section1_right+length_section2_right:]

    section_mask = np.zeros_like(hemisphere_strip)

    for idx in section1_indices_left:
        section_mask[idx[0], idx[1]] = 1

    for idx in section2_indices_left:
        section_mask[idx[0], idx[1]] = 2

    for idx in section3_indices_left:
        section_mask[idx[0], idx[1]] = 3
        
    for idx in section1_indices_right:
        section_mask[idx[0], idx[1]] = 4

    for idx in section2_indices_right:
        section_mask[idx[0], idx[1]] = 5

    for idx in section3_indices_right:
        section_mask[idx[0], idx[1]] = 6
        
    return section_mask


def get_point_on_perp_bisector(point1,point2,distance,hemisphere_side,icv_mask,grad) :
    
    x1,y1 = point1 
    x2,y2 = point2
    # Calculate the midpoint between the two given points
    midpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    intersection_x , intersection_y = midpoint
    
    original_slope = (y2 - y1)/(x2-x1)
    perpendicular_slope = -1 / original_slope
        
    # Calculate the angle of the line with respect to the x-axis
    theta = np.arctan(perpendicular_slope)

    # Calculate the x and y components of the desired points
    dx = distance * np.cos(theta)
    dy = distance * np.sin(theta)

    # Calculate the two points on the perpendicular bisector
    point1_perpendicular = (int(intersection_x + dx), int(intersection_y + dy))
    point2_perpendicular = (int(intersection_x - dx), int(intersection_y - dy))

    if grad : 
        if icv_mask[point1_perpendicular[-1] , point1_perpendicular[0] ] == 1 :
            return point1_perpendicular
        if icv_mask[point2_perpendicular[-1] , point2_perpendicular[0] ] == 1 :
            return point2_perpendicular
    else :
        if hemisphere_side == "left" :
            return point1_perpendicular
        if hemisphere_side == "right" : 
            return point2_perpendicular

class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def random(self, min= 0, max= 1):
        self.x = random.uniform(min,max)
        self.y = random.uniform(min,max)

class QuadBezier(object):
    def __init__(self, p0x= 0, p0y= 0, p1x= 0, p1y= 0, p2x= 0, p2y= 0):
        self.p0 = Point(p0x, p0y)
        self.p1 = Point(p1x, p1y)
        self.p2 = Point(p2x, p2y)

    def calc_curve(self, granuality=100):
        'Calculate the quadratic Bezier curve with the given granuality.'
        B_x = []
        B_y = []
        for t in range(0, granuality):
            t = t / granuality
            x = self.p1.x + (1 - t)**2 * (self.p0.x-self.p1.x) + t**2 * (self.p2.x - self.p1.x)
            y = self.p1.y + (1 - t)**2 * (self.p0.y-self.p1.y) + t**2 * (self.p2.y - self.p1.y)
            B_x.append(x)
            B_y.append(y)
        return [B_x, B_y]

def get_largest_island(mask):
    """Extracts the largest island (in terms of number of voxels/pixels) in the mask"""

    th = mask.sum()

    new_value = 1
    largest_island_key = 1
    island = {}
    flooded_mask = mask

    max_size = 0
    sum_size = 0

    while len(np.argwhere(flooded_mask == 1)) > 0:
        seed_points = np.argwhere(flooded_mask == 1)
        new_value = new_value + 1

        flooded_mask = flood_fill(flooded_mask, tuple(seed_points[-1]), new_value)

        island[new_value] = len(np.argwhere(flooded_mask == new_value))

        if island[new_value] > th // 2:
            largest_island_key = new_value
            break

        sum_size += island[new_value]

        if max_size < island[new_value]:
            max_size = island[new_value]
            largest_island_key = new_value

        if max_size > th - sum_size:
            break

    cleaned_mask = np.zeros_like(mask)
    cleaned_mask[flooded_mask == largest_island_key] = 1

    return cleaned_mask




if __name__ == "__main__":
    df_normal = pd.read_csv("/home/users/shubham.kumar/projects/qct_training_framework/notebooks/train_normal_ncct.csv")
    df_normal = df_normal.sample(5000,random_state=85)
    study_uids_to_work  = df_normal.StudyUID.values

    valid_datapath =  pd.read_csv("/home/users/shubham.kumar/projects/qct_training_framework/notebooks/train_normal_ncct_valid_data.csv").datapath.values

    params = (328.5292564674929, -90.53601685038265, 0.47700904381184084)

    base_path = "/cache/fast_data_nas8/qer/shubham/ich"
    # study_uids_to_work = ["1.3.6.1.4.1.25403.52241001737.5304.20150617061532.1","1.2.840.113619.2.278.3.2831165736.718.1344910414.478"]
    study_uids_to_work = ["1.2.840.113619.2.55.3.2831165733.951.1397502532.227"]

    for idx in tqdm(range(4230,len(study_uids_to_work))) :
        uid = study_uids_to_work[idx]
        try :
            path = os.path.join(base_path, f"{uid}.h5")
            if path not in valid_datapath :
                continue
            if os.path.exists(path) : 
                f2 = h5py.File(path , "r")
                arr = f2["image"]
                sitk_img = sitk.GetImageFromArray(arr)
                sitk_img.SetSpacing((1,1,5))

                mls_arr , icv_mask_arr = get_mls_icv(sitk_img)
                contour = get_thick_contour(icv_mask_arr , 7)
                slices_to_pick = np.unique(np.argwhere(mls_arr ==1)[:,0])
                random.shuffle(slices_to_pick)
                slice_to_pick = slices_to_pick[0]

                # select one slice of CT , # 2d image
                req_image  = arr[slice_to_pick]  
                req_mls_arr = mls_arr[slice_to_pick]
                req_icv_mask = icv_mask_arr[slice_to_pick]
                req_contour = contour[slice_to_pick]

                final_image_to_work , seperator_mask = get_hemi_contour(req_image, req_mls_arr, req_contour)
                icv_edges = get_icv_edge(req_icv_mask)

                final_image_to_work_with_enhance_edges = final_image_to_work.copy()
                final_image_to_work_with_enhance_edges[icv_edges==1] = 3

                hemisphere_side_li = ["left","right"]
                section_li = ["frontal","temporal","occipital"]
                num_segments_li = [4,5,6,7]

                hemisphere_side = random.choice(hemisphere_side_li)
                section = random.choice(section_li)
                num_segments = random.choice(num_segments_li)

                segments_to_pick = random.choice(np.arange(num_segments))

                final_coordinates = get_coordinates(final_image_to_work ,icv_edges ,side = hemisphere_side , section = section, num_segments=num_segments)
                point1 ,point2 = final_coordinates[segments_to_pick]

                points_grad_mask =  get_point_on_perp_bisector(point1,point2, 80,hemisphere_side, req_icv_mask,grad=True)

                f2.close()
                for perp_dist in range(0,30) : 
                    point_perpendicular =  get_point_on_perp_bisector(point1,point2, perp_dist,hemisphere_side,req_icv_mask,grad=False)
                    
                    curve = QuadBezier(p0x = point1[0], p0y = point1[1],p1x = point_perpendicular[0] , p1y = point_perpendicular[1],
                                    p2x = point2[0] , p2y = point2[1])

                    final_curve = curve.calc_curve()
                    curve_coordinates = zip(final_curve[0],final_curve[1])
                    
                    final_curve_mask = np.zeros(final_image_to_work.shape)
                    dummy = final_image_to_work_with_enhance_edges.copy()
                    
                    valid_curve = True
                    
                    for pair in curve_coordinates :
                        final_curve_mask[int(pair[0])][int(pair[1])] = 3
                        if dummy[int(pair[0])][int(pair[1])] == 0 :
                            valid_curve = False
                            break
                    if valid_curve == False :
                        continue
                            
                    kernel_size = 1  # Adjust this to control the thickness of the edges
                    dilated_curve_mask = cv2.dilate(final_curve_mask, np.ones((kernel_size, kernel_size), np.uint8))
                    dummy[dilated_curve_mask == 3] = 3
                    final_artificial_sdh_mask = binary_closing(dummy == 3 , disk(4)) - icv_edges

                    rejector = binary_opening(final_artificial_sdh_mask.copy() , disk(1))*req_icv_mask
                        
                    if np.sum(rejector) == 0 :
                        continue

                    final_artificial_sdh_mask = get_largest_island(final_artificial_sdh_mask)
                    final_artificial_sdh_mask = binary_dilation(final_artificial_sdh_mask,disk(1))*req_icv_mask


                    final_mask = np.zeros(icv_mask_arr.shape)
                    final_mask[slice_to_pick] = final_artificial_sdh_mask

                    if valid_curve : 
                        synthetic_data = stats.gamma.rvs(*params, size=np.sum(final_artificial_sdh_mask > 0))
                        synthetic_data.sort()
                        # Apply the generated HU values to the binary mask
                        generated_hu_image = np.zeros(final_artificial_sdh_mask.shape)
                        # generated_hu_image[final_artificial_sdh_mask > 0] = synthetic_data.astype(np.int16)

                        gradient_mask = np.zeros(final_artificial_sdh_mask.shape)
                        for i in range(512):
                            for j in range(512):
                                gradient_mask[i, j] = ((points_grad_mask[0] - i)**2 + (points_grad_mask[1] - j)**2)**(0.5)

                        B = gradient_mask*final_artificial_sdh_mask

                        li = []
                        for (i,j) in np.argwhere(B>0) :
                            li.append((i , j , B[i][j]))
                        sorted_list = sorted(li, key=lambda x: x[-1])

                        for k in range(len(sorted_list)) :
                            tup = sorted_list[k]
                            generated_hu_image[tup[0] , tup[1]] = synthetic_data[k]
                            
                        noise = np.random.normal(loc=0, scale=0.5, size=(512,512))
                        noise[generated_hu_image == 0] = 0
                        generated_hu_image = generated_hu_image + noise

                        generated_hu_image = generated_hu_image.astype(req_image.dtype)

                        req_image_copy = req_image.copy()
                        req_image_copy[final_artificial_sdh_mask > 0] = 0

                        # Alpha blend the blurred Gaussian image and the original image
                        blended_image = cv2.addWeighted(req_image_copy, 1, generated_hu_image, 1, 0)
                        
                        blended_sitk_img = sitk.GetImageFromArray(blended_image)
                        blended_sitk_img.SetSpacing(sitk_img[:,:, int(slice_to_pick)].GetSpacing())
                        blended_sitk_img.SetOrigin(sitk_img[:,:, int(slice_to_pick)].GetOrigin())
                        blended_sitk_img.SetDirection(sitk_img[:,:, int(slice_to_pick)].GetDirection())
                                
                        sitk_img[:,:, int(slice_to_pick)]  = blended_sitk_img
                        artificial_arr = sitk.GetArrayFromImage(sitk_img)
                        results = load_and_run_model("hemorrhages_quantification",sitk_img)
                        mask = sitk.GetArrayFromImage(results['results']['heatmaps']['ICH'])
                        vol = results['results']['quantification value']['ICH']

                        if vol == 0 :                                 
                            h5f = h5py.File(f"/cache/fast_data_nas8/qer/shubham/test/{uid}_{perp_dist}.h5", "w")
                            dseta = h5f.create_dataset("image", data= artificial_arr)
                            dseta = h5f.create_dataset("mask", data = final_mask)
                            # dseta = h5f.create_dataset("icv_edge", data = icv_edges)
                            dseta.attrs['hemisphere_side'] = hemisphere_side
                            dseta.attrs['section'] = section
                            h5f.close()
        except Exception as e:
            logger.debug(e)


                    







                