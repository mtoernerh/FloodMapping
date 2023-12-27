# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:36:25 2023

@author: mfth
"""
import numpy as np
from tqdm import tqdm
from scipy.ndimage import generic_filter
from collections import namedtuple

from region_growing import region_growing_sar
from sar_analysis import GMMBimodalityTester, KDELocalMinimumFinder
from utility import extract_coords_and_values, impute_values, calculate_tpi, find_bounding_box
 
def box_decomposition(box_sizes, array, global_threshold, row, col, upper_lower = False):
    """
    Perform box decomposition algorithm on the given array.

    Parameters:
    - box_sizes (list): List of box sizes to iterate over.
    - array (numpy.ndarray): Input 2D array.
    - global_threshold (float): Global threshold for the bimodality test.
    - row (int): Row index for the center of the box.
    - col (int): Column index for the center of the box.

    Returns:
    BoxDecompositionResult: Named tuple containing decomposition results.
    """
    if upper_lower:
    # Define a named tuple for the results
        BoxDecompositionResult = namedtuple( 'BoxDecompositionResult',
        ['lower_threshold', 'local_threshold', 'upper_threshold'])
    else:
        BoxDecompositionResult = namedtuple( 'BoxDecompositionResult',
        ['local_threshold'])
    # Iterating over windows, starting with the biggest
    GMMBimodal = GMMBimodalityTester(plot=False)
    kde_min_finder = KDELocalMinimumFinder()
    for box in box_sizes:
        # Retriving the absolute row, col index
        row_start = max(0, row - box // 2)
        row_end   = min(array.shape[0], row + box // 2)
        col_start = max(0, col - box // 2)
        col_end   = min(array.shape[1], col + box // 2)
        
        # Creating subarray to perform bimodality test
        subarray = array[row_start:row_end, col_start:col_end]
        # Bimodality test
        GMMBimodal.fit(subarray)
        GMMBimodal_check, n1, n2 = GMMBimodal.test_bimodality(subarray)
        # If bimodal distribution is True, move on the finding local threshold
        if GMMBimodal_check:
            kde_min_finder.fit(subarray)
            try:
                lower_threshold, local_threshold, upper_threshold = kde_min_finder.find_local_threshold(global_threshold, n1, n2, method = 'crossing_point')
            except ValueError:
                lower_threshold, local_threshold, upper_threshold = np.nan, np.nan, np.nan
            if np.isnan(local_threshold):
                continue
            else:
                if upper_lower:
                    # Returning the local threshold along with the absolute row and col index values
                    return BoxDecompositionResult(lower_threshold, local_threshold, upper_threshold)
                else:
                    return BoxDecompositionResult(local_threshold)
    if upper_lower:
        # If no valid result is found, return a named tuple with NaN values
        return BoxDecompositionResult(np.nan, np.nan, np.nan)
    else:
        return BoxDecompositionResult(np.nan)

def box_segmentation(index_pairs, array, dem, global_threshold, box_sizes, upper_lower = False):
    # Height and width of S1 array
    array_height, array_width = array.shape[:2]
    if upper_lower:
        RG_output = np.full((array_height, array_width, 3), np.nan)
    else:
        RG_output = np.full((array_height, array_width, 1), np.nan)
    RG_temporary = np.full((array_height, array_width), np.nan)
    for row, col in tqdm(index_pairs, desc="Processing", unit="pair"):
        # If the given row,col already has a value higher than 0, it has already been processed
        if RG_output[row,col,0] > 0:
            continue
        else:
            # Applying box decomposition algorithm on the given row,col 
            result = box_decomposition(box_sizes, array, global_threshold, row, col, upper_lower)
            
            local_threshold = result.local_threshold
            if upper_lower:
                lower_threshold = result.lower_threshold
                upper_threshold = result.upper_threshold
                thresholds = [local_threshold, lower_threshold, upper_threshold]
            else:
                thresholds = [local_threshold]
            # If the local minimum could not be determined move on to next row col immediately
            if np.isnan(local_threshold):
                continue
            else:
                for threshold_level in range(len(thresholds)):
                    if RG_output[row, col, threshold_level] > 0: 
                        continue
                   
                    # Store the modified subarray in a temporary output array
                    RG_temporary = np.zeros((array.shape[0], array.shape[1]), dtype=float)
                    
                    # RG from a single point as seed (row, col)
                    RG_temporary[row, col] = np.float32(thresholds[threshold_level])
                    
                    # Retriving seed row,col index and associated threshold values (only makes sense when using RG with all cells below threshold as seeds)
                    seeds, values = extract_coords_and_values(RG_temporary)
                    
                    # Applying Region Growing algorithm
                    segmentation = region_growing_sar(array, seeds, values, connectivity = "Moore")
                    
                    # Retriving index pair and values of region grown segmented area
                    RG_coords, RG_values = extract_coords_and_values(segmentation)
                    
                    # Get coordinates of segmented feature
                    seg_coords = np.where(segmentation == 1)
                    
                    # Retrieve boundary box index values
                    try:
                        min_row, max_row = min(seg_coords[0]), max(seg_coords[0])
                        min_col, max_col = min(seg_coords[1]), max(seg_coords[1])
                    except ValueError:
                        continue
                    
                    # Create bounding box
                    bounding_box = find_bounding_box(min_row, min_col, max_row, max_col, 25)
                    
                    # Applying bounding box on segmented feature and elevation
                    seg_bb = segmentation[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]]
                    dem_bb = dem[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]]
                    
                    # Calculate the offset between smaller and larger arrays
                    x_offset, y_offset = bounding_box[1], bounding_box[0]
                    # Defining neighborhood variable for tpi function
                    neighborhood_size = (25,25)
                    
                    # Calculating TPI for bounding box
                    tpi_array = generic_filter(dem_bb, calculate_tpi, footprint=np.ones(neighborhood_size))
                    
                    # Masking with segmented feature
                    tpi_feature = tpi_array[seg_bb == 1]
                    
                    # Counting pixels with positive tpi
                    tpi_positive_count = np.sum(tpi_feature > 0)
                    
                    # Counting total number of pixels within segmented feature
                    feature_total_count = np.sum(seg_bb == 1)
                    
                    # Calculating ratio within feature
                    tpi_positive_ratio =  tpi_positive_count/feature_total_count
                    
                    # Compare mean elevation inside and outside segmented area
                    mean_ele1 = np.nanmean(dem_bb[seg_bb == 1])
                    mean_ele0 = np.nanmean(dem_bb[seg_bb == 0])
                    
                    if tpi_positive_ratio >= 0.5 or mean_ele1 >= mean_ele0:
                        # Set value to 2 to indicate false segmentation
                        segmentation[segmentation == 1] = 2
                        RG_coords, RG_values = extract_coords_and_values(segmentation)
                        # Storing segmenetation result in contiounsly updating array
                        RG_output[:,:,threshold_level] = impute_values(RG_coords, RG_values, RG_output[:,:,threshold_level])
                    else:
                        # Masking out segmented pixels with TPI higher than 0.5 (experimental)
                        RG_temporary = np.full((array_height, array_width), np.nan)
                        try:
                            seg_bb[tpi_array > 0.5] = 0
                            RG_temporary[y_offset:(y_offset + tpi_array.shape[0]), x_offset:(x_offset + tpi_array.shape[1])] = seg_bb
                        except ValueError:
                            RG_temporary = segmentation
                            
                        RG_coords, RG_values = extract_coords_and_values(RG_temporary)
                        # Add another mask that removes pixels without x neighbors, and fill out holes if some size
                        # Various post-processing techniques could be added here, before segmentation is added to output
                        RG_output[:,:,threshold_level] = impute_values(RG_coords, RG_values, RG_output[:,:,threshold_level])
    return RG_output