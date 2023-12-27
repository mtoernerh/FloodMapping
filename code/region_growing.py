# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 07:52:40 2023

@author: mfth
"""
import numpy as np
from queue import Queue

# Region growing function
def region_growing_sar(backscatter_array, coordinates_list, thresholds, connectivity = "Moore"):
    height, width = backscatter_array.shape
    segmented_array = np.zeros((height, width), dtype=np.uint8)

    # Create a queue for region-growing process
    queue = Queue()

    # Define 8-connectivity neighbors (adjust as needed)
    if connectivity == "Moore":
        neighbors = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    if connectivity == "Cardinal":
        neighbors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    # Handle NaN values in the backscatter array
    backscatter_array = np.nan_to_num(backscatter_array, nan=np.inf)

    for i, (x, y) in enumerate(coordinates_list):
        if segmented_array[x, y] == 0:
            threshold_value = thresholds[i]
            backscatter_value = backscatter_array[x, y]
            if backscatter_value <= threshold_value:
                segmented_array[x, y] = 1
                queue.put((x, y))

                # Region-growing process
                while not queue.empty():
                    x_seed, y_seed = queue.get()

                    for dx, dy in neighbors:
                        nx, ny = x_seed + dx, y_seed + dy

                        # Check if the neighboring pixel is within the array boundaries and not processed before
                        if 0 <= nx < height and 0 <= ny < width and segmented_array[nx, ny] == 0:
                            threshold_value = thresholds[i]
                            backscatter_value = backscatter_array[nx, ny]
                            if backscatter_value <= threshold_value:
                                segmented_array[nx, ny] = 1
                                queue.put((nx, ny))

    return segmented_array

def region_growing_elevation(array1, array2, coordinates_list, thresholds1, thresholds2, max_segmented_cells, connectivity = "Moore"):
    height, width = array1.shape
    segmented_array = np.zeros((height, width), dtype=np.uint8)

    # Create a queue for region-growing process
    queue = Queue()

    # Define 8-connectivity neighbors (adjust as needed)
    if connectivity == "Moore":
        neighbors = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    if connectivity == "Cardinal":
        neighbors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    # Handle NaN values in the backscatter array
    array1 = np.nan_to_num(array1, nan=np.inf)
    array2 = np.nan_to_num(array2, nan=np.inf)
    
    # Counter for segmented cells
    segmented_cell_count = 0
    
    for i, (x, y) in enumerate(coordinates_list):
        if segmented_array[x, y] == 0:
            threshold_value1 = thresholds1[i]
            threshold_value2 = thresholds2[i]
            
            value1 = array1[x, y]
            value2 = array2[x, y]
            if value1 <= threshold_value1 and value2 >= threshold_value2 - 5:
                segmented_array[x, y] = 1
                queue.put((x, y))
                segmented_cell_count += 1
                
                # Region-growing process
                while not queue.empty():
                    x_seed, y_seed = queue.get()

                    for dx, dy in neighbors:
                        nx, ny = x_seed + dx, y_seed + dy

                        # Check if the neighboring pixel is within the array boundaries and not processed before
                        if 0 <= nx < height and 0 <= ny < width and segmented_array[nx, ny] == 0:
                            threshold_value1 = thresholds1[i]
                            threshold_value2 = thresholds2[i]
                            value1 = array1[nx, ny]
                            value2 = array2[nx, ny]
                            if value1 <= threshold_value1 and value2 >= threshold_value2 - 5:
                                segmented_array[nx, ny] = 1
                                queue.put((nx, ny))
                                segmented_cell_count += 1
                                
                                # Check if the number of segmented cells exceeds the limit
                                if segmented_cell_count >= max_segmented_cells:
                                    return  np.zeros((height, width), dtype=np.uint8)  # Return empty array 
    return segmented_array