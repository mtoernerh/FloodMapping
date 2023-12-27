# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:53:47 2023

@author: mfth
"""

import numpy as np 
from tqdm import tqdm
from scipy.ndimage import generic_filter


from region_growing import region_growing_sar
from sar_analysis import GMMBimodalityTester, KDELocalMinimumFinder
from utility import extract_coords_and_values, impute_values, calculate_tpi, find_bounding_box

class Quadrant():
    def __init__(self, data, bbox, depth, GLOBAL_THRESHOLD):
        self.bbox = bbox
        self.depth = depth
        self.children = None
        self.leaf = False
        #self.initialize_analysis = None

        # crop data to quadrant size
        data = data[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        # Call GMMBimodalityTester and KDELocalMinimumFinder once
        self.initialize_analysis(data, GLOBAL_THRESHOLD)
        
    def initialize_analysis(self, data, GLOBAL_THRESHOLD):
        GMMBimodal = GMMBimodalityTester(plot=False)
        GMMBimodal.fit(data)
        self.result = GMMBimodal.test_bimodality(data)
        self.local_minima = None

        if self.result[0]:
            n1, n2 = self.result[1], self.result[2]
            local_min_finder = KDELocalMinimumFinder()
            local_min_finder.fit(data)
            
            lower_threshold, local_threshold, upper_threshold = local_min_finder.find_local_threshold(GLOBAL_THRESHOLD, n1, n2, method='crossing_point')
            
            self.local_minima = local_threshold
            
        
        
        def reset_state(self):
            # Reset the state of the Quadrant
            self.analysis_initialized = None
            self.result = None
            self.local_minima = None

            
    def is_leaf(self, custom_depth):
        return self.depth == custom_depth or (self.local_minima is not None and self.result[0])
    
    def split_quadrant(self, data, GLOBAL_THRESHOLD):
        if not self.is_leaf(custom_depth=4):
            left, top, width, height = self.bbox
    
            # get the middle coords of bbox
            middle_x = left + (width - left) // 2
            middle_y = top + (height - top) // 2
    
            # split root quadrant into 4 new quadrants
            upper_left = Quadrant(data, (left, top, middle_x, middle_y), self.depth+1, GLOBAL_THRESHOLD)
            upper_right = Quadrant(data, (middle_x, top, width, middle_y), self.depth+1, GLOBAL_THRESHOLD)
            bottom_left = Quadrant(data, (left, middle_y, middle_x, height), self.depth+1, GLOBAL_THRESHOLD)
            bottom_right = Quadrant(data, (middle_x, middle_y, width, height), self.depth+1, GLOBAL_THRESHOLD)
    
            # add new quadrants to root children
            self.children = [upper_left, upper_right, bottom_left, bottom_right]

class QuadTree():
    def __init__(self, data, MAX_DEPTH, GLOBAL_THRESHOLD):
        self.height, self.width = data.shape

        # keep track of max depth achieved by recursion
        self.max_depth = 0

        # start compression
        self.start(data, MAX_DEPTH, GLOBAL_THRESHOLD)
        
    def start(self, data, MAX_DEPTH, GLOBAL_THRESHOLD):
        # create initial root
        self.root = Quadrant(data, (0, 0, self.width, self.height), 0, GLOBAL_THRESHOLD)
        
        # build quadtree
        self.build(self.root, data, MAX_DEPTH, GLOBAL_THRESHOLD)

    def build(self, root, data, MAX_DEPTH, GLOBAL_THRESHOLD):
        if root.depth >= MAX_DEPTH or root.is_leaf(custom_depth=4):
            if root.depth > self.max_depth:
                self.max_depth = root.depth

            # assign quadrant to leaf and stop recursing
            root.leaf = True
            return 
        
        # split quadrant if there is too much detail
        root.split_quadrant(data, GLOBAL_THRESHOLD)
        
        if root.children is not None:
            for child in root.children:
                self.build(child, data, MAX_DEPTH, GLOBAL_THRESHOLD)
    
    def reset_state(self):
        # Reset the state of the QuadTree
        self.max_depth = 0
        self.root = None
    
    def create_image(self, max_depth):
            # create blank image canvas (NumPy array)
            image = np.zeros((self.height, self.width), dtype=float)
            
            for custom_depth in range(max_depth + 1):
                leaf_quadrants = self.get_leaf_quadrants(custom_depth)
    
                # draw rectangle size of quadrant for each leaf quadrant
                for quadrant in leaf_quadrants:
                    # Convert bbox coordinates to integers
                    bbox_int = tuple(map(int, quadrant.bbox))
                    # Draw rectangle on the canvas
                    image[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]] = quadrant.local_minima
    
            return image
    
    def get_leaf_quadrants(self, custom_depth):
        quadrants = []

        # search recursively down the quadtree
        self.recursive_search(self.root, custom_depth, quadrants.append) #

        return quadrants

    def recursive_search(self, quadrant, custom_depth, append_leaf):
        if quadrant.is_leaf(custom_depth):
            append_leaf(quadrant)
        elif quadrant.children is not None:
            for child in quadrant.children:
                self.recursive_search(child, custom_depth, append_leaf)
                

def quadtree_segmentation(array, dem, GLOBAL_THRESHOLD, subarray_size = 256, MAX_DEPTH = 4, upper_lower = False):
    
    # Height and width of S1 array
    array_height, array_width = array.shape[:2]
    if upper_lower:
        RG_output = np.full((array_height, array_width, 3), np.nan)
    else:
        RG_output = np.full((array_height, array_width, 1), np.nan)
    RG_temporary = np.full((array_height, array_width), np.nan)
    output_temporary = np.zeros((array_height, array_width), dtype=float)
    
    for i in tqdm(range(0, array_height, subarray_size),desc="Processing", unit="Row"):
        for j in range(0, array_width, subarray_size):
            
            subarray = array[i:i+subarray_size, j:j+subarray_size]
            quad_tree = QuadTree(subarray, MAX_DEPTH, GLOBAL_THRESHOLD)
            
            output_temporary[:] = np.nan
            RG_temporary = np.full((array_height, array_width), np.nan)
            # Assuming you have a method to create an image from the quadtree
            subarray_result = quad_tree.create_image(MAX_DEPTH)
            output_temporary[i:i+subarray_size, j:j+subarray_size] = np.asarray(subarray_result)
            
            # Reset classes ?
           # quad_tree.Quadrant.reset_state()
            quad_tree.reset_state()
            
            # Example of seeding
            output_temporary_2 = np.where(output_temporary < -1, array, np.nan)
            percentile_5 = np.percentile(output_temporary_2, 5)
            output_temporary_2[output_temporary_2 > percentile_5] = np.nan
            
            
            output_temporary = np.where(output_temporary_2 < -1, output_temporary, np.nan)
            
            # Retriving seed row,col index and associated threshold values (only makes sense when using RG with all cells below threshold as seeds)
            seeds, values = extract_coords_and_values(output_temporary)
            
            # Applying Region Growing algorithm
            segmentation = region_growing_sar(array, seeds, values, connectivity = "Cardinal")
            
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
                RG_output = impute_values(RG_coords, RG_values, RG_output)
            else:
                # Masking out segmented pixels with TPI higher than 0.5 (experimental)
                RG_temporary = np.full((array_height, array_width), np.nan)
    
                try:
                    # RG_temporary[x_offset:(x_offset + tpi_array.shape[0]), y_offset:(y_offset + tpi_array.shape[1])] = tpi_array
                    seg_bb[tpi_array > 0.5] = 0
                    RG_temporary[y_offset:(y_offset + tpi_array.shape[0]), x_offset:(x_offset + tpi_array.shape[1])] = seg_bb
                except ValueError:
                #    print(f"is {row} and {col} close to the edge?") # Nothing should happen, just move on the with the script
                    # Precausion
                    RG_temporary = segmentation
                 
                RG_coords, RG_values = extract_coords_and_values(RG_temporary)
                RG_output = impute_values(RG_coords, RG_values, RG_output)
    return RG_output