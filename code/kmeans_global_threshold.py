# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:15:58 2023

@author: mfth
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.morphology import medial_axis, remove_small_holes
from scipy import ndimage
from osgeo import gdal
import warnings


def kmeans_global_threshold(array, mask, save_geotiff = False, output_path=None, dataset = None, data_format="dB", log = True):
    """
    Parameters
    ----------
    array : numpy.ndarray
        2D array representing the input data.
        
    mask : numpy.ndarray
        2D array binary mask representing land (1) and ocean (0).
        
    save_geotiff : bool, optional
        If True, a GeoTIFF file will be saved. The default is False.
        
    output_path : str, optional
        The file path for saving the GeoTIFF. Required if save_geotiff is True.
        
    dataset : TYPE, optional
        An object representing a raster dataset and storing transform and crs. It can be created using gdal.Open().  Required if save_geotiff is True.
        
    data_format : str, optional
        The format of the input data. Can be "dB" (default) or "Power" (Power transformed sigma nought backscatter values).

    Returns
    -------
    sampled_indices : list of tuple
        List of index tuples (y, x) representing sampled points.
        
    global_threshold : float
        The calculated global upper water threshold.

    """
    warnings.filterwarnings("ignore", message="Any labeled images will be returned as a boolean array. Did you mean to use a boolean array?")
    
    
    array = array.clip(np.nanpercentile(array, 1), np.nanpercentile(array, 99))
    
    array = np.where(mask == 0, np.nan, array)
    
    # Saving dimensions
    x, y = array.shape
    
    # Flatten the clipped data
    flattened_data = array.reshape(-1)

    # Find NaN values and store them
    nan_mask = np.isnan(flattened_data)
    non_nan_data = flattened_data[~nan_mask]
    standardized_data = non_nan_data.reshape(-1, 1)
    # Standardize non-NaN data
    scaler = StandardScaler()
    #standardized_data = scaler.fit_transform(non_nan_data.reshape(-1, 1))
    
    if log:
        standardized_data = np.log1p(non_nan_data.reshape(-1, 1))
        standardized_data = scaler.fit_transform(standardized_data)
    scaled_inertia_values = []
    
    # Calculating KMeans with two to twelve components
    for k in range(2, 12):  
        kmeans = KMeans(n_clusters=k, random_state=42, init="k-means++", n_init='auto')
        kmeans.fit(standardized_data)
        scaled_inertia = kmeans.inertia_ / np.square((standardized_data - standardized_data.mean(axis=0))).sum() + 0.02 * k
        scaled_inertia_values.append(scaled_inertia)
        
    # Finding KMeans with best fit
    best_k_index = scaled_inertia_values.index(min(scaled_inertia_values))
    best_k = best_k_index + 2  # Add 2 because starting from 2 clusters
    print(f"Best number of clusters (Scaled Inertia): {best_k}")
    
    kmeans = KMeans(n_clusters=best_k, random_state=42,  n_init='auto')
    kmeans.fit(standardized_data)
    
    # Create 2D array
    cluster_assignments = np.empty((x * y,), dtype=int)
    cluster_assignments.fill(-1)  
    cluster_assignments[~nan_mask[:]] = kmeans.labels_
    cluster_assignments = cluster_assignments.reshape(x, y)
    
    # Calculating median for each cluster
    unique_labels = np.unique(cluster_assignments)
    # Calculate medians for each label
    label_medians = {
        label: np.nanpercentile(array[cluster_assignments == label], 50)
        for label in unique_labels
        if label != -1  # Exclude points labeled as -1 (if applicable)
        and np.nanpercentile(array[cluster_assignments == label], 50) != 0  # Exclude labels with median 0
    }
    
    # Finding cluster with lowest median (should represent water)
    min_median_label = min(label_medians, key=label_medians.get)
    
    # Creating binary with water (1) and background (0)
    cluster_assignments[cluster_assignments != min_median_label] = 0
    cluster_assignments[cluster_assignments == min_median_label] = 1
    
    # Test remove small clusters
    cluster_assignments = remove_small_holes(cluster_assignments, 25)
    
    # Label connected components
    labeled_array, num_features = ndimage.label(cluster_assignments)
    
    # Get the size of each cluster
    cluster_sizes = ndimage.sum(cluster_assignments, labeled_array, range(num_features + 1))
    
    # Set a threshold to filter out small clusters
    threshold = 25
    filtered_data = labeled_array.copy()
    filtered_data[np.isin(labeled_array, np.where(cluster_sizes < threshold))] = 0
    
    # Skeletonization
    skeletonized_data = medial_axis(filtered_data)
    clustered_skeleton = skeletonized_data * filtered_data
    unique_cluster_ids = np.unique(clustered_skeleton)

    sampled_indices = []
    
    for cluster_id in unique_cluster_ids:
        cluster_size = np.count_nonzero(filtered_data == cluster_id)
        num_samples = 4 if cluster_size < 800 else (6 if 800 <= cluster_size < 5000 else 9)
        
        cluster_indices = np.argwhere(clustered_skeleton == cluster_id)
      
        y_length = max(cluster_indices[:, 0]) - min(cluster_indices[:, 0])
        x_length = max(cluster_indices[:, 1]) - min(cluster_indices[:, 1])
        
        dominant_dimension = 0 if y_length > x_length else 1
        dominant_length = y_length if y_length > x_length else x_length
          
        min_index = min(cluster_indices[:, dominant_dimension])
        
        sample_positions = np.linspace(min_index, min_index + dominant_length, num_samples, endpoint=True)
        
        for position in sample_positions:
            idx = np.searchsorted(cluster_indices[:, dominant_dimension], position)
            sampled_indices.append(cluster_indices[idx - 1])

    # Set non-zero values to 1
    filtered_data[filtered_data > 0] = 1
    
    # Calculating a global threshold, corrected global threshold is only used as an input for quadtree and box thresholding
    threshold = np.nanpercentile(array[cluster_assignments == 1], 99)
    std = np.nanstd(array[cluster_assignments == 1])
    seeds = len(sampled_indices)
    correction_factor = ((abs(threshold) / seeds) * 100 + std)
    global_threshold = threshold + correction_factor
    
    
    # Defining upper limit for global threshold, change this yourself if not suitable
    if data_format == "dB" and global_threshold > -11:
        global_threshold = -11
    elif data_format == "Power" and global_threshold > 0.776:
        global_threshold = 0.776
    elif data_format == "None":
        global_threshold = global_threshold
    
    if save_geotiff and output_path:
        # Saving water cluster as GeoTiff
        ds_out = gdal.GetDriverByName('GTiff').Create(output_path, filtered_data.shape[1], filtered_data.shape[0], 1, gdal.GDT_Int16)
        ds_out.SetGeoTransform(dataset.GetGeoTransform())
        ds_out.SetProjection(dataset.GetProjection())
        ds_out.GetRasterBand(1).WriteArray(filtered_data)
        ds_out = None
    
    print(f"\nNumber of seeds: {len(sampled_indices)} \nGlobal threshold: {threshold} \nCorrected global threshold: {global_threshold}")
    
    # 
    sampled_indices = list(map(tuple, sampled_indices))
    # Returning a list of index tuples (y, x) and global upper water threshold
    return sampled_indices, global_threshold

