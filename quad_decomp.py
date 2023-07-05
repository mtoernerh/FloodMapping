# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:01:47 2023

@author: mfth
"""
from operator import add
from functools import reduce
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import seaborn as sns
import diptest
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import normaltest
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['OMP_NUM_THREADS'] = "3"
# Loading data
GTiff_driver = gdal.GetDriverByName("GTiff")
ds = gdal.Open(r"\\netapp2p\DKmodel_users\FloodMapping\Sentinel1_data\Odense_2015_12_26_PP.tif", gdal.GA_Update)
raster = np.array(ds.GetRasterBand(2).ReadAsArray())
raster[raster==0]=np.nan

# Defining function to split raster into 4 equal parts
def split4(raster):
    half_split = np.array_split(raster, 2)
    res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
    return reduce(add, res)

def check_nan_zero_threshold(raster, threshold=0.49):
    total_elements = raster.size
    if total_elements == 0:
        return True
    else:
        zero_count = total_elements - np.count_nonzero(raster)
        nan_count = np.count_nonzero(np.isnan(raster))
        nan_zero_percentage = (nan_count + zero_count) / total_elements
        
        if nan_zero_percentage >= threshold:
            # Skip or continue to the next iteration
            return True
        
        return False

# Performing split
# split_img = split4(raster = ds_np)
# split_img[0].shape

# # Visualizing 
# fig, axs = plt.subplots(2, 2)
# axs[0, 0].imshow(split_img[0])
# axs[0, 1].imshow(split_img[1])
# axs[1, 0].imshow(split_img[2])
# axs[1, 1].imshow(split_img[3])

# Defining function for combining split parts
# def concatenate4(north_west, north_east, south_west, south_east):
#     top = np.concatenate((north_west, north_east), axis=1)
#     bottom = np.concatenate((south_west, south_east), axis=1)
#     return np.concatenate((top, bottom), axis=0)

# Visualizing
# full_img = concatenate4(split_img[0], split_img[1], split_img[2], split_img[3])
# plt.imshow(ds_np)
# plt.show()

# Freeman, J. B., and Dale, R. (2013). Behav. Res. Methods 45, 83–97. doi: 10.3758/s13428-012-0225-x
# BC > 0.555 are taken to indicate bimodality
def bimodality_coefficient(x, na_rm=False):
    # Remove missing values, if desired
    if na_rm:
        x = x[~np.isnan(x)]
    
    n = len(x)
    if n == 0:
        # The coefficient is not defined for empty data
        return np.nan
    else:
       # m3 = moment(x, moment=3)
        m3 = skew(x, axis=0, bias=True)
        #m4 = moment(x, moment=4)
        m4 = kurtosis(x, axis=0, bias=True)
        
        # Calculate the coefficient based on the above
        bc = (m3**2 + 1) / (m4 + 3 * ((n - 1)**2 / ((n - 2) * (n - 3))))
        
        return bc

# https://www.hindawi.com/journals/mpe/2019/4819475/ 
#  See above article for details on bimodality test
def mdip_bimodality_test(raster):
    # if p < 0.05 distribution is multimodal
    # or 0.10 > p > 0.05 
    
    # Flatten raster from 2D array to 1D
    raster_flat = raster.flatten()
   # raster_flat = raster_flat[raster_flat < -7]
    raster_flat = raster_flat[~np.isnan(raster_flat)]
    if check_nan_zero_threshold(raster_flat):
        return False
    if len(raster_flat) < 10:
        return False
    else:
        α_u = 0.32
        α_l = 0.05
        d_statistic, p_value = diptest.diptest(raster_flat)
        bc = bimodality_coefficient(raster_flat)
        α = (α_u - α_l) * bc**2 + α_l
        is_bimodal = p_value < α
        return is_bimodal


# data = np.concatenate((np.random.normal(10, 2, 100), np.random.normal(20, 2, 100)))
# sns.histplot(data, kde=True)
# plt.show()

# raster_flat = split_img[0].flatten()
#sns.histplot(raster_flat, kde=True)
#plt.show()
from scipy.spatial import distance

def gmm_bimodality_test(raster, plot = True):
    # Flatten raster from 2D array to 1D
    raster_flat = raster.flatten()
    #raster_flat = raster_flat[raster_flat < -7]
    raster_flat = raster_flat[~np.isnan(raster_flat)]
    
    if check_nan_zero_threshold(raster_flat):
        return False
    if len(raster_flat) < 10:
        return False
    else:
        # Fit the GMM
        gmm_1 = GaussianMixture(n_components=1, random_state=0, covariance_type='full').fit(raster_flat.reshape(-1, 1))
        gmm_2 = GaussianMixture(n_components=2, random_state=0, covariance_type='full').fit(raster_flat.reshape(-1, 1))
        
        #x_vals = np.linspace(min(raster_flat.reshape(-1)), max(raster_flat.reshape(-1)), 1000)
        x_vals = np.linspace(np.min(raster_flat), np.max(raster_flat), 1000)
        
        # Extract the mean and covariance of each component from the GMM
        means = gmm_2.means_
        covariances = gmm_2.covariances_
        
        # Calculate the Mahalanobis distance for each component
        #distances = [distance.mahalanobis(mean, cov) for mean, cov in zip(means, covariances)]
        std1 = np.sqrt(covariances[0])
        std2 = np.sqrt(covariances[1])
        AD_coefficient = (2**1/2)*((abs(means[0] - means[1]))/abs((std1**2 + std2**2)/2)**1/2)
        # Calculate the Ashman D coefficient
        #ad_coefficient = np.abs(distances[0] - distances[1]) / np.sqrt(2 * (distances[0]**2 + distances[1]**2))
        
        # Compute the weighted mean and covariance of the GMM
        # = gmm_2.weights_
        #weighted_mean = np.average(means, axis=0, weights=weights)
        #weighted_covariance = np.average(covariances, axis=0, weights=weights)
        
        # Calculate the Surface Ratio
        #sr_ratio = weights[0] / weights[1]
        # Calculate the Bhattacharyya coefficient
        #bc_coefficient = 0.25 * weighted_mean[0] * np.linalg.inv(weighted_covariance) * weighted_mean[0]
        
        #[ np.sqrt(  np.trace(gmm2_cov[i])/2) for i in range(0,2) ]
        # Generate x-axis values for plotting

        
        # Calculate the probabilities of the observed data under each GMM
        probs_1 = np.exp(gmm_1.score_samples(x_vals.reshape(-1, 1)))
        probs_2 = np.exp(gmm_2.score_samples(x_vals.reshape(-1, 1)))
        
        # Calculate the density values using the fitted KDE
        log_dens = gmm_2.score_samples(x_vals.reshape(-1, 1))
        dens = np.exp(log_dens)
        if plot:
        # Plot the observed data and the estimated density curves for each GMM
            fig = plt.figure()
            plt.hist(raster_flat, bins='auto', density=True, alpha=0.5, label='Observed values')
            plt.plot(x_vals, probs_1, label='GMM-1')
            plt.plot(x_vals, probs_2, label='GMM-2')
            plt.xlabel('x')
            plt.ylabel('Density')
            plt.title('Gaussian Mixture Models')
            #plt.axvline(gmm_2.score_samples(x_vals.reshape(-1, 1))[test], color='g', ls='--');
            plt.legend()
            plt.show()
        
        # Compare the log-likelihoods of the observed data under each GMM
        #log_likelihood_1 = gmm_1.score(raster_flat.reshape(-1, 1))
        #log_likelihood_2 = gmm_2.score(raster_flat.reshape(-1, 1))
        
        # Compare BIC (Bayesian Information Criterion) and AIC (Akaike Information Criterion)
        gmm_1_bic = gmm_1.bic(raster_flat.reshape(-1, 1))
        gmm_2_bic = gmm_2.bic(raster_flat.reshape(-1, 1))
        gmm_1_aic = gmm_1.aic(raster_flat.reshape(-1, 1))
        gmm_2_aic = gmm_2.aic(raster_flat.reshape(-1, 1))
        
       # print("AIC GMM-1:", gmm_1_aic)
       # print("AIC GMM-2:", gmm_2_aic)
        
       # print("BIC GMM-1:", gmm_1_bic)
       # print("BIC GMM-2:", gmm_2_bic)
        
        if AD_coefficient < 2:
            is_bimodal = False
            print("GMM indicates no bimodality.")
        else:
            is_bimodal = True
            print("GMM indicates bimodality.")
        if plot:
            return(is_bimodal, fig)
        else:
            return(is_bimodal)
        
# def find_optimal_bandwidth(raster):
#     # Flatten raster from 2D array to 1D
#     raster_flat = raster.flatten()
#     raster_flat = raster_flat[~np.isnan(raster_flat)]
    
#     kde = gaussian_kde(raster_flat)
    
#     # Perform grid search for bandwidth values
#     grid = np.linspace(0.05, 1.0, 20)
#     best_bandwidth = None
#     best_score = float('-inf')
    
#     for bandwidth in grid:
#         kde.set_bandwidth(bandwidth)
#         log_likelihood = kde.logpdf(raster).sum()
        
#         if log_likelihood > best_score:
#             best_score = log_likelihood
#             best_bandwidth = bandwidth
    
#     return best_bandwidth
# Generate the first mode with mean -2 and standard deviation 0.5
mode1 = np.random.normal(loc=-14, scale=1, size=100)

# Generate the second mode with mean 2 and standard deviation 0.5
mode2 = np.random.normal(loc=-3, scale=0.5, size=100)

# Concatenate the two modes to create a bimodal array
bimodal_array = np.concatenate((mode1, mode2))

# Shuffle the array to randomize the order of the values
np.random.shuffle(bimodal_array)
def kde_local_minimum(raster, plot = True):
    """
    Parameters
    Fit a Kernel Density to a 1D array and find the local minimum between the
    highest peak above -10 and the highest peak below -10.
    ----------
    raster_flat : 1D array of float32
        A flattened raster.
        
    plot : bool, optional
        If True, a figure is plotted. Default is True.

    Returns
    -------
    float or np.nan.
    The value of the local minimum, or np.nan if no valid minimum is found.

    """
    # Flatten raster from 2D array to 1D
    raster_flat = raster.flatten()
    #raster_flat = raster_flat[raster_flat < -7]
    raster_flat = raster_flat[~np.isnan(raster_flat)]
    # Calculate Gaussian KDE
    kde = gaussian_kde(raster_flat)
    x = np.linspace(np.min(raster_flat), np.max(raster_flat), 1000)
    y = kde.evaluate(x)
    
    try:
    # # Find the two highest peaks in the KDE
    #     peaks, _ = find_peaks(y)
    
    # # Find the indices of the two highest peaks
    #     peak_indices = np.argsort(y[peaks])[-2:]
    #     peak_1_idx, peak_2_idx = peaks[peak_indices[0]], peaks[peak_indices[1]]
    
    # # Find the minimum value between the two peaks
    #     min_value_idx = np.argmin(y[peak_1_idx:peak_2_idx])
    #     min_value = x[peak_1_idx:peak_2_idx][min_value_idx]   
        # Find the peaks in the KDE
        peaks, _ = find_peaks(y)

        # Filter peaks based on their corresponding x values
        peaks_below_x = peaks[x[peaks] < -10]
        peaks_above_x = peaks[x[peaks] >= -10]

        # Find the index of the highest peak below x_value
        peak_below_x_idx = peaks_below_x[np.argmax(y[peaks_below_x])]

        # Find the index of the highest peak above x_value
        peak_above_x_idx = peaks_above_x[np.argmax(y[peaks_above_x])]

        # Find the minimum value between the two peaks
        min_value_idx = np.argmin(y[peak_below_x_idx:peak_above_x_idx])
        min_value = x[peak_below_x_idx:peak_above_x_idx][min_value_idx]
        if min_value > -5:
            # Filter peaks based on their corresponding x values
            peaks_below_x = peaks[x[peaks] < -10]
            peaks_above_x = peaks[x[peaks] >= -12]
    except IndexError:
        return(np.nan)
    except ValueError:
        return(np.nan)
        
    
    
    fig = plt.figure()
    plt.plot(x, y, 'r', label='KDE')
    plt.hist(raster_flat, bins=50, density=True, alpha=0.5, label='Histogram')
    #plt.plot(x[peak_1_idx:peak_2_idx][min_value_idx], min_value, 'bo', label='Minimum')
    #plt.axvline(x=min_value, color='b', linestyle='--',label='Minimum')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram and KDE')
    
    # Show the legend
    plt.legend()
    
    # Display the plot
    plt.show()
    if plot:
        return(min_value, fig)
    else:
        return(min_value)
    
def array_indicies(start_row, start_col, corner, cells):
    # NW subarray
    if corner == 0:
        start_row = int(start_row)
        end_row   = int(start_row + (cells/2))
        start_col = int(start_col)
        end_col   = int(start_col + (cells/2))
    # NE subarray
    if corner == 1:
        start_row = int(start_row)
        end_row   = int(start_row + (cells/2))
        start_col = int(start_col + (cells/2))
        end_col   = int(start_col + (cells))
    # SW subarray
    if corner == 2:
        start_row = int(start_row + (cells/2))
        end_row   = int(start_row + (cells))
        start_col = int(start_col)
        end_col   = int(start_col + (cells/2))
    # SE subarray
    if corner == 3:
        start_row = int(start_row+ (cells/2))
        end_row   = int(start_row + (cells))
        start_col = int(start_col + (cells/2))
        end_col   = int(start_col + (cells))
        
    return(start_row, end_row, start_col, end_col)



# Assuming 'raster_array' is the 2D or 3D raster array
# x is the desired size of subregions in meters
x = 400
target_shape = (20000, 20000)
# Calculate the amount of padding needed in each dimension
pad_y = target_shape[0] - raster.shape[0]
pad_x = target_shape[1] - raster.shape[1]

# Calculate the padding amounts for each side of the array
pad_y_before = pad_y // 2
pad_y_after = pad_y - pad_y_before
pad_x_before = pad_x // 2
pad_x_after = pad_x - pad_x_before

# Pad the array with empty cells
padded_array = np.pad(raster, ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)), mode='constant')

raster_height, raster_width = padded_array.shape[:2]

#subregion_height = raster_height // x
#subregion_width = raster_width // x
#subregion_height = 400
#subregion_width = 400
#i = 4
#j = 7
import copy
test_copy = copy.deepcopy(padded_array)
test_copy.fill(np.nan)

for i in range(int(raster_height /x)):
    print(str(i))
    for j in range(int(raster_width /x)):
        # Calculate the start and end indices for the subregion
        if i == 0:
            start_row_true = 0
            end_row_true = start_row_true + x
        else:
            start_row_true = i * x
            end_row_true = start_row_true + x
        if j == 0:
            start_col_true = 0
            end_col_true = start_col_true + x
        else:
            start_col_true = j * x
            end_col_true = start_col_true + x
        
        print(str(i))
        # Extract the subregion
        raster_400 = padded_array[start_row_true:end_row_true, start_col_true:end_col_true]
        start_row, end_row, start_col, end_col = array_indicies(start_row_true, start_col_true, 0, 800)
        total_elements = raster_400.size
        zero_count = total_elements - np.count_nonzero(raster_400)
        zero_percentage = (zero_count + np.count_nonzero(np.isnan(raster_400))) / total_elements
        if zero_percentage <= 0.7:
            # Bimodality test
            mdip_bm_test = mdip_bimodality_test(raster_400)
            gmm_bm_test = gmm_bimodality_test(raster_400, plot = False)
            if mdip_bm_test != gmm_bm_test:
                print(f"mdip and gmm bimodality (4000m) tests disagree on bimodality for i-index = {i} and j-index ={j}")
            if mdip_bm_test == False:
                # Split raster into 4 (2000 x 2000 meter squares)
                split_raster_200 = split4(raster = raster_400)
                n1 = 0
                cells = 400
                for raster_200 in split_raster_200:
                    if check_nan_zero_threshold(raster_200):
                        gmm_bm_test = False
                        mdip_bm_test = False
                        continue
                    else:
                        mdip_bm_test = mdip_bimodality_test(raster_200)
                        gmm_bm_test = gmm_bimodality_test(raster_200, plot = False)
                        start_row, end_row, start_col, end_col = array_indicies(start_row, start_col, n1, cells)
                    n1 += 1
                    if mdip_bm_test != gmm_bm_test:
                        print(f"mdip and gmm bimodality (2000m) tests disagree on bimodality for i-index = {i} and j-index ={j}")
                    if mdip_bm_test == False:
                        # Split raster into 4 (1000 x 1000 meter)
                        split_raster_100 = split4(raster = raster_200)
                        n2 = 0
                        cells = 200
                        for raster_100 in split_raster_100:
                            if check_nan_zero_threshold(raster_100):
                                gmm_bm_test = False
                                mdip_bm_test = False
                                continue
                            else:
                                mdip_bm_test = mdip_bimodality_test(raster_100)
                                gmm_bm_test = gmm_bimodality_test(raster_100, plot = False)
                                start_row, end_row, start_col, end_col = array_indicies(start_row, start_col, n2, cells)
                            n2 += 1
                            if mdip_bm_test != gmm_bm_test:
                                print(f"mdip and gmm bimodality (1000m) tests disa5gree on bimodality for i-index = {i} and j-index ={j}")
                            if mdip_bm_test == False:
                                # Split raster into 4 (500 x 500 meter)
                                split_raster_50 = split4(raster = raster_100)
                                n3 = 0
                                cells = 100
                                for raster_50 in split_raster_50:
                                    if check_nan_zero_threshold(raster_50):
                                        gmm_bm_test = False
                                        mdip_bm_test = False
                                        continue
                                    else:
                                        mdip_bm_test = mdip_bimodality_test(raster_50)
                                        gmm_bm_test = gmm_bimodality_test(raster_50, plot = False)
                                        start_row, end_row, start_col, end_col = array_indicies(start_row, start_col, n3, cells)
                                    n3 += 1
                                    if mdip_bm_test != gmm_bm_test:
                                        print(f"mdip and gmm bimodality (500m) tests disagree on bimodality for i-index = {i} and j-index ={j}")
                                    if mdip_bm_test == False:
                                        # Split raster into 4 (250 x 250 meter)
                                        split_raster_25 = split4(raster = raster_50)
                                        n4 = 0
                                        cells = 50
                                        for raster_25 in split_raster_25:
                                            if check_nan_zero_threshold(raster_25):
                                                gmm_bm_test = False
                                                mdip_bm_test = False
                                                continue
                                            else:
                                                mdip_bm_test = mdip_bimodality_test(raster_25)
                                                gmm_bm_test = gmm_bimodality_test(raster_25, plot = False)
                                                start_row, end_row, start_col, end_col = array_indicies(start_row, start_col, n4, cells)
                                            n4 += 1
                                            if mdip_bm_test != gmm_bm_test:
                                                print(f"mdip and gmm bimodality (250m) tests disagree on bimodality for i-index = {i} and j-index ={j}")
                                            if mdip_bm_test == False:
                                                #print("Entire raster region is unimodal.")
                                                break
                                            else:
                                                print(f"Bimodality test at 250m passed for for i-index = {i} and j-index ={j}")
                                                flood_threshold = kde_local_minimum(raster_25, plot = False)
                                                test_copy[start_row:end_row, start_col:end_col] = np.float32(flood_threshold)
                                                break
                                    else:
                                        print(f"Bimodality test at 500m passed for for i-index = {i} and j-index ={j}")
                                        flood_threshold = kde_local_minimum(raster_50, plot = False)
                                        test_copy[start_row:end_row, start_col:end_col] = np.float32(flood_threshold)
                                        break
                            else:
                                print(f"Bimodality test at 1000m passed for for i-index = {i} and j-index ={j}")
                                flood_threshold = kde_local_minimum(raster_100, plot = False)
                                test_copy[start_row:end_row, start_col:end_col] = np.float32(flood_threshold)
                                break
                    else:
                        print(f"Bimodality test at 2000m passed for for i-index = {i} and j-index ={j}")
                        flood_threshold = kde_local_minimum(raster_200, plot = False)
                        test_copy[start_row:end_row, start_col:end_col] = np.float32(flood_threshold)
                        break
            else:
                print(f"Bimodality test at 4000m passed for for j-index = {i} and i-index ={j}")
                flood_threshold = kde_local_minimum(raster_400, plot = False)
                test_copy[start_row_true:end_row_true, start_col_true:end_col_true] = np.float32(flood_threshold)
                break
        else:
            continue
        
        

ds_out = GTiff_driver.Create(r"\\netapp2p\DKmodel_users\FloodMapping\Python\Output\threshold_map.tif", test_copy.shape[1], test_copy.shape[0], 1, gdal.GDT_Float32)

ds_transform = list(ds.GetGeoTransform())
ds_transform[0] = ds_transform[0] - (np.round((test_copy.shape[1] - raster.shape[1])/2))*10
ds_transform[3] = ds_transform[3] + (np.round((test_copy.shape[0] - raster.shape[0])/2))*10
ds_transform = tuple(ds_transform)

ds_out.SetGeoTransform(ds_transform)
ds_out.SetProjection(ds.GetProjection())
ds_out.GetRasterBand(1).WriteArray(test_copy)
ds_out = None


# def threshold_otsu_custom(x , *args, **kwargs) -> float:
#     """Find the threshold value for a bimodal histogram using the Otsu method.

#     If you have a distribution that is bimodal (AKA with two peaks, with a valley
#     between them), then you can use this to find the location of that valley, that
#     splits the distribution into two.

#     From the SciKit Image threshold_otsu implementation:
#     https://github.com/scikit-image/scikit-image/blob/70fa904eee9ef370c824427798302551df57afa1/skimage/filters/thresholding.py#L312
#     """
#     counts, bin_edges = np.histogram(x, *args, **kwargs)
#     bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

#     # class probabilities for all possible thresholds
#     weight1 = np.cumsum(counts)
#     weight2 = np.cumsum(counts[::-1])[::-1]
#     # class means for all possible thresholds
#     mean1 = np.cumsum(counts * bin_centers) / weight1
#     mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

#     # Clip ends to align class 1 and class 2 variables:
#     # The last value of ``weight1``/``mean1`` should pair with zero values in
#     # ``weight2``/``mean2``, which do not exist.
#     variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

#     idx = np.argmax(variance12)
#     threshold = bin_centers[idx]
#     return threshold

