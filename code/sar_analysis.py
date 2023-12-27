# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:08:36 2023

@author: mfth
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks        # Not used if crossing point
from skimage.filters import threshold_otsu # Not used if crossing point
from scipy.optimize import brentq


from utility import check_nan_zero

# Gaussian Mixture Model Bimodality class
class GMMBimodalityTester:
    def __init__(self, plot=True):
        self.plot = plot
        self.gmm_model = None
    
    # Fitting Two-component Gaussian Mixture Model (GMM) to array
    def fit(self, array):
        array_flat = array.flatten()
        array_flat = array_flat[~np.isnan(array_flat)]
        
        # Checking if array contains many nan or zero values
        if check_nan_zero(array_flat) or len(array_flat) < 10:
            self.gmm_model = None
            return False

        self.gmm_model = GaussianMixture(n_components=2, random_state=0, covariance_type='full').fit(array_flat.reshape(-1, 1))
        return True
    
    # Extract component N1 and N2
    def _get_component_assignments(self, array_flat):
        if self.gmm_model is None:
            return False
        probabilities = self.gmm_model.predict_proba(array_flat.reshape(-1, 1))
        component_assignments = np.argmax(probabilities, axis=1) + 1
        return component_assignments
    
    # Calculating mean and standard deviation for each component
    def _get_stats(self):
        if self.gmm_model is None:
            return False
        n1mean, n2mean = self.gmm_model.means_[:, 0]
        n1std, n2std = np.sqrt(self.gmm_model.covariances_[:, 0, 0])
        return n2mean, n1mean, n2std, n1std
    
    # Testing bimodaility using Ashman D (AD) and Between-Class-Variance (BCV)
    def test_bimodality(self, array):
        # Placeholder to return result with False, meaning bimodal distribution could not be determined
        n1, n2 = [1, 1], [1, 1]
        result = (False, n1, n2)
        if self.gmm_model is None:
            return result
        
        # Should probably use self here since this is the not the firs I perform these actions
        array_flat = array.flatten()
        array_flat = array_flat[~np.isnan(array_flat)]
        
        if len(array_flat) <= 15: #or np.min(array_flat) > -15: # CHANGE
            return result
        
        # Calculating statistics mean and std of component N1 and N2
        component_assignments = self._get_component_assignments(array_flat)
        n1mean, n2mean, n1std, n2std = self._get_stats()

        n2std_range1, n2std_range2 = n2mean - 3 * n2std, n2mean + 3 * n2std
        n1std_range1, n1std_range2 = n1mean - 3 * n1std, n1mean + 3 * n1std

        n1percent = np.count_nonzero(component_assignments == 1) / len(component_assignments)
        n2percent = np.count_nonzero(component_assignments == 2) / len(component_assignments)
        
        # n1 and n2 component from fitted gaussian distribution
        n1 = array_flat[component_assignments == 1]
        n2 = array_flat[component_assignments == 2]
        
        # Calculting dynamic bin width, only used when plotting
        p25, p75 = np.percentile(array_flat, [25, 75])
        IQR = p75 - p25
        W = int(2 * IQR * len(array_flat) ** (1 / 3))
        
        # Calculating Ashman D
        #numerator = np.abs(n1mean - n2mean)
        #denominator = np.sqrt((n1std ** 2 + n2std ** 2))
        #ashman_d = (2 ** 0.5) * (numerator / denominator)
        # If Ashman D coefficient is smaller than 2, distribution is unimodal, return False # CHANGE
       # if ashman_d < 2:
       #     return result
        
        # Calculate normalized BCV
        #######
        total_variance = np.var(array_flat)
        w1 = (len(n1)/len(array_flat))
        w2 = (len(n2)/len(array_flat))
        between_class_var = w1*w2*(n1mean-n2mean)**2
        B_t = between_class_var/total_variance
        #######
        
        # If  normalized BCV (B(t)) is smaller than 0.55, distribution is unimodal, return False
        if B_t < 0.55:
            return result
        
        # If either component only comprise less than 10 % of the current array, return False
        if any(percent < 0.1 for percent in (n1percent, n2percent)):
                return result

        if self.plot:
            if self.gmm_model is None:
                return result
            plt.axvline(x=n2mean, color='black', linestyle='--',label='n-2 mean', zorder=3)
            plt.axvline(x=n1mean, color='black', linestyle='--',label='n-1 mean', zorder=3)
            plt.axvspan(n2std_range1, n2std_range2, facecolor='#377eb8', alpha=0.2, zorder=1) #e41a1c
            plt.axvspan(n1std_range1, n1std_range2, facecolor='#e41a1c', alpha=0.2, zorder=1)
            plt.hist([n1, n2], bins=W, color=['#377eb8', '#e41a1c'], alpha=1, label=['n1', 'n2'], rwidth=2, zorder=2)
            plt.xlabel('Sentinel1 SAR [dB]')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
        # If function makes it this far the distribution is bimodal and True is returned along with N1 and N2 1D arrays
        result = (True, n1, n2)
        return result

# Class for fitting KDE to N1 and N2 and finding the local threshold
class KDELocalMinimumFinder:
    def __init__(self, plot = True):
        self.kde = None
        self.x = None
        self.y = None
        self.array_flat = None
        self.W = None
        self.plot = plot
    # Fitting KDE to whole array (actually redundant if crossing point method is used) #to be fixed#
    def fit(self, array):
        array_flat = array.flatten()
        array_flat = array_flat[~np.isnan(array_flat)]
        self.array_flat = array_flat
        self.kde = gaussian_kde(array_flat)
        self.x = np.linspace(np.min(array_flat), np.max(array_flat), 1000)
        self.y = self.kde.evaluate(self.x)
        
        p25, p75 = np.percentile(array_flat, [25, 75])
        IQR = p75 - p25
        self.W = int(2 * IQR * len(array_flat) ** (1 / 3))
    # Function for finding local threshold between the two classes n1 and n2
    def find_local_threshold(self, global_threshold, n1, n2, method = 'crossing_point'):
        if method == 'lm':
            naive_threshold = threshold_otsu(self.array_flat)
            
            peaks, _ = find_peaks(self.y)
            peaks_below_x = peaks[self.x[peaks] < naive_threshold]
            peaks_above_x = peaks[self.x[peaks] >= naive_threshold]
            
            if len(peaks_below_x) == 0 or len(peaks_above_x) == 0:
                return np.nan, np.nan, np.nan
            
            peak_below_x_idx = peaks_below_x[np.argmax(self.y[peaks_below_x])]
            peak_above_x_idx = peaks_above_x[np.argmax(self.y[peaks_above_x])]
    
            min_value_idx = np.argmin(self.y[peak_below_x_idx:peak_above_x_idx])
            local_threshold = self.x[peak_below_x_idx:peak_above_x_idx][min_value_idx]
    
            if local_threshold < self.x[peak_below_x_idx] or not (local_threshold < global_threshold):
                return np.nan, np.nan, np.nan
        if method == 'crossing_point':
            # Create KDEs for both N1 and N2
            try:
                n1kde = gaussian_kde(n1)
                n2kde = gaussian_kde(n2)
            except np.linalg.LinAlgError:
                return np.nan, np.nan, np.nan

            # Find the range of x values for the plot
            x_range = np.linspace(min(n1.min(), n2.min()), max(n1.max(), n2.max()), 1000)

            # Shape KDE to match input values, otherwise they will appear with the same density
            n1kdeCurve = n1kde(x_range)*n1.shape[0]
            n2kdeCurve = n2kde(x_range)*n2.shape[0]
            
            # Find crossing point
            try:
                local_threshold = brentq(lambda x: n1kdeCurve[int((x - x_range.min()) / (x_range[1] - x_range[0]))] - n2kdeCurve[int((x - x_range.min()) / (x_range[1] - x_range[0]))], x_range.min(), x_range.max())
            except ValueError:
                naive_threshold = threshold_otsu(self.array_flat)
                
                peaks, _ = find_peaks(self.y)
                peaks_below_x = peaks[self.x[peaks] < naive_threshold]
                peaks_above_x = peaks[self.x[peaks] >= naive_threshold]
                
                if len(peaks_below_x) == 0 or len(peaks_above_x) == 0:
                    return np.nan, np.nan, np.nan
                
                peak_below_x_idx = peaks_below_x[np.argmax(self.y[peaks_below_x])]
                peak_above_x_idx = peaks_above_x[np.argmax(self.y[peaks_above_x])]
        
                min_value_idx = np.argmin(self.y[peak_below_x_idx:peak_above_x_idx])
                local_threshold = self.x[peak_below_x_idx:peak_above_x_idx][min_value_idx]
            # Find upper and lower intervals
            n1_crop = x_range[n1kdeCurve > 1]
            n2_crop = x_range[n2kdeCurve > 1]
            if np.mean(n1) > np.mean(n2):
                try:
                    lower_threshold = np.min(n1_crop)
                    upper_threshold = np.max(n2_crop)
                except ValueError:
                    lower_threshold = np.nan
                    upper_threshold = np.nan
            else:
                try:
                    upper_threshold = np.max(n1_crop)
                    lower_threshold = np.min(n2_crop)
                except ValueError:
                    lower_threshold = np.nan
                    upper_threshold = np.nan
            # Final check to see if the local threshold is below the global upper water threshold
            if local_threshold <= global_threshold: # CHANGE back to "<"
                if self.plot:
                    fig = plt.figure()
                    plt.fill_between(x_range, 0, n1kdeCurve, color='r', alpha=0.4, label='N1-kde')
                    plt.fill_between(x_range, 0, n2kdeCurve, color='b', alpha=0.4, label='N2-kde')
                    plt.axvline(x=local_threshold, color='b', linestyle='--', label='Crossing point')
                    plt.axvline(x=upper_threshold, color='black', linestyle='dotted')
                    plt.axvline(x=lower_threshold, color='black', linestyle='dotted')
                    plt.xlabel('Sentinel1 SAR (dB)')
                    plt.ylabel('Density')
                    plt.title('Histogram and KDE')
                    plt.legend()
                    plt.show()
                # Thhe local threshold is returned
                return lower_threshold, local_threshold, upper_threshold
            else:
                return np.nan, np.nan, np.nan
        return lower_threshold, local_threshold, upper_threshold
    
