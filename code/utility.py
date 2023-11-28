import numpy as np
import os
import re
from datetime import datetime
from osgeo import gdal, ogr


def extract_datetime_from_filename(filename, date_format="%Y_%m_%d"):
    # Use regular expression to find a date pattern in the filename
    date_match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
    
    if date_match:
        date_str = date_match.group(1)
        try:
            # Parse the date string into a datetime object
            date_object = datetime.strptime(date_str, date_format)
            return date_object
        except ValueError:
            print(f"Error parsing date in filename {filename}")
    
    return None

def tif_files_and_datetime(root_folder, subfolders, file_type='VV'):
    """
   Get a sorted list of TIF files with associated datetime information from specified subfolders.

   Parameters
   ----------
   root_folder : str
       The root folder containing the subfolders.

   subfolders : list
       List of subfolders to search for TIF files.

   file_type : str, optional
       The file type to filter (e.g., 'VV' or 'VH' in the filename). Defaults to 'VV'.

   Returns
   -------
   list
       A sorted list of tuples, where each tuple contains a TIF file path and its associated datetime.
   """
    tif_files_with_datetime = {}

    for subfolder in subfolders:
        folder_path = os.path.join(root_folder, subfolder)
        
        # Use the os.listdir() method to get a list of all files in the folder
        all_files = os.listdir(folder_path)
        
        # Filter the files to only include .tif files
        tif_files = [os.path.join(folder_path, file) for file in all_files if file.endswith('.tif')]
        
        # Extract datetime from filenames and create a dictionary
        for file in tif_files:
            if file_type in file:
                filename = os.path.basename(file)
                date_object = extract_datetime_from_filename(filename)
                
                if date_object:
                    # Add the filename and datetime to the dictionary
                    tif_files_with_datetime[file] = date_object
    
    sorted_tif_files = sorted(tif_files_with_datetime.items(), key=lambda x: x[1])
    return sorted_tif_files

def check_nan_zero(data_array, threshold=0.8):
    """
  Check if the percentage of NaN and zero values in the array exceeds a specified threshold.

  Parameters
  ----------
  data_array : numpy.ndarray
      The array to be checked for NaN and zero values.

  threshold : float, optional
      The threshold for the percentage of NaN and zero values.
      Defaults to 0.8 (80%).

  Returns
  -------
  bool
      True if the percentage of NaN and zero values exceeds the threshold, False otherwise.
  """
    total_elements = data_array.size
    
    if total_elements == 0:
        return True
    else:
        zero_count          = total_elements - np.count_nonzero(data_array)
        nan_count           = np.count_nonzero(np.isnan(data_array))
        nan_zero_percentage = (nan_count + zero_count) / total_elements
        
        if nan_zero_percentage >= threshold:
            return True
        return False
    
def tif_to_shapefile(input_path, output_path, file_name):
    """
    Convert a GeoTIFF file to a shapefile using GDAL.
    Parameters
    ----------
    input_path : str
        The file path to the input GeoTIFF file.
    output_path : str
       The directory where the output shapefiles will be saved.
    file_name : str
        The base filename for the output shapefiles, with no extension.

    Returns
    -------
    None.
    The function performs the conversion and saves the shapefiles in the specified directory.

    """
    # Convert TIF to GeoJSON using gdal_polygonize
    type_mapping = { gdal.GDT_Byte: ogr.OFTInteger,
                     gdal.GDT_UInt16: ogr.OFTInteger,   
                     gdal.GDT_Int16: ogr.OFTInteger,    
                     gdal.GDT_UInt32: ogr.OFTInteger,
                     gdal.GDT_Int32: ogr.OFTInteger,
                     gdal.GDT_Float32: ogr.OFTReal,
                     gdal.GDT_Float64: ogr.OFTReal,
                     gdal.GDT_CInt16: ogr.OFTInteger,
                     gdal.GDT_CInt32: ogr.OFTInteger,
                     gdal.GDT_CFloat32: ogr.OFTReal,
                     gdal.GDT_CFloat64: ogr.OFTReal}
    ds = gdal.Open(input_path, gdal.GA_Update)
    srcband = ds.GetRasterBand(1)
    drv = ogr.GetDriverByName("ESRI Shapefile")
    
    shapefile_extensions = [".shp", ".shx", ".dbf", ".prj", ".sbn", ".sbx", ".cpg"]
    for extension in shapefile_extensions:
        filename = file_name + extension
        file_path = os.path.join(output_path, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
    dst_ds = drv.CreateDataSource(output_path)
    dst_layer = dst_ds.CreateLayer(file_name, srs = None )
    raster_field = ogr.FieldDefn('Water', type_mapping[srcband.DataType])
    dst_layer.CreateField(raster_field)
    gdal.Polygonize( srcband, None, dst_layer, 0, [], callback=None)
    
def find_bounding_box(x1, y1, x2, y2, size):
    """
   Adjust the coordinates of a bounding box to make its width and height divisible by a specified size.

   Parameters
   ----------
   x1 : float
       The x-coordinate of the lower-left corner of the initial bounding box.

   y1 : float
       The y-coordinate of the lower-left corner of the initial bounding box.

   x2 : float
       The x-coordinate of the upper-right corner of the initial bounding box.

   y2 : float
       The y-coordinate of the upper-right corner of the initial bounding box.

   size : int
       The desired size for the adjusted bounding box. Width and height will be adjusted to be divisible by this size.

   Returns
   -------
   tuple
       A tuple containing the adjusted coordinates (new_x1, new_y1, new_x2, new_y2) of the bounding box.

   Example
   -------
   >>> find_bounding_box(0, 0, 15, 20, 5)
   (0, 0, 20, 25)
   """
    # Calculate the width and height of the initial box
    width = x2 - x1
    height = y2 - y1

    # Calculate the new width and height that are divisible by 25 and even
    new_width = width + size - (width % size) if width % size != 0 else width
    new_height = height + size - (height % size) if height % size != 0 else height

    # Calculate the adjustments needed for the starting coordinates
    x_adjustment = (new_width - width) // 2
    y_adjustment = (new_height - height) // 2

    # Adjust the starting and ending coordinates
    new_x1 = x1 - x_adjustment
    new_y1 = y1 - y_adjustment
    new_x2 = x1 + new_width
    new_y2 = y1 + new_height

    return (new_x1, new_y1, new_x2, new_y2)

def extract_coords_and_values(array):
    """
    Extract the coordinates and corresponding values of non-zero and non-NaN elements in a 2D array.

    Parameters
    ----------
    data_array : np.ndarray
        A 2D NumPy array from which coordinates and values will be extracted.

    Returns
    -------
    tuple
        A tuple containing:
        - A list of tuples representing the coordinates of non-zero and non-NaN elements.
        - A list of corresponding values for each coordinate.

    Example
    -------
    >>> import numpy as np
    >>> data = np.array([[1, 0, 3], [0, np.nan, 5], [7, 8, 0]])
    >>> extract_nonzero_nonnan_coordinates_values(data)
    ([(0, 0), (0, 2), (1, 2), (2, 0), (2, 1)], [1, 3, 5, 7, 8])
    """
    # Find the indices (coordinates) of non-zero and non-NaN elements in the array
    coordinates = np.argwhere(np.logical_and(array != 0, ~np.isnan(array)))

    # Get the corresponding values for each coordinate
    values = [array[coord[0], coord[1]] for coord in coordinates]

    # Convert the coordinates to a list of tuples if needed
    coordinates_list = [tuple(coord) for coord in coordinates]

    return coordinates_list, values

def impute_values(source_coords, source_values, target_array):
    """
   Impute values from the source_values into the target_array based on source_coords.

   Parameters
   ----------
   source_coords : list of tuples
       List of index pairs (row, col) from the source array.

   source_values : list
       List of values to be imputed.

   target_array : numpy.ndarray
       The array where values will be imputed.

   Returns
   -------
   numpy.ndarray
       The updated target array with imputed values.
   """
    for (row, col), value in zip(source_coords, source_values):
        target_array[row, col] = value
        
    return target_array

def calculate_tpi(elevation_neighborhood):
    """
   Calculate the Topographic Position Index (TPI) for a given elevation neighborhood.

   Parameters
   ----------
   elevation_neighborhood : numpy.ndarray
       1D array representing the elevation values in the neighborhood.

   Returns
   -------
   float
       The Topographic Position Index (TPI) for the neighborhood.
   """
    central_elevation  = elevation_neighborhood[len(elevation_neighborhood) // 2]
    mean_elevation = np.nanmean(elevation_neighborhood)
    tpi = central_elevation  - mean_elevation
    return tpi