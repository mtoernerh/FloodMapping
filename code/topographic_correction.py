# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 07:56:53 2023

@author: mfth
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from shapely.geometry import Polygon, Point, LineString, MultiLineString
import geopandas as gpd
from skimage import measure
from rasterio import transform
import rasterio
import rasterio.mask
from rasterio.features import rasterize
from shapely.validation import make_valid 
from shapely.strtree import STRtree
from scipy import ndimage
from skimage.morphology import remove_small_holes
from tqdm import tqdm
#from osgeo import gdal, ogr
from utility import calculate_distances, elevation_boundary_interpolation, extract_coords_and_values, tif_to_geopandas, impute_values
from region_growing import region_growing_elevation
import warnings

def contour_polygons(array, land_mask, geotransform, contour_interval, plot = False, outpath = None):
    """
    Create elevation contours from a 2D array.

    Parameters:
    - array (numpy.ndarray): 2D array representing elevation.
    - land_mask (numpy.ndarray): Binary mask indicating land areas.
    - geotransform (tuple): Geotransform parameters.
    - contour_interval (float): Elevation interval for contour lines.
    - plot (bool): Whether to generate individual plots for each contour. Default is False.
    - outpath (str): path to where individual plots are saved. Default is none.

    Returns:
    geopandas.GeoDataFrame: GeoDataFrame containing elevation contours.
    """
    # Need to define "Agg" due memory error
    if plot:
        import matplotlib
        matplotlib.use('Agg')
    
    # Find the minimum and maximum elevation values in the DEM [meter]
    min_elevation = np.nanmin(array)
    max_elevation = np.nanmax(array)
    
    # Create an empty list to store elevation contour polygons
    elevation_contours = []
    elevation_level = []
    
    # Iterate through elevation values with the specified interval
    for elevation in np.arange(min_elevation, max_elevation + contour_interval, contour_interval):
        
        # Create contours for the current elevation value
        contours = measure.find_contours(array, level=elevation, mask = land_mask == 1)
        
        # Set to True if you want to follow process under Plots
        if plot:
            fig, ax = plt.subplots(figsize=(6, 12))
            ax.imshow(array, cmap='terrain')
            
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='black')
         
            ax.axis('image')
            ax.set_xticks([])
            ax.set_yticks([])
            elevation_print = round(elevation,1)
            ax.set_title(f"Contour elevation: {elevation_print} meters")
            file_path = f"{outpath}\frame_{elevation_print}.png"
            
            # Save the figure using FigureCanvasAgg
            canvas = FigureCanvas(fig)
            canvas.print_png(file_path)

            plt.close(fig)
        
        # Process each contour and convert to a polygon
        for contour in contours:
            try:
                contour[:, 0], contour[:, 1] = transform.xy(geotransform, contour[:, 0], contour[:, 1])
                polygon = Polygon(contour)
                elevation_contours.append(polygon)
                elevation_level.append(elevation)     
            except ValueError:
                continue
        
    # Create a GeoDataFrame from the elevation contours
    gdf_elevation = gpd.GeoDataFrame({'geometry': elevation_contours, 'elevation': elevation_level})
    return gdf_elevation

def find_matching_polygons(target_polygon, potential_candidates, gdf, MIN_AREA_COVERED):
    """
    Find polygons that match a target polygon based on intersection over union and area coverage.

    Parameters:
    - target_polygon (shapely.geometry.Polygon): Target polygon for comparison.
    - potential_candidates (list): List of indices for potential candidate polygons.
    - gdf (geopandas.GeoDataFrame): GeoDataFrame containing polygon information.
    - MIN_AREA_COVERED (float): Minimum area coverage threshold.

    Returns:
    list: List of tuples containing IoU, candidate polygon area, candidate polygon, and elevation for matching polygons.
    """
    iou_values = []
    target_area = target_polygon.area
    
    # Uncomment the following line if you intend to use clip_buffer
    clip_buffer = target_polygon.buffer(4000)

    for potential_index in potential_candidates:
        # Extract potential candidate information
        pelevation = gdf.loc[potential_index, "elevation"]
        ppolygon = gdf.loc[potential_index, "geometry"].buffer(0.1)
        ppolygon_area = ppolygon.area
        
        # Ensure the potential polygon is valid
        ppolygon = clip_buffer.intersection(ppolygon)
        ppolygon = make_valid(ppolygon)
        
        # Calculate intersection and intersection area
        polygon_intersection = target_polygon.intersection(ppolygon)
        intersection_area = polygon_intersection.area
        
        # Calculate area coverage
        area_covered = intersection_area / target_area
        
        # Must be larger than the global variable min coverage
        if area_covered > MIN_AREA_COVERED:
            # Calculation of IoU and append information
            union = target_polygon.union(ppolygon)
            iou = intersection_area / union.area
            iou_values.append((iou, ppolygon_area, ppolygon, pelevation))
    
    return iou_values

def calculate_topography_correction(target_polygon, iou_polygon, dem, iou_elevation, flood, streams, geotransform, MAX_SIZE_FACTOR):
    """
    Perform topographic correction based on the relationship between the flood and topography.

    Parameters:
    - target_polygon (shapely.geometry.Polygon): Target polygon for correction.
    - iou_polygon (shapely.geometry.Polygon): Polygon with the best IoU fit.
    - dem (numpy.ndarray): Digital Elevation Model array.
    - iou_elevation (float): Elevation corresponding to the IoU polygon.
    - flood (numpy.ndarray): Binary array representing the flooded area.
    - streams (geopandas.GeoDataFrame): GeoDataFrame containing stream information.
    - geotransform (tuple): Geotransform parameters.

    Returns:
    numpy.ndarray: Corrected array representing topography.
    """
    contour_elevation_gdf  = gpd.GeoDataFrame({'geometry': [iou_polygon]})
    contour_geom  = [shapes for shapes in contour_elevation_gdf.geometry]
    rasterized_contour  = rasterize(contour_geom,
                          out_shape=dem.shape,
                          fill=0,
                          out=None,
                          transform=geotransform,
                          all_touched=False,
                          default_value=1,
                          dtype=int)
    if MAX_SIZE_FACTOR * target_polygon.area > iou_polygon.area:
        # Reduce the iou_polygon to the clipped buffer
        clip_buffer = target_polygon.buffer((target_polygon.area / 10000) * 15)
        iou_polygon = clip_buffer.intersection(iou_polygon)
        
        # Calculate flood depth
        flood_depth = np.where(rasterized_contour  == 1, iou_elevation - dem, np.nan) # CHANGE
        # Filter out shallow areas
        flood_depth[flood_depth < 0] = np.nan # Change
        
        array_output = flood_depth
    # Follow this path if the area exceed too much, fluvial or false flooding
    else:
        """ Variable 1 for Region Growing (Distance from closest downstream vertex) """
        # Masking flooding with contour elevation polygon
        mask_false_flooding  = rasterized_contour != 1
        flood[mask_false_flooding ] = 0
        
        # # Label connected components
        labeled_array, num_features = ndimage.label(flood)
        # # Get the size of each cluster
        cluster_sizes = ndimage.sum(flood, labeled_array, range(num_features + 1))
        # # Set a threshold to filter out small clusters
        threshold = 26
        flood[np.isin(labeled_array, np.where(cluster_sizes < threshold))] = 0
        
        # Step 1: Filter streams that intersect with the target polygon
        intersects_mask = streams["geometry"].intersects(target_polygon)
        
        # If no branches intersectm then defintely a false flooding
        if not intersects_mask.any():
            array_output = np.full((dem.shape[0], dem.shape[1]), np.nan)
            return array_output
        
        intersecting_streams           = streams.loc[intersects_mask].copy()
        intersecting_streams["length"] = intersecting_streams["geometry"].length
        longest_stream                 = intersecting_streams.loc[intersecting_streams["length"].idxmax()]
        stream_line                    = longest_stream["geometry"]
        
        # Process the stream line to find the closest vertex outside the target polygon
        streamline_vertices = []
        target_buffer = target_polygon.buffer(200)
        if isinstance(stream_line, LineString):
            for vertex in reversed(stream_line.coords):
                vertex_point = Point(vertex)
                if vertex_point.within(target_buffer):
                    break
                streamline_vertices.append(vertex_point)
                
        elif isinstance(stream_line, MultiLineString): 
            for line_string in stream_line.geoms:
                for vertex in reversed(line_string.coords):
                    vertex_point = Point(vertex)
                    if vertex_point.within(target_buffer):
                        break
                    streamline_vertices.append(vertex_point)
        
        if len(streamline_vertices) == 0:
            try:
            # Remove the longest stream from the DataFrame
                intersecting_streams  = intersecting_streams[intersecting_streams["geometry"] != stream_line]
                # Find the second-longest stream
                second_longest_stream  = intersecting_streams.loc[intersecting_streams["length"].idxmax()]
                
                stream_line            = second_longest_stream["geometry"]
                
                streamline_vertices = []
                if isinstance(stream_line, LineString):
                    for vertex in reversed(stream_line.coords):
                        vertex_point = Point(vertex)
                        if vertex_point.within(target_buffer):
                            break
                        streamline_vertices.append(vertex_point)
                        
                elif isinstance(stream_line, MultiLineString): 
                    for line_string in stream_line.geoms:
                        for vertex in reversed(stream_line.coords):
                            vertex_point = Point(vertex)
                            if vertex_point.within(target_buffer):
                                break
                            streamline_vertices.append(vertex_point)
            except ValueError:
                array_output = np.full((dem.shape[0], dem.shape[1]), np.nan)
                return array_output
                
        # Convert to row col index location within grid domain
        indices_to_try = [-4, -3, -2, -1]
        for index in indices_to_try:
            try:
                row, col = rasterio.transform.rowcol(geotransform, streamline_vertices[index].x, streamline_vertices[index].y)
                break
            except IndexError:
        # Handle the exception or just continue to the next index
                array_output = np.full((dem.shape[0], dem.shape[1]), np.nan)
                return array_output
        
        # Create empty distance
        distance_array = np.full((len(dem[:,0]), len(dem[0,:])), np.nan)
        
        # Calculate distance from end point to all other cells
        distances = calculate_distances(distance_array, x = col, y = row)
        
        # Removing interior holes and islands from flood feature
        seeds = remove_small_holes(flood, 500)  
        
        # Distance seeds for region growing
        distance_seeds = np.where(seeds, distances, np.nan)
        
        """ Variable 2 for Region Growing (Flood boundary elevation) """
        # Removing interior holes and islands from flood feature
        boundary_elevation = elevation_boundary_interpolation(array = flood, dem = dem)
        if np.isnan(boundary_elevation).all():
            array_output = np.full((dem.shape[0], dem.shape[1]), np.nan)
            return array_output
        elevation_seeds    = np.where(seeds, boundary_elevation, np.nan)
        
        """ Using Variable 1 and Variable 2 for Region Growing """
        # Extracting cell coordinates and values
        RG_coords1, RG_values1 = extract_coords_and_values(elevation_seeds) 
        RG_coords2, RG_values2 = extract_coords_and_values(distance_seeds)
        
        # Size of initial flood will be compared to elevation RG size
        pre_flood_count  = np.sum(flood == 1)
        max_segmented_cells = pre_flood_count * MAX_SIZE_FACTOR
        
        # Performing actual Region Growing using gathered information
        array_output = region_growing_elevation(array1           = dem,
                                                array2           = distances,
                                                coordinates_list = RG_coords1, 
                                                thresholds1      = RG_values1, 
                                                thresholds2      = RG_values2,
                                                max_segmented_cells  = max_segmented_cells, 
                                                connectivity     = "Moore")
             
        if np.isnan(array_output).all() or (array_output == 0).all(): 
            flood_dem = seeds * dem
            array_output = boundary_elevation - flood_dem
            percentage_negative = ((np.sum(array_output[seeds] < 0)) / (np.sum(seeds))) * 100
                                   
            if percentage_negative > 30:
                array_output = np.full((dem.shape[0], dem.shape[1]), np.nan)
                return array_output
            else:
                return array_output
        
        """ Perform boundary elevation interpolation again to calculate flood depth """
        # Boundary elevation interpolation again to calculate flood depth
        boundary_elevation = elevation_boundary_interpolation(array = array_output, dem = dem)
        out_elevation = np.where(array_output == 1, dem, np.nan)
        array_output = boundary_elevation - out_elevation
    
    return array_output

def topographic_correction(filepath_flood, dem, streams, land_mask, geotransform, geoprojection, GeoTIFF = False, filepath_out = None, contour_interval = 0.1, MIN_AREA_COVERED = 0.8, MAX_SIZE_FACTOR = 4):
    """
    Perform topographic correction based on elevation contours and flood polygons.

    Parameters:
    - filepath_flood (str): Path to the raster file containing flood information.
    - filepath_out (str): Output path for the corrected raster file.
    - dem (numpy.ndarray): Digital Elevation Model array.
    - streams (geopandas.GeoDataFrame): GeoDataFrame containing stream information.
    - contour_interval (float): Elevation contour interval. Defaults to 0.1 m.
    - MIN_AREA_COVERED (float): Minimum area covered by flood polygons. Defaults to 80 %.
    - MAX_SIZE_FACTOR (float): Maximum size factor for flood polygons. Defaults to 4.

    Returns:
    Topographically corrected flooding
    """
    warnings.filterwarnings("ignore", message="Any labeled images will be returned as a boolean array. Did you mean to use a boolean array?")
    warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a slice from a DataFrame.")
    
    height, width = dem.shape[:2]
    array_output = np.full((height, width), np.nan)
    
    gdf = tif_to_geopandas(filepath_flood)
    
    # Calculate the area of each polygon and add it as a new column
    gdf['area'] = gdf.geometry.area

    """ Removing polygons that are not flooding, or too small to be worth mapping  """
    for column_name in gdf.columns:
        # Check if the column contains only binary values (0 and 1)
        is_binary = gdf[column_name].isin([0, 1, 2]).all()
        if is_binary:
            break
    for index, row in gdf.iterrows():
        # Check if the area of the current polygon is less than 2000 or not flooded polygons (gaps or holes in flooded areas)
        if row['area'] < 5000 or row[column_name] != 1:
            # If the area is less than 2000, drop the row by its index
            gdf = gdf.drop(index)
    
    # Define the elevation contour interval [meter]
    gdf_elevation     = contour_polygons(dem, land_mask, geotransform, contour_interval, plot = False)
    elevation_STRtree = STRtree(gdf_elevation["geometry"])
    
    for polygon  in tqdm(gdf.geometry,total=len(gdf), desc="Processing Flooded Areas"):
        # Finding elevation contour polygons that overlap with target bounding box
        potential_candidates = list(elevation_STRtree.query(polygon, predicate = "overlaps"))
        
        # Find the contour polygon that best describe the extent of the flooded area
        iou_values = find_matching_polygons(polygon, potential_candidates, gdf_elevation, MIN_AREA_COVERED)
        
        if len(iou_values) > 0:
            # Find the index of the polygon with highest iou
            index_max_iou = max(range(len(iou_values)), key=lambda i: iou_values[i][0])
            
            # Extract the geometry and elevation 
            iou_polygon   = iou_values[index_max_iou][2]
            iou_elevation = iou_values[index_max_iou][3]
        else:
            iou_polygon   = polygon.buffer(polygon.area * MAX_SIZE_FACTOR + 100)
            iou_elevation = -9999
            
            # Loading raster for flood feature in question # Alternatively use unique cluster IDs
        with rasterio.open(filepath_flood, "r") as src:
             # Retriving flood cluster as raster for target polygon
             flood, transform = rasterio.mask.mask(src, [polygon.buffer(10)], crop=False)
             flood = flood[0,:,:]
             geotransform  = src.transform
             geoprojection = src.crs
        
        segmentation = calculate_topography_correction(
            polygon, iou_polygon, dem, iou_elevation, flood, streams=streams, geotransform=geotransform, MAX_SIZE_FACTOR = MAX_SIZE_FACTOR)
        
        if np.isnan(segmentation).all() or (segmentation == 0).all(): 
            continue
        coords, values = extract_coords_and_values(segmentation) 
        array_output = impute_values(coords, values, array_output)

    # Saving as a GeoTiff
    if GeoTIFF:
        with rasterio.open(filepath_out, 'w', driver='GTiff', height=array_output.shape[0], width=array_output.shape[1],
                       count=1, dtype=array_output.dtype, crs=geoprojection, transform=geotransform) as dst:
        # Write the array to the new raster
            dst.write(array_output, 1)
    return array_output
