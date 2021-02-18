import numpy as np
import rasterio
import glob, os
import os.path  
import skimage
import skimage.morphology
import skimage.measure
import pandas as pd
import geopandas
from shapely.geometry import Polygon
from itertools import compress

def load_bands(path,bands):
    """
    load_bands(path,bands,clip=True,cliplow=0.1,cliphigh=0.9,Driver='JP2OpenJPEG'):
    
    This function loads images of type .jp2 from a IMG_DATA folder definded from "path".
    The images are loaded as numpy arrays(NxM) in a list of size "bands". 
     
    Supported for L2A products
    object cant be returned for L1C products
    
    Version 1.2
    ================================================
    Input:
        path(str) = path img IMG_DATA folder.
        bands(list of str) = A list of strings containing the band names.
    
    Output: 
        img_container(list) = A list of size (bands) containing each band as a numpy array image of same size.
        rasterio_obj(list) = A list of rasterio objects. Usefull for coordinate transformation. 
                            !!! Only works frot R10m rest is returned as None "                          
    
    Example:
        get list of images:
        images = load_bands('F:\\S2_billeder\\..\\IMG_DATA',bands=['B03','B04','B05','B01'])
    
    Author:
        Krisitan Sørensen
        Marts 2019  
        Revised: June 2019
    
    """
    ####################################
    # Assertions
    ####################################    
    assert len(bands)>0,"No bands are given"
    
    Driver='JP2OpenJPEG'
    
    #count number og .jp2 files in .IMG folder. If there is some, if old format
    # if none, then theres subfolders and new format.
    folder = glob.glob(os.path.join(path,"*.jp2"))
    if (len(folder)>0):
        data_type = 'L1C'
        print('(Load_bands message) Image is L1C format.')
    else:
        data_type = 'L2A'
        print('(Load_bands message) Image is L2A format.')
        assert os.path.exists(os.path.join(path,'R10m')),"band folders doesn't exist in path folder"
        
    
    
    ####################################
    #Loading the bands
    ####################################
    
    img_container = []
    rasterio_obj = []
    #in data_type='new', the images are found in different subfolders.
    if (data_type =='L2A'):
        for i in range(len(bands)):
            #For bands B02,B03,B04 and B08, chaning path to R10m(10 m resolutuon)
            if (bands[i]=='B02') or (bands[i]=='B03') or (bands[i]=='B04')  or (bands[i]=='B08') or (bands[i]=='AOT') or (bands[i]=='TCI') or (bands[i]=='WVP'):    
                img_path = os.path.join(path,'R10m')
                img_path = os.path.join(img_path,'')
                img_extend = bands[i]+"_10m.jp2"
                for file in os.listdir(img_path):
                    if file.endswith(img_extend):
                        img = rasterio.open(img_path+file, driver=Driver)
                        rasterio_obj.append(img)
                        img_container.append((np.asarray(img.read(window=((0,img.shape[0]),(0,img.shape[1])))))[0,:,:])
                                   
            if (bands[i]=='B05') or (bands[i]=='B06') or (bands[i]=='B07')  or (bands[i]=='B11') or (bands[i]=='B12') or (bands[i]=='SCL'):
                img_path = os.path.join(path,'R20m')
                img_path = os.path.join(img_path,'')
                img_extend = bands[i]+"_20m.jp2"
                for file in os.listdir(img_path):
                    if file.endswith(img_extend):
                        img = rasterio.open(img_path+file, driver=Driver)
                        rasterio_obj.append(img)
                        img = (np.asarray(img.read(window=((0,img.shape[0]),(0,img.shape[1])))))[0,:,:]
                        img = np.repeat(img, 2, axis=0)
                        img = np.repeat(img, 2, axis=1)
                        img = img.astype('float')
                        img_container.append(img)    
                        
            # 
            if (bands[i]=='B01') or (bands[i]=='B09'):
                # Aerosol Optical Thickness AOT. 
                img_path = os.path.join(path,'R60m')
                img_path = os.path.join(img_path,'')
                img_extend = bands[i]+"_60m.jp2"
                for file in os.listdir(img_path):
                    if file.endswith(img_extend):
                        img = rasterio.open(img_path+file, driver=Driver)
                        rasterio_obj.append(img)
                        img = (np.asarray(img.read(window=((0,img.shape[0]),(0,img.shape[1])))))[0,:,:]
                        img = np.repeat(img, 6, axis=0)
                        img = np.repeat(img, 6, axis=1)
                        img = img.astype('float')
                        img_container.append(img)   
    
    #if data_type ='old', all .jp2 files is in .IMG folder.
    elif (data_type=='old'):
        for i in range(len(folder)):
            for j in range(len(bands)):
                if folder[i].endswith(bands[j]+'.jp2'):
                    img = rasterio.open(folder[i], driver=Driver)
                    rasterio_obj.append(None)
                    img = (np.asarray(img.read(window=((0,img.shape[0]),(0,img.shape[1])))))[0,:,:]
                    
                    if (bands[j]=='B05') or (bands[j]=='B06') or (bands[j]=='B07')  or (bands[j]=='B11') or (bands[j]=='B12') or (bands[j]=='SCL'):
                        img = np.repeat(img, 2, axis=1)
                        img = np.repeat(img, 2, axis=0)
                    elif (bands[j]=='B01') or (bands[j]=='B09') or (bands[j]=='B10'):
                        img = np.repeat(img, 6, axis=1)
                        img = np.repeat(img, 6, axis=0)                     
                    img_container.append(img) 
        
        
    return img_container, rasterio_obj




def contours(img,mode='ndvi',method='skimage'):
    """
    # contours(img,mode='ndvi',method='skimage')
    # This function finds contours in an image(numpy array).
    # The function takes an image(NxM) of type (method). 
    # By applying the specific thresholds (found from method)
    # A binary image is found for each contour type.
    # This binary image is the used to find the coordinates for each contour.
    # 
    # The different contour types, saved in (Contours_list) 
    # are then saved in (Contours) .
    #
    # NOTE:
    #   - Ndvi - bare soil is not added. 
    #   - Still a bit too rough with the morphology and thresholds.
    #
    # ================================================
    # Input:
    #     img(NxMx1) = numpy array of image. 
    #     mode(str) = string of type of image. 
    #     method(str) = String of type of countour finding. Currently only skimage.
    #
    # Output: 
    #     Contours(list) = list contining arrays of contours.
    #                      Each array is (Nx2) with each N consisting of 
    #                      (row, column)  coordinates along the contour.
    #
    #     Contours_type(list) = List with the names of contours.
    #                           
    # Example:
    #     contours, contours_types = contours(ndvi_image,mode='ndvi',method='skimage')
    #     - returning healthy plants and weak plants  contours in arrays.
    #   
    """
    
    assert mode == 'ndvi', "so far, only ndvi is supported."
    assert method == 'skimage', "so far, only skimage contour finding is supported."
    
    #initializing list.
    contours = []
    contours_type = []
    
    #thresholding image based on mode.
    if mode =='ndvi':
        healthy_plants = img>0.6
        kinda_healthy = img>0.4
        weak_plant = img>0.25
        #no_vegetation = img<0.2
        kernel = np.ones((2,2),np.uint8)
        kernel_2 = np.ones((3,3),np.uint8)
        #laver lige nogle opratatioer for sjov skyld da.
        healthy_plants = skimage.morphology.binary_closing(healthy_plants,kernel)
        healthy_plants = skimage.morphology.binary_opening(healthy_plants,kernel_2).astype(float)
        healthy_temp = np.absolute(healthy_plants-1)
        #
        kinda_healthy = skimage.morphology.binary_closing(kinda_healthy,kernel)
        kinda_healthy = skimage.morphology.binary_opening(kinda_healthy,kernel_2).astype(float)
        kinda_healthy_temp = np.absolute(kinda_healthy-1)
        #
        weak_plant = skimage.morphology.binary_closing(weak_plant,kernel)
        weak_plant = skimage.morphology.binary_opening(weak_plant,kernel).astype(float)
        weak_plant = weak_plant*healthy_temp*kinda_healthy_temp
        #
        #no_vegetation = skimage.morphology.binary_closing(no_vegetation,kernel)
        #no_vegetation = skimage.morphology.binary_opening(no_vegetation,kernel).astype(float)
        
        
        
        if method =='skimage':
            cc_healthy = skimage.measure.find_contours(healthy_plants, 0.8)
            cc_kinda_healthy = skimage.measure.find_contours(kinda_healthy, 0.8)
            cc_weak = skimage.measure.find_contours(weak_plant, 0.8)
            #cc_bare = skimage.measure.find_contours(no_vegetation, 0.8)
            
        
        contours.append(cc_healthy)
        contours.append(cc_kinda_healthy)
        contours.append(cc_weak)
        #contours.append(no_vegetation)
        
        contours_type.append('Healthy vegetation')
        contours_type.append('Moderate vegetation')
        contours_type.append('Weak vegetation')
        #contours_type.append('no vegetation')
            
    return contours, contours_type 


def contour2poly(contours,tolerance=1):
    """
    # contour2poly(contours,tolerance=1)
    # This function takes a list of contours(each of different kind)
    # and then approximate the contour into a polygon.
    # the sharpness of the polygon is defined by (tolerance).
    # A tolerance=0 gives the original contours.
    #
    # ================================================
    # Input:
    #     contours(list) = A list of contours.
    #                      Each element in the list is (Nx2).
    #                      Each N is (row, column) of coordinates.
    #     tolerance(float) = Value of sharpness. 0 = contours.
    #                        A high value = less edges.
    #
    # Output: 
    #     polygons(list) = 
    #                           
    # Example:
    #     poly =  contour2poly(contours)
    #     poly =  contour2poly(contours,tolerance=2) # more sharp
    #     poly =  contour2poly(contours,tolerance=0.1) # more detailed
    #
    # Author:
    #     Krisitan Sørensen
    #     April 2019  
    #
    # NOTE:
    #    - Must make tolerance based on size of image to make polygons prettyyyy
    """
    polygons = []

    for i in range(len(contours)):
        temp = []
        for j in range(len(contours[i])):
            temp.append(skimage.measure.approximate_polygon(contours[i][j],tolerance))
        
        polygons.append(temp)
        
    return polygons




def rgb_img(r,g,b):
    """
    rgb color representation.
    just illustration.. 
    (NxMx3) -> (NxMx1)
    """
    
    #color_image = np.stack((r*100,(g*42), (b*10)), axis=2)
    color_image = np.stack((r,g, b), axis=2)
    color_image = color_image/np.quantile(color_image,0.99)
    
    return color_image


def img_clip(img,cliphigh=0.9,cliplow=0.1):
    """
    Just at simple clipping used for our stuff.
    Maybe use CFAR later? (not)
    
    """
    for i in range(len(img)):
        img[i] = np.clip(img[i], np.quantile(img[i],cliplow), np.quantile(img[i],cliphigh))
        
    img_clipped = img
    
    return img_clipped

import numpy as np
import cv2


def masking_scl(SCL_map,masks):
    """
    This function is used for various masks for the Sentinel L2A data product.
    The following masks can be applied:
    - Clouds
    - Vegetation
    - Non vegetated land areas
    - Water
    - Snow
    - Other
    The function used the L2A product Scene Classification ("SCL")
    
    ================================================
    input:
        SCL_map[NxM] = SCL granule as array from the ESA L2A product. 
        masks[str] = list of masks wanted in string format
                     Possible masks:
                     - ["cloud"]
                     - ["water"]
                     - ["vegetation"]
                     - ["non_vegetation"]
    output:
        mask[NxM] = The mask wanted. If several masks[str] is given, this is a combination of all.
                    Note. The SCL is from R20, so if this is to be used with R10, an up-or downsample must be made.
        
    Example:
        masking_scl(SCL_band,masks=["cloud"])
              returns a cloud mask
        masking_scl(SCL_band,masks=["cloud","water"])
              returns one mask, masking out both clouds and water.
              
              
    Author:
        Kristian Aalling Soerensen
        kaas@yoeo.dk
    
    Disclaimer: 
         This code comes with no guarantee at all and its author is not liable for any damage that its utilization may cause.
         
        
    -------------- More detailed description ----------------
    The easiest, but most non customisable was of perfoming classification is by using ESAS L2A product, SCL
    see: 
    https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
    or 
    https://sentinel.esa.int/documents/247904/685211/Sentinel-2_User_Handbook
    This can, e.g be used to masks clouds
    "The scene classification algorithm allows detection of clouds, snow and cloud shadows and
    generation of a classification map, which consists of four different classes for clouds
    (including cirrus), together with six different classifications for shadows, cloud shadows,
    vegetation, soils/deserts, water and snow." (p.49 of user handbook.)
    ---------------------------------------------------------
         
    """
    print("(Warning: masking_scl). Only usable for L2A products")
    classification = np.copy(SCL_map);
    
    mask=np.zeros((np.size(SCL_map[0]),np.size(SCL_map[1])))
    #Note the non-wanted pixels are assigned 1. This enables a dilation of the mask.
    # Emperical, visual examination of the ESA masks, shows they under assign to ensure no false positives. 
    # Therefor, a dilation is made to better mask out the un-wanted, low-probability nabouring pixels. 
    
    #thereafter, non-wanted pixels are assigned 0, and wanted 1.
    for i in range(len(masks)):
        if masks[i]=="cloud":
            mask[classification==3]=1 #cloud shadows
            mask[classification==8]=1 #cloud medium probability
            mask[classification==9]=1 # cloud high probability
            mask[classification==10]=1 #thing_cirrus  
            
        elif masks[i]=="vegetation":
            mask[classification==4]=1 #vegatation
        elif masks[i]=="water":
            mask[classification==6]=1 # water
        elif masks[i]=="non_vegetation":
            mask[classification==5]=1 # water
            
    kernel = np.ones((40,40),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    #Change 0 to 1, and 0 to 1...
    mask = abs(mask-1)        

    return mask
