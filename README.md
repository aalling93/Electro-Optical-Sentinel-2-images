# Sentinel-2 toolbox

Work with sentinel-2 images easily. If you know how download Sentinel-2 Multi-Spectral-Images (or Electro-Optical), but you don't know how to work with them in Python, I have it all implemented here for you.


## Content
- Functions
- Illustration of functions 
- Demonstration on usage


## Functions:
- load_bands()
- contours()
- contour2poly()
- ndvi_index()
- rvi_index()
- savi_index()
- evi_index()
- rgb_img()
- 
## Demonstration on usage:
```
images = load_bands('F:\\S2_billeder\\..\\IMG_DATA',bands=['B03','B04','B05','B01'])
```
This function will load the bands of your chosing into python. You **could** load them all, but it is recommended to only load the bands that are needed for later analyses, as they take a lot of memory (i.e. calcuating a simple NDVI index.)

```
contours, contours_types = contours(ndvi_image,mode='ndvi',method='skimage')
```
This function finds contours in a false colours image. This is useful when working with e.g. Index values(NDVI etc.). 

```
poly =  contour2poly(contours,tolerance=2)
```
This function creates geo-references polygons. Often, when working with e.g. satellite images, the results must be displyed in a GIS software, e.g. QGIS. The results from an analysis must consequently be turned into polygons that can be exported into e.g. .shape files.
Thus, calculate an index product, and thereafter find the contours for different resut, and lastly find the polygons. This can then be used in QGIS.

#### INDEX PRODUCTS 
Index products are simple products calculated using band-operations.
```
ndvi_index = ndvi(B04, B08)

```
normalized difference vegetation index(NDVI).

# Getting started

### Prerequisites

### Installing
* pip install pyshp



# Authors


