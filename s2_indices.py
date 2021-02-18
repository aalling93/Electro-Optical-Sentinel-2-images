import numpy as np

def ndvi_index(red, nir):
    """
    # ndvi_index(red, nir):
    #
    # This function takes a red image and a NIR image as numpyt arrays
    # and returns a normalized difference vegetation index(NDVI) image from them
    # 
    # ================================================
    # Input:
    #     red(NxM) = red image
    #     nir(NxM) = NIR image
    # Output:
    #     NDVI(NxM) = NDVI image
    #
    # Example:
    #     ndvi_index = ndvi(B04, B08)
    #
    """
    assert len(nir)==len(red),"red and nir must be same size"
    assert type(nir) == np.ndarray,"nir image must be numpy array"
    assert type(red) == np.ndarray,"red image must be numpy array"
    nir = nir.astype('float')
    red = red.astype('float')
    NDVI = ((nir - red)/(nir + red))

    return NDVI




def rvi_index(r, nir):
    """
    # rvi_index(r, nir):
    #
    # This function takes a red image and a NIR image as numpy arrays
    # and returns a Ratio Vegetation Index (RVI) image from them
    # 
    # ================================================
    # Input:
    #     r(NxM) = red image
    #     nir(NxM) = NIR image
    # Output:
    #     rvi(NxM) = rvi image
    #
    # Example:
    #     rvi = rvi(B04, B08)
    #
    """
    
    rvi = nir/r
    
    return rvi



def savi_index(r, nir,L=0.5):
    """
    # savi_index(r, nir):
    #
    # This function takes a red image and a NIR image as numpy arrays
    # and returns a Soil Adjusted Vegetation Index (savi) image from them
    # 
    # ================================================
    # Input:
    #     r(NxM) = red image
    #     nir(NxM) = NIR image
    #     L(float) = canopy background adjustment factor
    # Output:
    #     savi(NxM) = savi image
    #
    # Example:
    #     savi = savi(B04, B08)
    #
    """
    
    savi = ((1+L)*(nir-r))/(nir+r+L)
    
    return savi



def evi_index(r, nir,b,G=2.5,L=0.5,C1=6,C2=7.5):
    """
    # evi_index(r, nir,b,L=0.5,C1,C2):
    #
    # This function takes a red image and a NIR image as numpy arrays
    # and returns a enhanced vegetation index (evi) image from them
    # 
    # G, L, C1, C2 are adopted from MODIS
    # ================================================
    # Input:
    #     r(NxM) = red image
    #     nir(NxM) = NIR image
    #     b(NxM) = blue image
    #     G = Gain
    #     L(float) = canopy background adjustment factor
    #     C1 = aerosol resistance term 1
    #     C2 = aerosol resistance term 2
    # Output:
    #     evi(NxM) = evi image
    #
    # Example:
    #     
    #
    """
    
    evi = G * ((nir-r)/(nir+C1*r-C2*b*L))
    
    return evi

