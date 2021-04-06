# Antarctic TMA
This repository contains the source code for the Antarctic TMA project:<br>
Historical aerial images from the Antarctia are used to create a 3D model based on photogrammetric methods.

## How To
For each image the exact coordinates (for x, y, z) and the direction, in which the image was taken, are known. Furthermore the internal camera parameters (e. g. focal lenght) are know. Images are not only taken in vertical but also in oblique direction. 

With this information it is possible to estimate three-dimensional structures with a technique called [Structure from motion](https://en.wikipedia.org/wiki/Structure_from_motion). The reconstruction tool that will be used is [MicMac](https://micmac.ensg.eu/index.php/Accueil).

## Why
The Antarctic Peninsula is one of the fastest changing regions on our planet, and experienced a warming unparalleled anywhere else in the Southern Hemisphere in the late-20th century. The created models will be compared to present-day elevation data, for example from satellite stereophotogrammetry and altimetry (ICESat-2), to obtain a detailed picture of elevation and mass changes over the past 50 years. Furthermore, a regional climate model will be employed to put the observed changes in a broader climatological perspective.  

## Data
The aerial images are part of the "Antarctic Single Frame Records", a collection of aerial photographs made by the US Navy for mapping purposes in between 1946 and 2000. 
It includes black-and-white, natural color and color infrared images with a photographic scale ranging from 1:1,000 to 1:64,000. More information can be found at the [USGS website](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-antarctic-single-frame-records?qt-science_center_objects=0#qt-science_center_objects). The data was digitized by the University of Minnesota and is publicly [available](https://www.pgc.umn.edu/data/aerial/).

<p align="center">
<img src="https://github.com/fdahle/Antarctic_TMA/blob/main/example_image.PNG" width="250" height="250"/><br>
<span>Exemplaric image from the dataset</span>
</p>
