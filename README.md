# Fast morphological operations for binary images

This package provides several morphological operation functions optimized for
binary (true/false) images in 1D, 2D, and 3D. The functions implemented are
* Erosion: Remove perimeter pixels using a structuring element
* Dilation: Add pixels on the perimeter using a structuring element
* Opening: Erode image and then dilate the result
* Closing: Dilate image and then erode the result

The dilation function is implemented using the perimeter of the original image
to avoid dilating internal pixels. Using the other functions are implemented
by taking advantage of the dilation function.

[GitHub](https://github.com/shoheiogawa/binmorphopy)
