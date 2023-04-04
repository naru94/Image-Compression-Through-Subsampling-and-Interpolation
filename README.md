# Image Compression Through Subsampling and Interpolation

## Introduction

In this project, we will learn how to compress image data by compromising the chrominance details since the human eye is less sensitive to it. This is achieved by transforming the RGB image to YCbCr and then subsampling the Cb and Cr components. We will use interpolation to reconstruct the compressed image. We will also examine the compression ratio achieved by this algorithm in terms of file size.

## Prerequisites

Before starting the project, make sure that you have the following packages installed in your system:

* NumPy: This library is used for mathematical functionality.
* Matplotlib: This library is used for visualization purposes. We will use the following packages:
  * pyplot: This module is used to create inline plots.
  * image: This module is used to access images and image files.
* OS: This library is used in this project to retrieve file sizes from the OS.

## Dataset
We will work with an RGB image of a cat. This image is stored as chelsea.ppm in the /dataset directory. In an RGB image, each pixel has three components corresponding to primary colors (red, green, and blue). Each pixel component has a value in the range of 0–1. The image can be considered a 3-D matrix.

## Project Tasks
The project can be divided into the following tasks:

1. Import the image file and display it in the Jupyter notebook.
2. Zero pad the image so that it can be divided into blocks of size 8×8.
3. Transform the image from RGB to YCbCr.
4. Subsample the chrominance components by taking an average of four values in all the non-overlapping sliding windows of size 2×2 in the image.
5. Write the image data in a file, enc_chelsea.ycbcr, such that:
6. The first three numbers in the file are the image’s original dimensions.
7. The succeeding numbers contain the original luminance and the subsampled chrominance components in row-major order.
8. Read the image data from the encoded file and store it in a NumPy array.
9. Restore the size of the chrominance layers through piecewise constant interpolation.
10. Reconstruct the compressed image using the YCbCr data and display it in the Jupyter notebook.
11. Examine the compression ratio achieved by this algorithm in terms of file size.

## Task 1: Read the image file and display it
* Read the image file and store it in a numpy array using the imread() function of the matplotlib.image module.
* Display the image using the imshow() function of the matplotlib.pyplot module.

## Task 2: Convert the RGB image to YCbCr
* Convert the RGB image to the YCbCr color space using the rgb_to_ycbcr() function of the color module from the skimage library.
* The resulting image will have three layers: luminance, blue chrominance, and red chrominance.

> rgb_to_yCbCr = np.array([[ 0.299, -0.168736,  0.5],
>                          [0.587, -0.331264, -0.418688],
>                          [ 0.114, 0.5, -0.081312]])

## Task 3: Subsample the chrominance layers
* Reduce the size of the blue and red chrominance layers by averaging over a non-overlapping sliding window of size 2x2.
* Store the subsampled layers in separate numpy arrays.
* The resulting images will have half the width and height of the original image.

## Task 4: Compress the image
* Store the image dimensions and subsampled chrominance layers in a numpy array.
* Use the tofile(filename) method of numpy to write the compressed data to a file.

## Task 5: Decompress the image
* Read the image dimensions from the file and store them in separate variables. Use the fromfile(filename) method of numpy to read the file data.
* Adjust the image dimensions for zero padding.
* Print the image dimensions. The image dimensions are stored in the first three data elements of enc_chelsea.ycbcr.
* Read the image data from the enc_chelsea.ycbcr file.
* Create a zero array yCbCr_dec of the same size as the image to store the decoded image.
* Store the image luminance data from the file in the first layer of yCbCr_dec.
* Store the subsampled chrominance data into another zero matrix, CbCr_dec.
* The size of the yCbCr_dec array will be length×width×depth.
* The size of the CbCr_dec array will be length/2×width/2×depth−1.

## Interpolation
Interpolation is a technique where missing data points are estimated using data points that are already available. In this task, we will use piecewise constant interpolation to restore the size of the chrominance layers.

* Create a zero array, CbCr_dec_full, which will store the interpolated chrominance layers.
* Fill the entries such that each 2×2 subsection of the layers is filled with the pixel value of the corresponding subsampled chrominance layer.
* The size of CbCr_dec_full will be length×width×depth−1.

## Conversion to RGB
* Subtract 128 from the entries of CbCr_dec_full and store the layer data in the yCbCr_dec array.
* Transform the YCbCr image back to the RGB domain using the yCbCr_to_rgb conversion matrix and scale their range to 0–1.
* Use the matmul() operation of numpy to perform matrix multiplication.
* After the transformation of the image back to the RGB domain, the pixel range may not necessarily be 0–1. Use the putmask() function of numpy to ensure the correct pixel range.
* Remove the zero padding to retrieve the final image.
* Display the resulting image alongside the original image.

> yCbCr_to_rgb = np.array([[1.0, 1.0, 1.0],
>                          [0.0, -0.344136, 1.772],
>                          [1.402, -0.714136, 0.0]])

## Task 6: Evaluate the compression ratio

Compression ratio is a metric that indicates the extent of data reduction due to compression. It is evaluated using the following formula:

> Compression ratio = Size of uncompressed image / Size of compressed image × 100

Use the file.stat.st_size attribute from the os library to get the file sizes.
Evaluate the compression ratio of the compressed and uncompressed images.