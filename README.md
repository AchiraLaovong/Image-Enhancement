# Image-Enhancement

Author: [Achira Laovong](https://github.com/AchiraLaovong)

Contact: achiralaovong@gmail.com

Date: 2024-08-10

This repository is a small project on SRCNN (Super Resolution Convolutional Neural Network) for image enhancement.
This is done as a personal project to understand the logic behind SRCNN and how it works. This project is implemented
using TensorFlow and Keras. The dataset used for training is the DIV2K dataset which is a high-quality dataset for image
super-resolution. The dataset can be downloaded from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

## Requirements
- Python 3.10.11
- TensorFlow 2.17.0
- Keras 3.4.1
- Numpy 1.24.3
- Matplotlib 3.9.1
- Scikit-Image 0.24.0
- Scikit-Learn 1.5.1
- Pillow 10.4.0

## Design

The SRCNN model is made up of three layers:
1. Convolutional Layer 1: 9x9 kernel size, 64 filters, ReLU activation, padding = 'same', kernel_initializer = 'glorot_uniform'
2. Convolutional Layer 2: 3x3 kernel size, 32 filters, ReLU activation, padding = 'same', kernel_initializer = 'glorot_uniform'
3. Convolutional Layer 3: 5x5 kernel size, 3 filters, Linear activation, padding = 'same', kernel_initializer = 'glorot_uniform'

The model is trained using the DIV2K dataset. The dataset is divided into 560 images for training, 120 images for validation, and 120 images for testing. 
The images are cropped to 512x512 pixels and downsampled to x3 smaller size and scaled back to 512x512 pixels using bicubic interpolation.

The model is trained using a custom loss function using the mean squared error (MSE), structural similarity index (SSIM), and VGG19 loss. 
The loss function is defined as follows:

```
loss = 0.8 * sigmoid(MSE) + 0.1 * (1 - (1 + SSIM)/2) + 0.1 * sigmoid(VGG19_loss)
```
This loss function is used to balance the trade-off between the MSE, SSIM, and VGG19 Perceptual loss. This custom loss function is not optimal and can be improved.

The model is trained using the Adam optimizer with a learning rate of 1e-4 and a batch size of 2. The model is trained for 100 epochs with early stopping if the validation loss does not improve for 10 epochs.

## Results

The model is evaluated using the test dataset. The results are shown below:

Original Image:

![Original Image](original.png)

Bicubic Interpolation:

![Bicubic Interpolation](bi_example.png)

SRCNN:

![SRCNN](sr_image_46.png)

SRCNN's Overall Evaluation Metrics:
- Average PSNR: 30.1528
- Average SSIM: 0.8926
- Average MSE: 107.8839
- Range of PSNR: 18.9426 - 49.5030
- Range of SSIM: 0.5610 - 0.9918
- Range of MSE: 0.7291 - 829.5155

Bicubic Interpolation's Overall Evaluation Metrics:
- Average PSNR: 34.2904
- Average SSIM: 0.8825
- Average MSE: 93.8065
- Range of PSNR: 18.7908 - 58.9477
- Range of SSIM: 0.4983 - 0.9982
- Range of MSE: 0.0829 - 859.0077

Although the Bicubic Interpolation scores higher for most of the evaluation metrics, the SRCNN model is able to produce better visual results compared to Bicubic Interpolation as shown in the images above.

## References

- [Ben Garber, Aitan Grossman, Sonja Johnson-Yu, "Image Super-Resolution Via a Convolutional Neural Network", 2020](https://medium.com/analytics-vidhya/srcnn-paper-summary-implementation-ad5cea22a90e)
- [Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang, "Image Super-Resolution Using Deep Convolutional Networks", 2016](https://arxiv.org/abs/1501.00092)
- [Sieun Park, “SRCNN Paper Summary & Implementation”, 2021](https://medium.com/analytics-vidhya/srcnn-paper-summary-implementation-ad5cea22a90e)