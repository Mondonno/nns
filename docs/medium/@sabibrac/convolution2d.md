# Implementing Image Processing Kernels from scratch using Convolution in Python | by Sabri Barac | Medium

# _Implementing Image Processing Kernels from scratch using Convolution in Python_

[

![Sabri Barac](https://miro.medium.com/v2/resize:fill:64:64/1*iuDNleXkiqTZ6fAnbAencA.jpeg)





](/@sabribarac?source=post_page---byline--4e966e9aafaf---------------------------------------)

[Sabri Barac](/@sabribarac?source=post_page---byline--4e966e9aafaf---------------------------------------)

Follow

5 min read

·

Jan 4, 2023

[

](/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fvote%2Fp%2F4e966e9aafaf&operation=register&redirect=https%3A%2F%2Fmedium.com%2F%40sabribarac%2Fimplementing-image-processing-kernels-from-scratch-using-convolution-in-python-4e966e9aafaf&user=Sabri+Barac&userId=28ac14924659&source=---header_actions--4e966e9aafaf---------------------clap_footer------------------)

36

2

[](/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fbookmark%2Fp%2F4e966e9aafaf&operation=register&redirect=https%3A%2F%2Fmedium.com%2F%40sabribarac%2Fimplementing-image-processing-kernels-from-scratch-using-convolution-in-python-4e966e9aafaf&source=---header_actions--4e966e9aafaf---------------------bookmark_footer------------------)

[

Listen









](/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2Fplans%3Fdimension%3Dpost_audio_button%26postId%3D4e966e9aafaf&operation=register&redirect=https%3A%2F%2Fmedium.com%2F%40sabribarac%2Fimplementing-image-processing-kernels-from-scratch-using-convolution-in-python-4e966e9aafaf&source=---header_actions--4e966e9aafaf---------------------post_audio_button------------------)

Share

Convolutions are used for Deep Learning, Convolutional Neural Networks (CNN) to extract features from images and use them in the classification process. A convolution filter (the correct term in this case is rather correlation filter) or also called, kernel, are usually not a fixed preset in CNN models, but rather they instead learn from training data using a optimisation process. This allows the model to extract the most useful features, rather than relying on predefined image processing techniques. And a bonus point is the kernel allows the model to learn invariance, so it can recognise the same feature regardless of the position of a certain feature in an image.

This process we can define it mathemically by a formula:

![](https://miro.medium.com/v2/resize:fit:455/1*qX6sZT0vJx56Zco7GDfMuQ.jpeg)
g(x,y) is the filtered image, f(x,y) is the original image and w is the kernel

> One key difference between correlation and convolution is that the former is not commutative, while the latter is. While this may seem like a minor distinction in practice, it has important implications in mathematics. Despite this, the terms “convolution” and “cross-correlation” are often used interchangeably, particularly when discussing image processing.

In this article we will create a kernel and apply the (3D) convolution to an RGB image from scratch just using **NumPy** and **PIL.** But first, let me explain what happens when a kernel is convolved to an image.

## Convolution Kernel

In convolutional neural networks, the process of convolution is applied to rank-3 tensors called feature maps. These tensors have three axes: height, width, and depth (also known as the channels axis). When working with color images, the depth of the feature map is typically 3, corresponding to the red, green, and blue color channels. For grayscale images, the depth of the feature map is typically 1. Convolution is applied separately to each channel of the feature map, sliding a kernel over the spatial dimensions (height and width) and computing the dot product at each location. This process is repeated for each channel, allowing the model to learn features that are specific to each color channel. The kernels are usually odd sized, typically 3x3 or 5x5. In this article, it will be 3x3, which is a common choice.

Let’s apply this to an example, in this case the kernel is a rank-3 tensor where the height, width is 3 and the depth 1 and the ‘_image_’ is 5x5x1.

![](https://miro.medium.com/v2/resize:fit:700/1*HlERw13cHl-ha8ZcWjjOKA.jpeg)
From left to right, ‘image’, kernel and output

## 3D Convolution Coded

To determine the size of the output of a convolutional operation, we can use the following formula:

![](https://miro.medium.com/v2/resize:fit:330/1*-3BBReknbIMaqYmk4FIFRQ.jpeg)
In this example the padding will be 0 and stride 1

We could implement padding and striding, but for the sake of keeping it relatively easy we will set the padding to 0 and stride to 1.

Padding is the number of pixels added to the border of the input data before the convolution is applied. Padding is used to control the size of the output of the convolution. If the input data is padded with zeros, the output of the convolution will be the same size as the input. If no padding is used, the output will be smaller than the input. In our case the image will be smaller by 2 pixels, 1 on each side.

Stride is the number of pixels the filter moves each time it is applied to the input data. A larger stride will result in a smaller output size, because the filter will cover fewer regions of the input data. A smaller stride will result in a larger output size, because the filter will cover more regions of the input data. In our case the stride will be 1, so for every pixel it will move by 1 pixel.

The following code is applied, the comments through the code will explain what happens:

from PIL import Image  
import numpy as np  
  
def apply\_convolution(img:np.array, kernel:np.array):  
      
    \# Get the height, width, and number of channels of the image  
    height,width,c =img.shape\[0\],img.shape\[1\],img.shape\[2\]  
      
    \# Get the height, width, and number of channels of the kernel  
    kernel\_height,kernel\_width = kernel.shape\[0\],kernel.shape\[1\]  
      
    \# Create a new image of original img size minus the border   
    \# where the convolution can't be applied  
    new\_img = np.zeros((height-kernel\_height+1,width-kernel\_width+1,3))   
      
    \# Loop through each pixel in the image  
    \# But skip the outer edges of the image  
    for i in range(kernel\_height//2, height-kernel\_height//2\-1):  
        for j in range(kernel\_width//2, width-kernel\_width//2\-1):  
            \# Extract a window of pixels around the current pixel  
            window = img\[i-kernel\_height//2 : i+kernel\_height//2+1,j-kernel\_width//2 : j+kernel\_width//2+1\]  
              
            \# Apply the convolution to the window and set the result as the value of the current pixel in the new image  
            new\_img\[i, j, 0\] = int((window\[:,:,0\] \* kernel).sum())  
            new\_img\[i, j, 1\] = int((window\[:,:,1\] \* kernel).sum())  
            new\_img\[i, j, 2\] = int((window\[:,:,2\] \* kernel).sum())  
        
    \# Clip values to the range 0-255  
    new\_img = np.clip(new\_img, 0, 255)  
    return new\_img.astype(np.uint8)  
  
if \_\_name\_\_ == "\_\_main\_\_":  
  
    \# kernel for edge detection  
    kernel = np.array(\[\[-1,-1,-1\], \[-1,8,-1\], \[-1,-1,-1\]\])  
      
    \# kernel for vertical edge detection  
    #kernel = np.array(\[\[-1,0,1\],\[-1,0,1\],\[-1,0,1\]\])  
  
    \# kernel for horizontal edge detection  
    \# kernel = np.array(\[\[-1,-1,-1\],\[0,0,0\],\[1,1,1\]\])  
  
    \# Kernel for box blur   
    \# kernel = np.array(\[\[1/9,1/9,1/9\],\[1/9,1/9,1/9\],\[1/9,1/9,1/9\]\])  
  
    \# Open the image and convert it to an array  
    \# Try to put your own picture!  
    img = Image.open('kitten3.jpg')  
    or\_img = np.asarray(img)  
      
    new\_img = apply\_convolution(or\_img, kernel)  
  
    \# Create a PIL image from the new image and display it       
    sImg = Image.fromarray(new\_img)  
    sImg.show() 

## Apply the code to an image

We will apply this code to the following kitten image, applying edge detection and blur the image:

![](https://miro.medium.com/v2/resize:fit:700/1*Pf7OcAeuX5MwrhBRnQwLbw.jpeg)
Original kitten image

After the convolution process:

![](https://miro.medium.com/v2/resize:fit:700/1*BLM6S4Un-E1-zvCPHw5Kqw.jpeg)
Applied box blur kernel

![](https://miro.medium.com/v2/resize:fit:700/1*q6QbI1XOP03t-d6M9VFN-w.jpeg)
Vertical edge detection

![](https://miro.medium.com/v2/resize:fit:700/1*MzUPlW_CpzSMFZ9bMJW7Sw.jpeg)
Horizontal edge detection

![](https://miro.medium.com/v2/resize:fit:700/1*cUjlb4kBfDlGERS0ktek6g.jpeg)
Edge detection

With the kernel applied through the convolution to the image, we see the differences between the different kernels we used, we have enhanced a blur to the image, and got edge detection.

Hopefully this article was clear how a convolution works and how to apply it through Python. The code can be found on GitHub, [here](https://github.com/sabribarac/convolution_numpy/blob/main/convolution.py).

Thanks for reading!

_If you found this article helpful, leave a clap or a comment!_

## Embedded Content