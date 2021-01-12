# Style-Transfer-Example
This example projects demonstrate the use of Style Transfer in python, iOS, Android mobile applications. 

## Artistic-Style-Transfer
Artistic style transfer takes two input images: one representing "content" and one representing "style"
, such as an artwork by a famous painter or a texture photo to resemble) and blend them together 
so the output image looks like the content image, but "painted" in the style of the style reference image.
![example](style_tx_cat.png)

### Structure 
Main folder contains two notebooks. Both uses VGG-19 pre-trained model from GPU/PyTorch and GPU/Tensorflow. VGG-19 network architecture looks as follows: 
![VGG-19](vgg19_convlayers.png)
* PyTorch VGG-19 pretrained model based style transfer
* Tensorflow VGG-19 pretrained model based style transfer
<!-- TODO --> 
These mobile platform requires lite-weight cpu-intensive model so that a pre-trained TensorFlow Lite model and its API are used. 
* [Android style Transfer](/android/README.md)
* [iOS Style Transfer](/ios/README.md)
