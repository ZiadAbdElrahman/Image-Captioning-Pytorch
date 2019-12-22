# Image Captioning with attention Pytorch

# Problem Definition
**Image caption generation is an interesting artificial intelligence problem where
a descriptive sentence is generated for a given image. It involves the dual
techniques from computer vision to understand the content of the image and
a language model from the field of natural language processing to turn the
understanding of the image into words in the right order**


# Examples
a man sitting in front of a laptop computer
---
![](./Img/ex1.png)

a close up of a red fire hydrant
---
![](./Img/ex3.png)

a person on a surfboard in the water
---
![](./Img/ex2.png)



# Contents
[***Dataset***](https://github.com/ZiadAbdElrahman/Image-Captioning-Pytorch-#Dataset)

[***Model Architecture***](https://github.com/ZiadAbdElrahman/Image-Captioning-Pytorch-#Model-Architecture)

[***extra Examples***](https://github.com/ZiadAbdElrahman/Image-Captioning-Pytorch-#extra-Examples)


# Dataset
**COCO dataset**
80K training Image 
40K val Image
with 5 captioning for every image

# Model-Architecture
[***Encoder***](https://github.com/ZiadAbdElrahman/Image-Captioning-Pytorch-#Encoder) 

[***Decoder***](https://github.com/ZiadAbdElrahman/Image-Captioning-Pytorch-#Decoder)

[***Attention***](https://github.com/ZiadAbdElrahman/Image-Captioning-Pytorch-#Attention)

[***Beam-Search***](https://github.com/ZiadAbdElrahman/Image-Captioning-Pytorch-#Beam-Search)

![](./Img/ModelArch.png)


# Encoder
Transfer learning : encode the image (224 * 224 * 3 )
into 4096 feature vector using **pre-trained VGG 16**
Linear layer : to get small feature vector with shape of 512



# Decoder
- Its job to encode the feature vector and generate word by
word
- Since it’s generating sequence , it will need RNN, we will
use LSTM
![](./Img/DecoderArch.png)



# Attention
In a setting with Attention, we want the Decoder to be able
to look at different parts of the image at different points in
the sequence.
![](./Img/AtteArch.png)



# Beam-Search
- The straightforward – and greedy – option would be to
choose the word with the highest score and use it to
predict the next word
- It would be best if we could somehow not decide until
we've finished decoding completely, and choose the
sequence that has the highest overall score from a basket
of candidate sequences.
![](./Img/Beemg.png)



# Extra-Examples

**pizza sitting on top of a
wooden table**

![](./Img/ex4.png)
---


**A man riding a wave on
top of a surfboard**

![](./Img/ex5.png)
---


**A close up of a plate of
food with broccoli**

![](./Img/ex6.png)
---


**A small boat on a body of
water**

![](./Img/ex7.png)
---


**A box filled with different
type of donuts**

![](./Img/ex8.png)
---


**A baseball player
swinging a bat at ball**

![](./Img/ex9.png)
---
