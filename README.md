# ChestXRay-Analysis
This repository consists  of the final year project on Pneumonia Detection using Chest X-Ray Analysis by Shuvam Aich, Rashmiman Nandi and Anjishnu Roy, Semester 6, Department of Computer Science, St. Xavier's College, Kolkata

PNEUMONIA DETECTOR: Convolutional Neural Network Architecture designed for the Detection of Pneumonia using Chest-X Rays



Anjishnu Roy
Roll: 553

Rashmiman Nandi
Roll: 540


Shuvam Aich
Roll: 588



Supervised By:
Prof. Sonali Sen


















St. Xavier’s College (Autonomous), Kolkata
Department of Computer Science
30, Mother Teresa Sarani, Kolkata-700016
1. INTRODUCTION
Pneumonia is a common disease caused by different microbial species such as bacteria, virus, and fungi. The word “Pneumonia” comes from the Greek word “neumon” which translates to the lungs. Thus, the word pneumonia is associated with lung disease. In medical terms, pneumonia is a disease that causes inflammation of either one or both lung parenchyma.
Some of the symptoms of pneumonia include the shortness of breath, fever, cough, chest pain, etc. Moreover, the people at risk of pneumonia are elderly people (above 65 years), children (below the age of 5 years), and people with other complications such as HIV/AIDS, diabetes, chronic respiratory diseases, cardiovascular diseases, cancer, hepatic disease, etc.
The risk of pneumonia is immense for many, especially in developing nations where billions face energy poverty and rely on polluting forms of energy. The WHO estimates that over 4 million premature deaths occur annually from household air pollution-related diseases including pneumonia.
COVID-19 is an extremely contagious disease caused by severe acute respiratory syndrome coronavirus 2 (SAR-CoV-2) which is the recent disease that is caused by one of the family members of Coronaviridae family. COVID-19 can be transmitted through respiratory droplets that are exhaled or secreted by infected persons. Coronaviruses invade the lung’s alveoli (an organ responsible for exchange of O2 and CO2), thereby causing pneumonia.


2. OBJECTIVE

Deep neural network models have conventionally been designed, and experiments were performed upon them by human experts in a continuing trial-and-error method. This process demands enormous time, know-how, and resources .We are proposing a deep learning-based model that can quickly prognose the presence of Pneumonia. The main objective of this study is focused on building a classification model that will bring down the false negatives (people with Pneumonia but predicted otherwise)to a substantially low number along a model with high evaluation metrics (precision, accuracy, sensitivity and specificity). As this is a prognosis model, we can accommodate false positives (people not having Pneumonia but predicted otherwise) to a small ex

3. PROPOSED ALGORITHM

Step 1: 
We have collected our dataset from Chest X-Ray Images (Pneumonia) | Kaggle. Our dataset consists of a total of 5856 Chest X-Ray images which has been splitted approximately in a 90-10% ratio. The original dataset consists of three main folders (i.e., training, testing, and validation folders) and two subfolders containing pneumonia (P) and normal (N) chest X-ray images, respectively. Only the frontal Chest-X Ray images have been chosen for our model.

Name of Sets
Total Number of Images
Pneumonia(P)
Normal(N)
Training Set
5216
3875
1341
Test Set
624
390
234
Validation Set
16
8
8
Total
5865
3673
1583


Step 2: 
The primary goal of using Convolutional Neural Network in most of the image classification tasks is to reduce the computational complexity of the model which is likely to increase if the input are images. So we have performed many image augmentation methods to increase the size and quality of the images. The original 3-channel images can be resized into 150×150 pixels to reduce the heavy computation and for faster processing.  The ‘rescale’ option helps in reducing the size or magnification of the image (rescale=1./255) and we can randomly rotate the images during training by 40 degrees. We can also add a width shift (horizontal translation) and height shift(vertical translation) of 0.2%. A shear range and a zoom range of 0.2 can also be added. Finally the images were flipped horizontally.  

Step 3: A Pandas dataframe is like a one dimensional array containing all our data. Our target is to create a dataframe, after fetching our training set with the help of the ‘os- operating system interface’ by Python, with the following class indices:
Normal : 0
Pneumonia : 1

Step 4: 
Matplotlib is a comprehensive library which has been used to create static, animated, and interactive visualizations in Python. We have plotted our dataframe with the help of matplotlib.



Step 5: 
With Keras, building models is as simple as stacking layers and connecting graphs. Keras allows us to express large complex networks as a collection of smaller and simpler networks. All we need to do is to specify our inputs, outputs and ensure that the layers are connected.

Keras Sequential Model: We use a sequential model as it works like a linear stack of layers and is best suited for our classification network. Here we treat every layer as an object that feeds into the next layer and so on.
We first specify the input dimensions, then we define our model architecture (i.e. Sequential model) and our CNN. Finally we select our optimiser i.e Adam/RMSprop/Adadelta and configure the learning rate. Our next step is to define the loss function ‘binary_crossentropy’. So for every step of our training we will be checking our accuracy of prediction by comparing the obtained value with the actual one. We check for the difference between them and obtain the loss. 
Our last step is to train the model with the training data and test the model with the test data to check if the model actually learned anything or not.

Step 6: 
The ML model deployed uses a CNN (Convolutional Neural Network) to extract relevant features from the images for detecting presence of Pneumonia.

The Convolutional Network is composed of several types of layers, and have been enlisted below:
Convolutional Layer: In this layer, there is a filter matrix with a stride length, and the filter matrix moves pixel by pixel over the entire image until it has been completely traversed.
Pooling layer: This layer is responsible for extraction of features from images by reducing the size of the convolved feature. We have used MaxPooling2D layers.

Our network structure has been designed with an increasing number of neurons in the Convolutional Layer. This has been done to ensure to extract the correct features present in an image to optimally alter the trainable parameters, so as to obtain maximum accuracy while obtaining an output.

For the extraction of image features, Convolution and Pooling are used in tandem. Convolutions work similarly to applying filters on an image. If we have an image of size 3x3, then we will also have a filter (or weight) matrix that has the same dimensions. After the application of these filters, certain features of the image get highlighted, while others get suppressed. Our goal here has been not just to simply match the labels of images to raw pixels, and hence convolutions has helped us to extract these important features from the image and help the model learn the differences between a normal and a pneumonic chest x-ray. Along with the use of loss functions, the convolutions help to extract the features which are important, and then the optimizers help tweak the parameters.

We have used MaxPooling2D layers of size 2x2 in our model. The way MaxPooling works is that we have a matrix of size 2x2 that moves over the image given as input. The 4 cells of the matrix correspond to 4 pixels of the image, after the convolution has been applied (and hence features highlighted). For every four pixels, the MaxPooling2D layer gives us the pixel having the maximum value as output. So, it gives a single pixel output for every 4 pixels, and hence the size of the image decreases to 25% at every step. Smaller images with highlighted features aid in computer vision.

If we look into model.summary(), the image dimensions are 148x148, 74x74,..., 7x7. This reduction in size is due to the MaxPooling2D layer taking the maximum weighted pixel out of every 4 pixels. Since the pixels at the corners and at the top may be evaluated more than once, the image size is first dropped to half, i.e, 148x148 to 74x74 and then the corner pixels are cropped out because they have been evaluated twice and the dimension drops to 72x72. The further reduction in the image size can be explained similarly.

Step 7: 
We are planning to train our model upto 10 epochs with steps per epoch = 500 since the number of images in the training set is 5216 (No. of images = Batch size(10) X No. of Steps) and no. of validation steps =62. On obtaining the accuracies, we plan to plot graphs in order to visualise the success rate of our model using matplotlib. 


CONCLUSION

We aim to build a very lightweight model that can be deployed to detect the presence of lung diseases.The accuracy obtained can be increased through using larger datasets. However, due to limited processing power we have obtained a dataset of size 5000 training images. 
There are existing models that are used for medical image classification like ResNet, VGG16, DenseNet, etc. These models give a very high accuracy in the fields of Medical Image classification due to their complex network structure and the relatively large dataset they have been trained upon. However, this also implies that creating a similar multi layered network requires greater processing capabilities, otherwise the training process is quite slow.
Presence of expert radiologists is the topmost necessity to properly diagnose any kind of thoracic disease. Our model aims to improve the medical adeptness in areas where the availability of radiotherapists is still limited. Our study facilitate the early diagnosis of Pneumonia to prevent adverse consequences (including death).
The development of algorithms in this domain can be highly beneficial for providing better health-care services. Our study will likely lead to the development of better algorithms for detecting Pneumonia in the foreseeable future.



REFERENCES


WHO (n.d.). Pneumonia. Retrieved March 31, 2021, from https://www.who.int/news-room/fact-sheets/detail/pneumonia#:~:text=Pneumonia%20is%20a%20form%20of,painful%20and%20limits%20oxygen%20intake.

Mooney, P. (2018, March 24). Chest x-ray Images (Pneumonia). Retrieved March 31, 2021, from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Ibrahim, A.U., Ozsoz, M., Serte, S. et al. Pneumonia Classification Using Deep Learning from Chest X-ray Images During COVID-19. Cogn Comput (2021). https://doi.org/10.1007/s12559-020-09787-5

Convolutional Neural Network Architecture: CNN Architecture | Analytics Vidhya | Accessed on: 31.03.2021 | Published on 14.01.2021 | https://www.analyticsvidhya.com/blog/2020/10/what-is-the-convolutional-neural-network-architecture/

Ng, Andrew (Instructor). (n.d.). Neural networks and Deep Learning [Video file]. Retrieved March 31, 2021, from https://www.coursera.org/learn/neural-networks-deep-learning


D. Varshni, K. Thakral, L. Agarwal, R. Nijhawan and A. Mittal, "Pneumonia Detection Using CNN based Feature Extraction," 2019 IEEE International Conference on Electrical, Computer and Communication Technologies (ICECCT), Coimbatore, India, 2019, pp. 1-7, doi: 10.1109/ICECCT.2019.8869364.

Jain, S. (2020, May 15). Regularization techniques: Regularization in deep learning. Retrieved March 31, 2021, from https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/


Ahsan, A. O. (2020, June 05). Convolutional neural network and regularization techniques with TensorFlow and Keras. Retrieved March 31, 2021, from https://medium.com/intelligentmachines/convolutional-neural-network-and-regularization-techniques-with-tensorflow-and-keras-5a09e6e65dc7

Moroney, L. (Writer). (n.d.). Improving deep neural Networks: Hyperparameter tuning, regularization and optimization [Video file]. Retrieved March 31, 2021, from https://www.coursera.org/learn/deep-neural-network

Moroney, L. (Director). (n.d.). Machine learning foundations [Video file]. Retrieved March 31, 2021, from https://www.youtube.com/playlist?list=PLOU2XLYxmsII9mzQ-Xxug4l2o04JBrkLV


Moroney, L. (Director). (n.d.). Introduction to tensorflow for artificial intelligence, machine learning, and deep learning [Video file]. Retrieved March 31, 2021, from https://www.coursera.org/learn/introduction-tensorflow

Uniyal, M. (Instructor). (2020, April 27). Detecting covid-19 from x-ray| Training a convolutional neural network | Deep learning [Video file]. Retrieved March 31, 2021, from https://www.youtube.com/watch?v=nHQDDAAzIsI

Covid-19 detection using x-ray images  (convolutional neural Network training) [Video file]. (2020, July 01). Retrieved March 31, 2021, from https://www.youtube.com/watch?v=ol0OYJoBC4A

Jain, R., Gupta, M., Taneja, S. et al. Deep learning based detection and analysis of COVID-19 on chest X-ray images. Appl Intell 51, 1690–1700 (2021). https://doi.org/10.1007/s10489-020-01902-1
