## Detection of Covid-19 using Chest X-Ray

This project aims to leverage chest X-ray images to develop a predictive model for COVID-19 diagnosis. By analyzing the characteristics and patterns observed in these images, the model seeks to accurately predict whether a patient is diagnosed with COVID-19. The proposed approach offers the potential for a faster and more accessible diagnostic method, especially in areas where RT-PCR testing resources may be limited.


### Background: 

The first documented instance of a highly transmissible illness caused by the Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2) was detected in Wuhan, China, in December 2019. This project aims to leverage chest X-ray images to develop a predictive model for COVID-19 diagnosis. The need for auxiliary diagnostic tools increased as there are no accurate automated toolkits available. Recent findings obtained using radiology imaging techniques suggest that such images contain salient information about the COVID-19 virus. Application of advanced artificial intelligence (AI) techniques coupled with radiological imaging offers the potential for a faster and more accessible diagnostic method, especially in areas where RT-PCR testing resources may be limited.

### Project Workflow

![image](https://github.com/ACM40960/project-divya-dwivedi-ucd/assets/133960362/07d0f198-46fc-457c-9afd-937995acad39)


The proposed system works in following steps:

1. Identifying and splitting the data by labels which can be used to train our model.

2. Designing a supervised learning model(CNN) which will learn from the different features of the image and provide predictions with the highest accuracy possible.

3. Finally, our model will be evaluated using the test dataset and the results will be noted down which will indicate if the proposed model can be used to detect Covid-19 cases. The focus here will be on the false-negatives as the goal is to predict the positive cases of Covid-19 correctly.

### Project Implementation

Follow this step by step guide to be able to predict if a person is diagnosed with Covid-19 or Pneumonia using a chest X-Ray.

The entire project is done using Jupyter Notebook (Python). The first step is to install Python on your device. You can use Anaconda to install Pythonm as it comes with a number of pre-installed packages generally used in machine learning and data science. This saves a lot of effort and time as one does not need to install each package separately.

Follow the link to install the latest version of Anaconda as per your system OS. https://docs.anaconda.com/free/anaconda/install/index.html

### Packages Required

For the purpose of the project you would need a number of packages within python. You can use the pip command in Anaconda prompt to install the following packages. Rest of the packages will be already installed through Anaconda. If not use pip to install them

```
pip install tensorflow
pip install keras
pip install opencv-python
pip install -U albumentations

```

Next, we open the Jupyter notebook to begin the implementation. First, we import the installed packages using the following commands.

```
import keras
import cv2, os, gc, glob

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import matplotlib.cm as cm

from tqdm import tqdm
from tensorflow.keras import layers, models

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense
from keras.optimizers import Adam as adam
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import albumentations as A

```


### Dataset

For the purpose of the model we use the following dataset which was taken from kaggle.

Link: https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

The Dataset consists of 6536 x-ray images and contains three sub-folders (COVID19, PNEUMONIA, NORMAL). The data is then divided into train, validation and test sets where the test set is 20% of the total data.

![image](https://github.com/ACM40960/project-divya-dwivedi-ucd/assets/133960362/abaf7980-562d-47f5-9495-92d1ac57375a)

The images above show the he count of individual images of each category. The dataset consists of 676 Covid X-Rays, 1585 Normal X-Rays and 4275 Pneumonia X-Rays.

Before evaluating a model, it is crucial to report the demographic statistics of their datasets, including age and sex distributions. Diagnostic studies commonly compare their models’ performance to that of RT–PCR.

The data set used for this study is clinical data for individuals aged between 10-90 years with 65% Male and 35% Female ratio. [1]


<img width="1046" alt="image" src="https://github.com/ACM40960/project-divya-dwivedi-ucd/assets/133960362/d7f30378-ba05-4d5c-b489-35bbef754afe">



### Model

The model designed here uses three convolutional layers with an input shape of (100, 100, 3) and Rectified Linear Unit (ReLU) activation function to introduce non-linearity in the model. The first layer has 32 filters to convolve over the input image. It also employs dropout and L2 regularization techniques to improve generalization and prevent overfitting. In the final dense layer, the softmax activation function is used to obtain the class probabilities for multi-class classification

![image](https://github.com/ACM40960/project-divya-dwivedi-ucd/assets/133960362/e5cd5125-c7a2-4be9-8458-7f7cf324ede6)

You can find the model in the main code file. The summary of the model can be seen below.


![image](https://github.com/ACM40960/project-divya-dwivedi-ucd/assets/133960362/08273686-3f6f-4854-bee8-aede11e6e880)

The model is then fit for 50 epochs with a batch size of 32. You can do this using the below code. 

**Beware - Running the model will take time and computing resources.**

```
epochs = 50
cnn = model.fit(trainX, trainY, validation_data = (valX, valY), epochs=epochs,
                      batch_size=32,  verbose = 1)

```

Alternatively you can directly used the trained model using the already trained model. The file can be downloaded from here https://drive.google.com/file/d/1G5PzPSijc4dIWoQeKjLQcKqUtxFXeggS/view?usp=sharing. You will have to load the file to run the model.

```
model = tf.keras.saving.load_model("Paste the file location here")
```

This has been our best attempt in creating the model with the highest accuracy. You can change the parameters as per your requirements. Keep reading to see the results from this model.


### Results

We are interested in the F1 score of the model. This score provides the balance between precision and recall or in other words it is the accuracy for individual class. From the output below, we can see the overall accuracy is 95.9% for validation and 95.4% for test data. The support count for each class represents the number of images on which the model training and testing was performed. 

<img src="https://github.com/ACM40960/project-SidBagwe98/assets/134402582/a48c870e-95b5-4591-84c4-f7454840c052" width="400" height="300"> <img src="https://github.com/ACM40960/project-SidBagwe98/assets/134402582/d083e420-b2c2-4e77-a1a3-a303a3ab2ebb" width="400" height="300">

To analyze the model classification we look into the Confusion matrix of our proposed
model. We can see that the sensitivity (Recall) of Covid-19 (96.2%) is at par with
sensitivity of Pneumonia (96.7%). Due to the fatality of the problem in hand, we aim
to focus on the False Negatives of the model which is only 1 case out of the total
dataset. This is likely due to the overlapping imaging characteristics.


### Grad-Cam Visualisation

Given the severity of the issue at hand, it is imperative that we create a visualization
of the scare tissue presence. Gradient-wighted Class Activation Mapping uses the
gradients from any target convolutional layer to create a local map which highlights
important feature that the model has learned.The images on the left column are the
actual input images of chest X-ray. heatmap of class activation is then generated
from the image based on the detected features from the image. Finally, the heatmap
is superimposed on the actual image to clearly show the presence of the COVID-19
scare tissue on the X-ray image.

![image](https://github.com/ACM40960/project-divya-dwivedi-ucd/assets/133960362/e30d7887-785e-434c-846c-333bf3f1d7d3)

The high-intensity visuals (blue and green) reflects the area of interest to our model
at the time of prediction

### Application: 

In light of our study to detect Covid-19 using X-ray images, the application of new models has shown promising potential in several areas:

**1. Early Detection and Diagnosis:** Advanced machine learning models can analyze raw chest X-ray images to detect COVID-19 at an early stage, allowing for quicker diagnosis and appropriate treatment, which is crucial for controlling the spread of the virus and providing timely care to patients.

**2. Screening and Triage:** Automated systems can be integrated into healthcare facilities to screen and triage patients, identifying those with potential COVID-19 symptoms, which can help reduce the burden on healthcare resources and minimize the risk of transmission in hospital settings.

**3. Remote and Point-of-Care Diagnosis:** These models could be deployed in remote or point-of-care settings, enabling healthcare professionals in underserved areas or mobile clinics to identify COVID-19 cases rapidly and accurately without the need for immediate access to specialized medical facilities.

**4. Support for Radiologists:** AI-based models can assist radiologists in interpreting chest X-rays, offering a second opinion, enhancing efficiency, and reducing diagnostic errors. Radiologists can focus more on critical cases, leading to better patient outcomes.

**5. Surveillance and Monitoring:** Automated COVID-19 detection systems can be used for real-time monitoring of large populations, helping public health authorities track the spread of the virus and implement targeted interventions to contain outbreaks.

**6. Research and Data Analysis:** The vast amount of data generated from chest X-ray images can be utilized for research purposes, gaining insights into the disease progression, risk factors, and treatment outcomes. AI models can aid researchers in identifying patterns and correlations that might otherwise be challenging to detect manually.

It's important to note that the field of AI in medical imaging and COVID-19 detection is continuously evolving. 

### Challenges and opportunities

Models developed for diagnosis and prognostication from radiological imaging data are limited by the quality of their training data. While many public datasets exist for researchers to train deep learning models for these purposes, we have determined that these datasets are not large enough, or of suitable quality, to train reliable models, and all studies using publicly available datasets exhibit a high or unclear risk of bias. However, the size and quality of these datasets can be continuously improved if researchers worldwide submit their data for public review. Because of the uncertain quality of many COVID-19 datasets, it is likely more beneficial to the research community to establish a database that has a systematic review of submitted data than it is to immediately release data of questionable quality as a public database.

The intricate link of any AI algorithm for detection, diagnosis or prognosis of COVID-19 infections to a clear clinical need is essential for successful translation. As such, complementary computational and clinical expertise, in conjunction with high-quality healthcare data, are required for the development of AI algorithms. Meaningful evaluation of an algorithm’s performance is most likely to occur in a prospective clinical setting. Like the need for collaborative development of AI algorithms, the complementary perspectives of experts in machine learning and academic medicine were critical in conducting this systematic review.

### Conclusion: 

AI computational models used to assess chest X-rays in the process of diagnosing COVID-19 should achieve sufficiently high sensitivity and specificity. Their results and performance should be repeatable to make them dependable for clinicians. Moreover, these additional diagnostic tools should be more affordable and faster than the currently available procedures. The performance and calculations of AI-based systems should take clinical data into account

In our experiments, we have also applied a color visualization approach by using the Grad-CAM technique to make the proposed deep learning model more interpretable and explainable. The obtained results reveal that the patient diagnosed with Pneumonia has more chances to get tested as a False Positive by the proposed algorithm. Therefore, to detect the COVID-19 cases accurately with higher recall, it is suggested to train the model on radiology images of patients with Pneumonia symptoms as well. This will help us to detect pneumonia patients as True Negative (just for clarification-here, COVID-19 cases are True Positive) which were previously detected as false positive. This results in an unbiased detection of COVID-19 cases in a real-time scenario.

### Acknowledgement 

I would like to express my sincere gratitude to Dr. Sarp Akcay for all his guidance and support throughout the module at University College Dublin. I would also like to thank Mr. Prashant Patel for combining the dataset and contributing towards the success of this project. I am grateful to the University College Dublin for providing resources that made this project possible. Lastly, I want to thank everyone who directly or indirectly supported me during this endeavor.


### References

[1]  Joseph Paul Cohen, IEEE. (n.d.). GitHub - ieee8023/covid-chestxray-dataset: We are building an open database of COVID-19 cases with chest X-ray or CT images. GitHub. https://github.com/ieee8023/covid-chestxray-dataset

[2] D. Cucinotta and M. Vanelli. “who declares covid-19 a pandemic”. Acta Biomedica: Atenei Parmensis, 91:157–160,
2020.

[3] Humayun M Jhanjhi NZ Gouda W, Almurafeh M. Detection of covid-19 based on chest x-rays using deep learning.
Healthcare (Basel), page 343, February 2022.

[4] P. K. Ratha P. K. Sethy, S. K. Behera and P. Biswas. Detection of coronavirus disease (covid-19) based on deep
features and support vector machine. 5:643–651, 2020.

[5] S. Kasaei R. Rouhi, M. Jafari and P. Keshavarzian. Benign and malignant breast tumors classification based on region
growing and cnn segmentation. Expert Systems with Applications, 42:990–1002, 2015.

[6] A. S. Lundervold and A. Lundervold. An overview of deep learning in medical imaging focusing on mri. Zeitschrift fur
Medizinische Physik, 29:102–127, 2019.


### License

Each image has License: CC BY 4.0 as specified in the Kaggle Dataset Metadata, including Apache 2.0, CC BY-NC-SA 4.0, CC BY 4.0.
All scripts and  documents are released under a CC BY-NC-SA 4.0 license. Companies are free to perform research. Beyond that contact us on divya.dwivedi@ucdconnect.ie and siddhesh.bagwe@ucdconnect.ie

