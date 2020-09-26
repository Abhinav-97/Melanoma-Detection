# Melanoma-Detection
Solution for Kaggle's SIIM-ISIC Melanoma Classification that got Rank 57 out of 3314 Participants (Top 2% silver Medal).

[Competition Link](https://www.kaggle.com/c/siim-isic-melanoma-classification/)

[Leaderboard link](https://www.kaggle.com/c/siim-isic-melanoma-classification/leaderboard)

## Dataset
[The training dataset](https://www.kaggle.com/c/siim-isic-melanoma-classification/data) consists of 33,126 lesion images. The test data contains 10982 images. We have also been provided with metadata of patients with corresponding images. The metadata consists of following columns.
* image_name - unique identifier, points to filename of related DICOM or JPEG image
* patient_id - unique patient identifier
* sex - the sex of the patient (when unknown, will be blank)
* age_approx - approximate patient age at time of imaging
* anatom_site_general_challenge - location of imaged site
* diagnosis - detailed diagnosis information (train only)
* benign_malignant - indicator of malignancy of imaged lesion
* target - binarized version of the target variable

Our task is to predict target variable. 0 as target denotes benign and 1 denotes malignant. In addition, we make the use of external dataset. The external dataset consists of malignant data from ISIC 2019 challenge and ISIS 2018 challenge . We also make use of malignant data provided in ISIC website. We deal with a huge imbalance in the training dataset with only 584 samples labelled as 1(malignant) while 32542 samples are labelled as 0. 

## Pre Processing
CNN’s searches for many patterns and features in images. By training models with varying images, we supply CNN’s with multiple features and patterns of different sizes We take ensemble of models trained on different image sizes.

## Data Augmentation
Data Augmentation techniques random rotation and shear augmentation. Augmentation was also provided by random zoom and horizontal and vertical shifting the images. In addition, we also used   random brightness, contrast and saturation. At last random images were centre cropped to provide more augmentation.Some of the Augmented Images are shown below.

<p float="left" align="middle">
  <img src="images/Melanoma1.png" width="60%" />
</p>
<p float="left" align="middle">
  <font size+=1><em>Augmentation applied on various images</em></font>
</p>

<p float="left" align="middle">
  <img src="images/Melanoma2.png" width="60%" />
</p>
<p float="left" align="middle">
  <font size+=1><em>Illustration of various augmentation applied for training, on a single image.</em></font>
</p>
