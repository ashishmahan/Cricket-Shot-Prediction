# Cricket-Shot-Prediction
Overview
This project aims to classify different types of cricket shots using computer vision techniques and machine learning algorithms. It utilizes the Mediapipe library for extracting pose keypoints from cricket shot images and a Random Forest classifier to predict the type of shot based on these keypoints.

Key Features
Shot Classification: The system can classify cricket shots into various types such as drive, flick, pull, and defense.

Pose Keypoint Extraction: Pose keypoints are extracted from cricket shot images using the Mediapipe library, which provides a robust solution for pose estimation.

Random Forest Classifier: The extracted pose keypoints are used as features for training a Random Forest classifier, which learns to classify different shot types based on these features.

Visualization: The system provides visualizations of the detected pose keypoints overlaid on the input images, allowing users to see the points of interest used for classification.

I have taken the dataset from Kaggle. This dataset is a collection of images from the internet.
General Information:
The directory drives consists of the cover drive, straight drive and off drive.
The directory legglance-flick contains the images for the leg glance and flick shot.
The directory pullshot has the images for pull shot.
The directory sweep has the image for sweep shot.

https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset/data 
