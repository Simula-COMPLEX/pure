# Pure: Prediction Surface Uncertainty Quantification in Object Detection Models for Autonomous Driving

## Description
Object detection in autonomous vehicles is commonly based on camera images and Lidar inputs, which are often used to train prediction models such as deep artificial neural networks for decision making for object recognition, adjusting speed, etc. A mistake in such decision making can be damaging; thus, it is vital to measure the reliability of decisions made by such prediction models via uncertainty measurement. Uncertainty, in deep learning models, is often measured for classification problems. However, deep learning models in autonomous driving are often multi-output regression models. Hence, we propose a novel method called PURE (Prediction sURface uncErtainty) for measuring prediction uncertainty of such regression models. We formulate object recognition problem as a regression model with more than one outputs for finding object locations in a 2-dimensional camera view. For evaluation, we modified three widely-applied object recognition models (i.e., YoLo, SSD300 and SSD512) and used the KITTI, Stanford Cars, Berkeley Deep Drive, and NEXET datasets. Results showed the statistically significant negative correlation between prediction surface uncertainty and prediction accuracy suggesting that uncertainty significantly impacts the decisions made by autonomous driving.  

## System overview
![Process](https://raw.githubusercontent.com/Simula-COMPLEX/pure/main/desc_images/system-overview.png)

## People
* Ferhat Ozgur Catak http://www.ozgurcatak.org
* Tao Yue https://www.simula.no/people/tao
* Shaukat Ali https://www.simula.no/people/shaukat
