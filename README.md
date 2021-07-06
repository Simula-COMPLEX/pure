# Pure: Prediction Surface Uncertainty Quantification in Object Detection Models for Autonomous Driving

## Description
Object detection in autonomous vehicles is commonly based on camera images and Lidar inputs, which are often used to train prediction models such as deep artificial neural networks for decision making for object recognition, adjusting speed, etc. A mistake in such decision making can be damaging; thus, it is vital to measure the reliability of decisions made by such prediction models via uncertainty measurement. Uncertainty, in deep learning models, is often measured for classification problems. However, deep learning models in autonomous driving are often multi-output regression models. Hence, we propose a novel method called **PURE (Prediction sURface uncErtainty)** for measuring prediction uncertainty of such regression models. We formulate object recognition problem as a regression model with more than one outputs for finding object locations in a 2-dimensional camera view. For evaluation, we modified three widely-applied object recognition models (i.e., YoLo, SSD300 and SSD512) and used the KITTI, Stanford Cars, Berkeley Deep Drive, and NEXET datasets. Results showed the statistically significant negative correlation between prediction surface uncertainty and prediction accuracy suggesting that uncertainty significantly impacts the decisions made by autonomous driving.  

## System overview
![Process](https://raw.githubusercontent.com/Simula-COMPLEX/pure/main/desc_images/system-overview.png)

## Tool
* In order to quantify the object prediction uncertainties, a user can use the *PURE* library's *util.py* file. Example:
   ```python
   import util
   # Dropout layers are activated with p=0.3
   mc_dropout_model = util.get_model(0.3, mc_dropout=True)
   image_path = 'images/Berkeley-BDD100K/0a0c3694-4cc8b0e3.jpg'
   mc_locations, uncertainties = util.get_pred_uncertainty(image_path,
                                                           mc_dropout_model,mc_dropout=True, 
                                                           T=50, plot_ground_truth=True)
   ```
   ![Output](https://raw.githubusercontent.com/Simula-COMPLEX/pure/main/desc_images/berkeley1.png)
* [The example Jupyter Notebook](https://github.com/Simula-COMPLEX/pure/blob/main/pure-object-detection-uncertainty-quantification.ipynb)
## People
* Ferhat Ozgur Catak https://www.simula.no/people/ozgur
* Tao Yue https://www.simula.no/people/tao
* Shaukat Ali https://www.simula.no/people/shaukat

## Paper
F.O. Catak, T. Yue and S. Ali. Prediction Surface Uncertainty Quantification in Object Detection Models for Autonomous Driving. *2021 IEEE International Conference On Artificial Intelligence Testing (AITest)*, 2021.