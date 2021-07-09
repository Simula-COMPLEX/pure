# Pure: Prediction Surface Uncertainty Quantification in Object Detection Models for Autonomous Driving

## Description
Object detection in autonomous vehicles is commonly based on camera images and Lidar inputs, which are often used to train prediction models such as deep artificial neural networks for decision making for object recognition, adjusting speed, etc. A mistake in such decision making can be damaging; thus, it is vital to measure the reliability of decisions made by such prediction models via uncertainty measurement. Uncertainty, in deep learning models, is often measured for classification problems. However, deep learning models in autonomous driving are often multi-output regression models. Hence, we propose a novel method called **PURE (Prediction sURface uncErtainty)** for measuring prediction uncertainty of such regression models. We formulate object recognition problem as a regression model with more than one outputs for finding object locations in a 2-dimensional camera view. For evaluation, we modified three widely-applied object recognition models (i.e., YoLo, SSD300 and SSD512) and used the KITTI, Stanford Cars, Berkeley Deep Drive, and NEXET datasets. Results showed the statistically significant negative correlation between prediction surface uncertainty and prediction accuracy suggesting that uncertainty significantly impacts the decisions made by autonomous driving.  

## System overview
![Process](https://raw.githubusercontent.com/Simula-COMPLEX/pure/main/desc_images/system-overview.png)

## Tool
* First step is to install the dependencies using *requirements.txt* file
   ```bash
   pip install -r requirements.txt 
   ```
   
* In order to quantify the object prediction uncertainties, a user can use the *PURE* library's *util.py* file. In the following example white dashed boxes show the ground-truth boxes, red boxes show the Monte-Carlo dropout based object predictions.
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

## Research Questions
### RQ1: Is the prediction surface an effective uncertainty quantification method for object detection predictions? 
This RQ assesses the effectiveness of PURE. *get_evaluation_result* method can show the prediction performance and the uncertainty quantification values of a given folder containing images and corresponding ground-truth information.

   ```python
    import util
    image_path = 'images/Stanford/'
    image_type = 'jpg'
    # Dropout layers are activated with p=0.3
    mc_dropout_model = util.get_model(0.3, mc_dropout=True)
    avg_iou, TP,precision,recall,f1, uncertainty,total_detected,total_objects = get_evaluation_result(tmp_model, image_path,image_type)
   ```
The output table:

|     Metric     |  Value   |
|----------------|---------:|
|    Avg. IoU    |  0.9548  |
|       TP       |  9.0000  |
|   Precision    |   0.4500 |
|     Recall     |   0.9000 |
|       F1       |   0.6000 |
|  Uncertainty   | 197.0125 |
| Total detected | 20.0000  |
| Total objects  | 10.0000  |

### RQ2: Is there any correlation between prediction surface uncertainty and the object detection performance of DNN models? 
This RQ tests the hypothesis that the prediction performance of a DNN model decreases with the increase in the uncertainty in images used for object detection. 

   ```python
    import util
    image_path = 'images/Stanford/'
    image_type = 'jpg'
    # Dropout layers are activated with p=0.3
    mc_dropout_model = util.get_model(0.3, mc_dropout=True)

    experiment_file = 'experiments_results_for_rq2.csv'
    f_out = open(experiment_file,'a')

    # Conduct 1000 experiments
    for _ in tqdm(range(1000), 'Overall experiments'):
        avg_iou, TP,precision,recall,f1, \
            uncertainty,total_detected,\
                total_objects = util.get_evaluation_result(tmp_model,
                                                           image_path,
                                                           image_type,
                                                           T=30)
        
        f_out.write(str(uncertainty) + '\t' + str(avg_iou) + '\n')
    f_out.close()
    util.plot_area_unc_vs_area(experiment_file)
   ```
![Output](https://raw.githubusercontent.com/Simula-COMPLEX/pure/main/desc_images/rq2.png)

## People
* Ferhat Ozgur Catak https://www.simula.no/people/ozgur
* Tao Yue https://www.simula.no/people/tao
* Shaukat Ali https://www.simula.no/people/shaukat

## Paper
F.O. Catak, T. Yue and S. Ali. Prediction Surface Uncertainty Quantification in Object Detection Models for Autonomous Driving. *2021 IEEE International Conference On Artificial Intelligence Testing (AITest)*, 2021.