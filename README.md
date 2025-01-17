# Waste Management Classification

Contributor: Stephen Williams

## Goal 
This project aims to classify images into organic and non-organic categories.  I have concerned myself with this to compose a means to efficiently identify and ultimately automate waste management.  Currently, the burden of sorting waste is left to individual households and the like.  Given their unreliable temperaments, it would be preferable to circumvent the human element in the waste management process.

## Contents 
  * [Architectures](https://github.com/smw150430/Waste-Management-Classification/tree/master/Architectures): json files of the convolutional neural networks that were built and trained with the exception of the final, hyperparameter tuned model since there is a storage size constraint 
  * [DATASET_2](https://github.com/smw150430/Waste-Management-Classification/tree/master/DATASET_2): compressed files of images (25081)  
  * [Images](https://github.com/smw150430/Waste-Management-Classification/tree/master/Images): example images and confusion matrices generated from the model predictions  
  * [Waste Management Classification.pdf](https://github.com/smw150430/Waste-Management-Classification/blob/master/Waste%20Management%20Classification.pdf): pdf of presentation slides
  * [working_notebook.ipynb](https://github.com/smw150430/Waste-Management-Classification/blob/master/working_notebook.ipynb): the notebook used to build and evaluate the models  
  * [ktrain.ipynb](https://github.com/smw150430/Waste-Management-Classification/blob/master/ktrain.ipynb): the notebook used for hyperparameter tuning and to obtain the best performing model

## Technologies Leveraged  
  * Cloud  
    * Google Cloud Platform 
  * Python  
    * Keras  
    * ktrain
    * Matplotlib  
    * NumPy  
    * pandas  
    * Scikit-learn  
    * seaborn  
    * TensorFlow  