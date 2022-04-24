# SC1015-Mini-Project
Stroke Predictor for Mini-Project 
![Screenshot (300)](https://user-images.githubusercontent.com/103939428/164981316-224d1e27-5fe6-45f2-be0e-b43e9487a2ee.png)

## About
This is a Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence) which focuses on stroke prediction from our dataset. In this repository, we are exploring oversampling methods and classification models for machine learning to create a model which solves the problem at hand. For detailed walkthrough, please view the source code in order from:
#### 1. Data Extraction and Cleaning
  - In this notebook we have successfully removed duplicate values, removed anomalies, dropped unnecessary columns, remove null values, and changed the object to categorical types. The more notable adjustment to the original dataset is the removal of the null values by changing them to meaningful values based on statistics such as median and probability. For more detailed explanation, please refer to our python notebook.
#### 2. Exploratory Data Analysis
  - EDA helped to reveal the relationships between the variables found in the dataset and stroke rate. We employed histograms, boxplots, countplots, and correlation maps to help visualise the relationships between our predictors and response data. 
#### 3. Data Splitting and Resampling
  - The cleaned_data dataset was split into train and test data, in the ratio of 0.8 : 0.2. By analysing the distribution of the two classes (stroke and no_stroke) in the train set, we noticed that our dataset faced the issue of class imbalance. Therefore, we employed the use of oversampling methods such as SMOTE, SMOTEENN, and ADASYN to increase the number of samples in the minority class to mitigate the problem of misclassification.
#### 4. Logistic Regression
  - Logistic regression is a supervised machine learning algorithm used to model the probability of a certain class or event. It is usually used for Binary classification problems. We have employed logistic regression to solve our classification problem. To evaluate the performance of the model, we used 5 different performance metrics, including accuracy, specificity, sensitivity, precision and F1-score. 
#### 5. Classification Trees
  - A Decision Tree is a supervised learning algorithm. Unlike other supervised learning algorithm, the decision tree algorithm can be used to solve regression and classification problems. The goal of using a Decision Tree is to create a training model that can be used to predict the class or value of the response variable by learning simple decision rules from training data. We created our second classification model using decision trees and evaluated its performance using the same performance metrics as mentioned above. 

## Problem Definition
- **“Are we able to determine if an individual is likely to suffer from stroke given his/her background?”**

**Context:** 
Disease prediction has the potential to benefit stakeholders such as the government and insurance companies. Furthermore, healthcare service providers can also shift to more preventive care, not only improving patients’ quality of life but also potentially saving money in the healthcare system.

**Aim:** 
To create classification models for prediction of stroke based on an individual's lifestyle and background factors by examining two different machine learning models. From which, compare and determine the better model using performance metrics such as precision, accuracy and specificity.

## Dataset Used
**Title:** Stroke Prediction Dataset

Download the dataset needed for running the code here: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

*We have also uploaded the dataset ('healthcare-dataset-stroke-data.csv') in the folder titled 'data'.*

## Models Used
- SMOTE, SMOTEENN and ADASYN Oversampling techniques
- Logistic Regression
- Decision Trees

## Conclusion
- Age, heart disease and hypertension has higher linear correlation with risk of getting a stroke
- Work type, residence type and gender has lower linear correlation with risk of getting a stroke
- SMOTE and SMOTEENN produced similar performance metrics
- Resampling unbalanced data improved model performance, as seen from increased performance metrics of precision, sensitivity and f1-score.
- Across the different oversampling methods, ADASYN performed best.
- Classification tree is a better model as compared to logistic regression.
- Achieving a classification accuracy of 88% using ADASYN and decision trees, our model can predict whether a person is likely to get a stroke.

## Learning Points
- Techniques to deal with null values in the raw dataset (E.g.: Removing/Filling up with mean or median/Making predictions based on probability)
- Recognise the significance of class imbalance in datasets
- Resampling Techniques including SMOTE, SMOTEENN and ADASYN
- Applying logistic regression and decision tree on real-life datasets
- Making use of performance metrics such as precision, specificity, sensitivity, and F1-score to determine the best model/oversampling method

## Contributors
- Jaslyn Tan (Data preparation & cleaning, Data Splitting and resampling)
- Sim Ian Leng (Logistic Regression, Classification Trees)
- Gerald Ong (Exploratory Data Analysis)

## References
- https://elitedatascience.com/imbalanced-classes 
- https://www.kaggle.com/code/rafjaa/resampling-strategies-for-imbalanced-datasets/notebook
- https://www.analyticsvidhya.com/blog/2020/11/popular-classification-models-for-machine-learning/
- https://www.analyticsvidhya.com/blog/2021/07/an-introduction-to-logistic-regression/
- https://www.analyticsvidhya.com/blog/2020/12/decluttering-the-performance-measures-of-classification-models/
- https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/
- https://www.analyticsvidhya.com/blog/2020/10/how-to-choose-evaluation-metrics-for-classification-model/
- https://wiki.pathmind.com/accuracy-precision-recall-f1#:~:text=That%20is%2C%20a%20good%20F1,total%20failure%20when%20it's%200%20. 
