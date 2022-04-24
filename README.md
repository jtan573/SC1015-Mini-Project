# SC1015-Mini-Project
Stroke Predictor for Mini-Project 
![Screenshot (300)](https://user-images.githubusercontent.com/103939428/164981316-224d1e27-5fe6-45f2-be0e-b43e9487a2ee.png)

## About
This is a Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence) which focuses on stroke prediction from our dataset. For detailed walkthrough, please view the source code in order from:
1. Data Extraction and Cleaning
2. Exploratory Data Analysis
3. Data Splitting and Resampling
4. Logistic Regression
5. Classification Trees

## Contributors
- @jtan573 (Data preparation & cleaning, Data Splitting and resampling)
- @i-anS (Logistic Regression, Classification Trees)
- @geraldd_d (Exploratory Data Analysis)

## Problem Definition
**The problem we are trying to solve is “Are we able to determine if an individual is likely to suffer from stroke given his/her background?”**

**Context:** 
Disease prediction has the potential to benefit stakeholders such as the government and insurance companies. Furthermore, healthcare service providers can also shift to more preventive care, not only improving patients’ quality of life but also potentially saving money in the healthcare system.

**Aim:** 
To examine two different machine learning models to predict whether a person is likely to get a stroke based on his/her background and lifestyle factors. We do this by applying supervised learning methods for stroke prediction by interpreting the dataset which include data of different types.

## Dataset Used
**Title:** Stroke Prediction Dataset

Download the dataset needed for running the code here: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

## Models Used
- SMOTE, SMOTEENN and ADASYN Oversampling techniques
- Logistic Regression
- Decision Trees

## Requirements
If your python do not already come with pip, then install using the command: `sudo apt-get install python-pip`

Pre-installed (Comes with Python)
- Pandas
- Seaborn
- Scikit-learn
- matplotlib

May require installation:
- Imblearn (Download using the command: `!pip install imblearn`)
- Matplotlib (Download using the command: `pip install matplotlib`)

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

## References
- https://elitedatascience.com/imbalanced-classes 
- https://www.kaggle.com/code/rafjaa/resampling-strategies-for-imbalanced-datasets/notebook
- https://www.analyticsvidhya.com/blog/2020/11/popular-classification-models-for-machine-learning/
- https://www.analyticsvidhya.com/blog/2021/07/an-introduction-to-logistic-regression/
- https://www.analyticsvidhya.com/blog/2020/12/decluttering-the-performance-measures-of-classification-models/
- https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/
- https://www.analyticsvidhya.com/blog/2020/10/how-to-choose-evaluation-metrics-for-classification-model/
- https://wiki.pathmind.com/accuracy-precision-recall-f1#:~:text=That%20is%2C%20a%20good%20F1,total%20failure%20when%20it's%200%20. 
