# Cardiovascular disease classification and analysis of the risk factors using Machine Learning models


## Introduction

Cardio and vascular diseases are among the leading causes of mortality in the world. 
Although most people know that the heart must be properly cared for, heart diseases had risen steadily over the last century and has become the leading cause of death for people in the United States.
It is predicted that the number of annual deaths will rise to 25 million people by 2030. Medical diagnosis is an important yet complicated task that needs to be executed accurately and efficiently. 
Heart disease classification systems can help detect anomalies in the patient’s readings and warn the doctors. Association rule mining and classification are two major techniques of data mining.

## Data Set
The heart disease data sets considered in the study are obtained from a UCI machine learning benchmark repository and IEEE database.
https://archive.ics.uci.edu/ml/datasets/heart+disease
https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive
 
### About the attributes:
The following are the parameters depicted in the dataset. In order to find out the role of each parameter in causing heart attack, we first identify the parameters which are abnormal. Then, the classification accuracy of each individual parameter is calculated. More the accuracy, more the role of the parameter in causing the heart attack. Then the rules of association are applied on the data to find out the combination of parameters which are linked with one another and thereby a find a combination of features that are more responsible for causing heat attacks.
--age: age in years
--sex: sex (1 = male; 0 = female) 
cp
Value 1: typical angina
Value 2: atypical angina
Value 3: non-anginal pain
Value 4: asymptomatic
considering 1,2,4 as danger, 3 is normal
trestbps: resting blood pressure (in mm Hg on admission to the hospital)
>120 abnormal 
chol
 >200 abnormal
Fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg: resting electrocardiographic results
Value 0: normal
Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria 
Thalach: maximum heart rate achieved
>100 abnormal
Exang: exercise induced angina (1 = yes; 0 = no)
oldpeak = ST depression induced by exercise relative to rest
slope: the slope of the peak exercise ST segment
-- Value 1: upsloping
-- Value 2: flat
-- Value 3: downsloping 
Ca: number of major vessels (0-3) colored by fluoroscopy
thal: 3 = normal; 6 = fixed defect; 7 = reversible defect
Target Variable: diagnosis of heart disease (angiographic disease status)
-- Value 0: < 50% diameter narrowing
-- Value 1: > 50% diameter narrowing 
 
 
 ## Hypothesis / Research Question(s)
1. 	What is the risk factor associated with different parameters?
2. 	How does the risk factor vary from person to person?
3. 	What is the association between the parameters?
4. 	How does a combination of parameters effect the risk factor for a person?
5. 	How accurately can the model classify the data?

## Implementation (Model)

### Classifiers under study:
Random Forest: Random forest is a flexible, easy to use machine learning algorithm that produces, even without hyper-parameter tuning, a great result most of the time. It is also one of the most used algorithms, because of its simplicity and diversity (it can be used for both classification and regression tasks). In this post we'll learn how the random forest algorithm works, how it differs from other algorithms and how to use it.

Decision Tree Algorithm:
Decision Tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning algorithms, the decision tree algorithm can be used for solving regression and classification problems too.
The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by learning simple decision rules inferred from prior data (training data).

Logistic regression: Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.

SVM: A support vector machine is a very important and versatile machine learning algorithm, it is capable of doing linear and nonlinear classification, regression and outlier detection. Support vector machines also known as SVM is another algorithm widely used by machine learning people for both classification as well as regression problems but is widely used for classification tasks. It is preferred over other classification algorithms because it uses less computation and gives notable accuracy. It is good because it gives reliable results even if there is less data. 

## Data Cleansing: 
Data cleansing is the process of identifying and resolving corrupt, inaccurate, or irrelevant data. This critical stage of data processing — also referred to as data scrubbing or data cleaning — boosts the consistency, reliability, and value of your company’s data.
Common inaccuracies in data include missing values, misplaced entries, and typographical errors. In some cases, data cleansing requires certain values to be filled in or corrected, while in other instances, the values will need to be removed altogether.
Also some of the things that has to be made sure before start of the analysis is to make sure the names of the attributes are correct and also to check for missing values in order to make sure the results or the analysis is not manipulated.

## Data Preparation:
Data preparation is the process of cleaning and converting raw data before processing and analysis. It's an important step before the process that often includes reformatting data, performing data corrections, and combining data sets to improve data. Data preparation is often a time-consuming process for data practitioners or business users. Still, it is needed as a requirement for putting data in context to transform it into insights and remove bias caused by poor data quality.
checked for missing values and found there are none for the selected dataset. And the dataset is labelled with no any empty or null entries.
As all the records were combined into a single a column separated by comma, I had to use delimiter to divide the values of the 13 attributes.
Check for spacings or errors in the values that contains decimals.


## References
https://archive.ics.uci.edu/ml/datasets/heart+disease
https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive
https://towardsdatascience.com/association-rules-2-aa9a77241654
https://www.researchgate.net/punlication/


