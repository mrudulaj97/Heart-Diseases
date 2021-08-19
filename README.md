# Cardiovascular disease classification and analysis of the risk factors using Machine Learning models


## Introduction

Cardio and vascular diseases are among the leading causes of mortality in the world. 
Although most people know that the heart must be properly cared for, heart diseases had risen steadily over the last century and has become the leading cause of death for people in the United States.
It is predicted that the number of annual deaths will rise to 25 million people by 2030. Medical diagnosis is an important yet complicated task that needs to be executed accurately and efficiently. 
Heart disease classification systems can help detect anomalies in the patient’s readings and warn the doctors. Association rule mining and classification are two major techniques of data mining.

### What is Association rule mining?
Association rule mining is an unsupervised learning method for discovering interesting patterns and their association in large data bases. Whereas classification is a supervised learning method used to find class label for unknown sample. A lot of focus is been devoted towards classification of the heart condition but a very little research is been done towards identifying the risk factors of individual patients and finding the association between the parameters. This research is aimed as improving the classification accuracy while analyzing the associated risk factors.

## Data Set
The heart disease data sets considered in the study are obtained from a UCI machine learning benchmark repository and IEEE database.
https://archive.ics.uci.edu/ml/datasets/heart+disease
https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive
 
### About the attributes:
The following are the parameters depicted in the dataset. In order to find out the role of each parameter in causing heart attack, we first identify the parameters which are abnormal. Then, the classification accuracy of each individual parameter is calculated. More the accuracy, more the role of the parameter in causing the heart attack. Then the rules of association are applied on the data to find out the combination of parameters which are linked with one another and thereby a find a combination of features that are more responsible for causing heat attacks.
age: age in years<br/>
sex: sex (1 = male; 0 = female) <br/>
cp<br/>
Value 1: typical angina<br/>
Value 2: atypical angina<br/>
Value 3: non-anginal pain<br/>
Value 4: asymptomatic<br/>
considering 1,2,4 as danger, 3 is normal<br/>
trestbps: resting blood pressure (in mm Hg on admission to the hospital)<br/>
 greater than 120 abnormal <br/>
chol<br/>
  greater than200 abnormal<br/>
Fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)<br/>
restecg: resting electrocardiographic results<br/>
Value 0: normal<br/>
Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)<br/>
Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria <br/>
Thalach: maximum heart rate achieved<br/>
greater than 100 abnormal<br/>
Exang: exercise induced angina (1 = yes; 0 = no)<br/>
oldpeak = ST depression induced by exercise relative to rest<br/>
slope: the slope of the peak exercise ST segment<br/>
-- Value 1: upsloping<br/>
-- Value 2: flat<br/>
-- Value 3: downsloping <br/>
Ca: number of major vessels (0-3) colored by fluoroscopy<br/>
thal: 3 = normal; 6 = fixed defect; 7 = reversible defect<br/>
Target Variable: diagnosis of heart disease (angiographic disease status)<br/>
-- Value 0: < 50% diameter narrowing<br/>
-- Value 1: > 50% diameter narrowing <br/>
 
 
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

## Visualizations or the Insights drawn from the data for further analysis:

![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/Figure_1.png)
The plot shows that the heart conditions increased drastically in the age groups of 55-63. In detail, people between the age 55 to 63 are more prone to heart diseases. People with ages 57, 58, 59 has the highest count of heart disease.

![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/Figure_2.png)
The figure displays that The number of heart conditions in Male are considerably higher than female. As can be seen from the bar graph that Male are much prone to heart diseases or  conditions than the females.

![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/Figure_3.png)
From this bar plot it can be seen that resting blood pressure was recorded abnormally for more number of patients as 120,130,140,150. So to be clear, at these given heart rates, patient's resting blood pressure was abnormal as the chances of getting a heart disease are higher at these resting rates.

![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/Figure_4.png)
From the above plot, it can be seen that More heart conditions occurred in patients with Chol > 230. Which means people with Cholesterol levels greater than 230 has high chances of getting heart diseases when compared to people  with cholesterol levels less than 230.




## Building Machine Learning Models:
I have used four machine learning algorithms for my analysis.<br/> 
The Models are:<br/>
1). SVM<br/>
2). Logistic Regression<br/>
3). Decision Tree<br/>
4). Random Forest<br/>


![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_22_0.png)

The figure shows the confusion matrix for the support vector machine model. A total of 61 data samples have been given for testing. Out of which 49 samples have been correctly identified. The values of TP, TN, FP, FN are 32, 17, 3 and 9 respectively. The overall accuracy obtained is 80.32.


![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_27_0.png)

The figure shows the confusion matrix for the Logistic Regression model. A total of 61 data samples have been given for testing. Out of which 49 samples have been correctly identified. The values of TP, TN, FP, FN are 32, 17, 3 and 9 respectively. The overall accuracy obtained is 80.32.


![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_16_0.png)

The figure shows the confusion matrix for the Decision Tree model. A total of 61 data samples have been given for testing. Out of which 49 samples have been correctly identified. The values of TP, TN, FP, FN are 29, 18, 6 and 8 respectively. The overall accuracy obtained is 80.32.


![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_32_0.png)

The figure shows the confusion matrix for the Random Forest model. A total of 61 data samples have been given for testing. Out of which 49 samples have been correctly identified. The values of TP, TN, FP, FN are 28, 17, 7 and 9 respectively. The overall accuracy obtained is 75.40.

### The Further analysis for this will be Association rule mining.

#### Association Rule Mining:
Association rule mining is a procedure which is meant to find frequent patterns, correlations, associations, or causal structures from data sets found in various kinds of databases such as relational databases, transactional databases, and other forms of data repositories.  And also Association Rule Mining is a Data Mining technique that finds patterns in data. The patterns found by Association Rule Mining represent relationships between items.

In My analysis, Association rules can be applied to extract the relationship in the data attributes. Here the parameters responsible for heart attack are studied in detail. The risk factor analysis on the attributed can be performed with support and confidence values. Support describes the probability that both the body and the head of the rule are in of a transaction as measured by all transactions. The confidence of a rule is defined as the proportion of transactions that Rule body and rule header contain the number of transactions that meet the rulebook. The experimental results show that the thalach is the parameter with highest association will other parameters in the abnormal range. The second and third highest parameters are found out as CP, Chol respectively.

These are the outputs when Association Rule Mining is applied to all the parameters used in the analysis. Here rule mining is applied to each parameter with all other parameters and the results can be observed below.

### Abnormal Association between cp and other parameters:
Support

![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_39_0.png)

Confidence

![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_40_0.png)

### Abnormal Association between trestbps and other parameters:
Support
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_42_0.png)
Confidence
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_43_0.png)

### Abnormal Association between Chol and other parameters:
Support
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_45_0.png)
Confidence
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_46_0.png)

### Abnormal Association between fbs and other parameters:
Support
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_48_0.png)
Confidence
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_49_0.png)

### Abnormal Association between restecg and other parameters:
Support
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_51_0.png)
Confidence
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_52_0.png)

### Abnormal Association between thalach and other parameters:
Support
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_54_0.png)
Confidence
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_55_0.png)

### Abnormal Association between exang and other parameters:
Support
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_57_0.png)
Confidence
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_58_0.png)

### Abnormal Association between ca and other parameters:
Support
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_60_0.png)
Confidence
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_61_0.png)

### Abnormal Association between thal and other parameters:
Support
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_63_0.png)
Confidence
![alt text](https://github.com/mrudulaj97/Heart-Diseases/blob/main/Association%20rule%20ops/output_64_0.png)

#### Here in the Association rule mining 
In the first figure, we can observe the abnormal association between cp and other parameters. The graph depicts the confidence value associated with the abnormal values of the cp with other parameters testbps, chol, fbs, resecg, thalach, exang, ca and thal. The highest association is found with thalach, chol and testbps in order. The least association is found with fbs. For patients with abnormal cp, the other parameters thalach, chol and testbps are also most abnormal. The patients with abnormal ca, thalach, testbps and chol have a very high risk of getting heart attack. Just like this, we can look at the abnormal association between all the parameters(each one), when compared with the all other parameters combined.

Support: It denotes the frequency of the rule within transactions. In this case, the support value indicates how frequently a parameter is becoming abnormal in heart disease patients. A high value means that the rule involves a great part of the database.

Confidence: It denotes the percentage of transactions containing A which contain also B. 
In this case, the confidence states how two parameters are linked with one another abnormally in causing a heart attack.
It is an estimation of conditioned probability.




## Conclusion

Heart disease prediction is a challenging task that is under research from many decades. There are several factors that cause heart attacks in patients, so these factors can be used to analyse and predict if a patient is having a risk of getting heart attack. Also the Machine learning models that were built will help predict if a patient is going to have a heart attack. The association analysis helps identify the most prominent risk factor in the patients that can cause heart attack. The analysis of the association can help doctors personalize the treatment based on the patient condition.



## References
https://archive.ics.uci.edu/ml/datasets/heart+disease </br>
https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive </br>
https://towardsdatascience.com/association-rules-2-aa9a77241654 </br>
https://www.researchgate.net/punlication/ </br>


