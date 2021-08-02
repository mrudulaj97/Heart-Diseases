import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)

# read the dataset file
df = pd.read_csv('data.csv', header = None)

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']

### 1 = male, 0 = female
df.isnull().sum()

df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['sex'] = df.sex.map({0: 'female', 1: 'male'})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())
df['sex'] = df.sex.map({'female': 0, 'male': 1})

################################## Plots

# distribution of target vs age 
sns.set_context("paper", font_scale = 1, rc = {"font.size": 10,"axes.titlesize": 10,"axes.labelsize": 10}) 
sns.catplot(kind = 'count', data = df, x = 'age', hue = 'target', order = df['age'].sort_values().unique())
plt.title('Variation of Age for each target class')
plt.show()

# distribution of target vs sex 
sns.set_context("paper", font_scale = 1, rc = {"font.size": 10,"axes.titlesize": 10,"axes.labelsize": 10}) 
sns.catplot(kind = 'count', data = df, x = 'sex', hue = 'target', order = df['sex'].sort_values().unique())
plt.title('Variation gender for each target class')
plt.show()

# distribution of target vs BP 
sns.set_context("paper", font_scale = 1, rc = {"font.size": 10,"axes.titlesize": 10,"axes.labelsize": 10}) 
sns.catplot(kind = 'count', data = df, x = 'trestbps', hue = 'target', order = df['trestbps'].sort_values().unique())
plt.title('Variation of resting blood pressure (mmHg) for each target class')
plt.show()

# distribution of target vs Chol 
sns.set_context("paper", font_scale = 1, rc = {"font.size": 10,"axes.titlesize": 10,"axes.labelsize": 10}) 
sns.catplot(kind = 'count', data = df, x = 'chol', hue = 'target', order = df['chol'].sort_values().unique())
plt.title('Variation of Cholectrol for each target class')
plt.show()

# distribution of target vs thalach 
sns.set_context("paper", font_scale = 1, rc = {"font.size": 10,"axes.titlesize": 10,"axes.labelsize": 10}) 
sns.catplot(kind = 'count', data = df, x = 'thalach', hue = 'target', order = df['thalach'].sort_values().unique())
plt.title('Variation of maximum heart rate achieved for each target class')
plt.show()

#trestbps
#chol
#Thalach


################################## data assignment
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#########################################   SVM   #############################################################

classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for svm = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

plot_confusion_matrix(classifier, X_test, y_test)  
plt.title("SVM confusion matrix")
plt.show()


matrix = confusion_matrix(y_train,y_pred_train, labels=[1,0])
print('Confusion matrix SVM: \n',matrix)

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(y_train,y_pred_train,labels=[1,0])
print('Classification report SVM : \n',matrix)

#########################################   Logistic Regression  #############################################################

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

plot_confusion_matrix(classifier, X_test, y_test)  
plt.title("Logistic Regression confusion matrix")
plt.show()

matrix = confusion_matrix(y_train,y_pred_train, labels=[1,0])
print('Confusion matrix Logistic Regression: \n',matrix)

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(y_train,y_pred_train,labels=[1,0])
print('Classification report Logistic Regression: \n',matrix)

#########################################   Decision Tree  #############################################################

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Decision Tree = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Decision Tree = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

plot_confusion_matrix(classifier, X_test, y_test)  
plt.title("Decision Tree confusion matrix")
plt.show()

matrix = confusion_matrix(y_train,y_pred_train, labels=[1,0])
print('Confusion matrix Decision Tree: \n',matrix)

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(y_train,y_pred_train,labels=[1,0])
print('Classification report Decision Tree: \n',matrix)


#########################################  Random Forest  #############################################################
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier = RandomForestClassifier(n_estimators = 10)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Random Forest = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Random Forest = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

plot_confusion_matrix(classifier, X_test, y_test)  
plt.title("Random Forest confusion matrix")
plt.show()

matrix = confusion_matrix(y_train,y_pred_train, labels=[1,0])
print('Confusion matrix Random Forest: \n',matrix)

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(y_train,y_pred_train,labels=[1,0])
print('Classification report Random Forest: \n',matrix)