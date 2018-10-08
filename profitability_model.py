"""
Solution to Use Case 1 : Model to determine which customers are profitable

Target Variable is 'profit' depicts profitability of each customer

if 'profit' = 0, Customer is not profitable
if 'profit' = 1 , Customer is not profitable

Note: Profit values for training dataset has been calculated 'profit.py'

"""


# Importing the libraries
import numpy as np    #library that contains mathematical tools.
import matplotlib.pyplot as plt   #going to help us plot nice charts.
import pandas as pd #best library to import data sets and manage data sets.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Importing the dataset
train_dataset = pd.read_csv('train_dataset.csv')
test_dataset = pd.read_csv('test_dataset.csv')

# Splitting inddependent feartures from dependent feature
X_train = train_dataset.iloc[:, 3:18].values
y_train = train_dataset.iloc[:, 18].values

# Defining  X_test for this use case
X_test = test_dataset.iloc[:,2:].values 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#spot check algorithms
models=[]
models.append(('LR',LogisticRegression()))
models.append(('linearSVC',LinearSVC()))
models.append(('DecisionTree',DecisionTreeClassifier()))
models.append(('RandomForest',RandomForestClassifier(n_estimators=100)))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
results = []
seed = 2 # setting it to the same number each time guarantees that the algorithm will come up with the same results
scoring = 'accuracy'
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
# cv_result - accuracies
# names - model 
"""
Random forest has highest avaerage accuracy 

"""
"""
Calculating y_pred (value of coulmn 'profit' == 0 or 1) using difrrent classification model

""" 

#fitting logistic regression to the training set
lr = LogisticRegression(random_state = 0)
lr.fit(X_train,y_train)
profit_logistic = lr.predict(X_test)
lr.score(X_train, y_train)
acc_logistic = round(lr.score(X_train, y_train) * 100, 2)


#Fitting linear SVC to the training set
from sklearn.svm import SVC
l_svc = SVC(kernel = 'linear', random_state = 0)
l_svc.fit(X_train,y_train)
profit_linear_svc = l_svc.predict(X_test)
l_svc.score(X_train, y_train)
acc_l_svc = round(l_svc.score(X_train, y_train) * 100, 2)



#Fitting DecisionTreeclassifier to the training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)
profit_Decision = classifier.predict(X_test)
classifier.score(X_train, y_train)
acc_decision_tree = round(classifier.score(X_train, y_train) * 100, 2)


#Fitting RandomForestclassifier to the training set
from sklearn.ensemble import RandomForestClassifier
randomforestclassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
randomforestclassifier.fit(X_train,y_train)
profit_randomforest = randomforestclassifier.predict(X_test)
randomforestclassifier.score(X_train, y_train)
acc_random_forest = round(randomforestclassifier.score(X_train, y_train) * 100, 2)

#Fitting k-nn to the training set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2 )
knn.fit(X_train,y_train)
profit_knn = knn.predict(X_test)
knn.score(X_train, y_train)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)

#Fitting naive Bayes to the training set
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
profit_nb = nb.predict(X_test)
nb.score(X_train, y_train)
acc_nb = round(nb.score(X_train, y_train) * 100, 2)


#Fitting Kernel SVC to the training set
from sklearn.svm import SVC
k_svc = SVC(kernel = 'rbf', random_state = 0)
k_svc.fit(X_train,y_train)
profit_svm = k_svc.predict(X_test)
k_svc.score(X_train, y_train)
acc_k_svc = round(k_svc.score(X_train, y_train) * 100, 2)


"""
Random forest has highst accuracy.mean on k-fold validation

Using profit(Profitability) predicted by Random forest

"""

test_dataset['profit'] = profit_randomforest
test_dataset.to_csv('test_dataset_with_profit.csv', index=False)

df_final = test_dataset[['ids','profit']]
df_final.to_csv('test_id_profit.csv', index=False)












