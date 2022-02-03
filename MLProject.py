
import sklearn as sk
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_excel('FinalDataset.xlsx');

cul = ['Depth', 'Oil Gravity', 'Viscosity',
       'Permeability', 'Porosity', 'Temperature']
data = data.drop_duplicates()
LabelClasses = data['Project Type'].unique()

index = data['Project Type'].factorize()[1]
data['Project Type'] = data['Project Type'].factorize()[0]

label = data['Project Type']

featurs=data[cul]


fig = plt.figure(figsize = (10, 5))
 
plt.bar(cul,featurs.mean(), color ='maroon',
        width = 0.4)
plt.savefig('figurs/MeanHist.png')
plt.close()

for i in cul:
    featurs[i] = featurs[i].fillna(value=np.mean(featurs[i]))
    featurs[i] = (featurs[i]-featurs[i].mean())/featurs[i].std()


featurs.boxplot(cul,figsize=[10,10])
plt.savefig('figurs/BoxPlot.png')
featurs.hist(cul,figsize=[10,10])
plt.savefig('figurs/Hist.png')
plt.close()
sns.pairplot(featurs)
plt.savefig('figurs/pairplot.png')
plt.close()

sns.heatmap(featurs)
plt.savefig('figurs/AllDataHeatMap.png')
plt.close()
sns.heatmap(featurs[0:9])
plt.savefig('figurs/heatmap8data.png')
plt.close()

from sklearn.model_selection import train_test_split
import sklearn.metrics as met
Selected =  featurs.T[[59,335,386,366]].T
SelectedTarget = label.T[[59,335,386,366]].T
xtrain,xtest,ytrain,ytest = train_test_split(featurs,label,test_size=0.2)
xtest = xtest.append(Selected)
ytest = ytest.append(SelectedTarget)
##################################################
#DecisionTreeClassifier    supervised
from sklearn import tree
model = tree.DecisionTreeClassifier()
model = model.fit(xtrain,ytrain)
ypredictSellectedTree = model.predict(Selected)
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model, 
                   feature_names=cul,  
                   class_names=LabelClasses,
                   filled=True)
plt.savefig('figurs/Tree.png')
print('*********************************')
print('score in Decision Tree: ',model.score(xtest,ytest))
#######################################################
#KNN   Supervised
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=30, metric='euclidean',p=2)
model = model.fit(xtrain,ytrain)
ypredictSellectedKNN = model.predict(Selected)
print('*********************************')
print('score in KNN: ',model.score(xtest,ytest))
training_accuracy = []
test_accuracy = []
neighbors_settings = range(3, 50)
for n_neighbors in neighbors_settings:
  model = KNeighborsClassifier(n_neighbors=n_neighbors)
  model.fit(xtrain, ytrain)
  training_accuracy.append(model.score(xtrain, ytrain))
  test_accuracy.append(model.score(xtest, ytest))
plt.figure()
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('figurs/KNN.png')
plt.close()
################################################################
# XGboost    Supervised
#pip install xgboost
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(xtrain,ytrain)
ypredictSellectedXGB = model.predict(Selected)
print('*********************************')
print('score wth XGBoost: ',model.score(xtest,ytest))
############################################################
#DBscan   Unsupervised
from sklearn.cluster import DBSCAN
model = DBSCAN(eps = 1.4, min_samples=10)
model.fit(xtrain)
clusters = model.labels_
print('****************************\n','DBScan Cluster: ',clusters)
ypredict = model.fit_predict(xtest,ytest)
ypredictSellectedDBS = model.fit_predict(Selected)

#########################################################
#LGBM  Supervised
from lightgbm import LGBMClassifier

model = LGBMClassifier(objective='multiclass', random_state=5)
model.fit(xtrain,ytrain)
ypredictSellectedLGB = model.predict(Selected)
print('*********************************')
print('score with LGBM: ',model.score(xtest,ytest))
########################################################
# mean shift     unsupervised
from sklearn.cluster import MeanShift
model = MeanShift()
model.fit(xtrain)
ypredictSellectedMeanShift = model.predict(Selected)
ypre = model.fit_predict(xtest,ytest)
print('*********************************')
print('clusters with MeanShift: ',ypre )
###############################################################
#GNB  Supervised
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(xtrain,ytrain)
ypredictSellectedGNB = model.predict(Selected)
print('*********************************')
print('score wth GNB: ',model.score(xtest,ytest))
#############################################################
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(23)
model = model.fit(xtrain)
ypre = model.fit_predict(xtest,ytest)
print('*********************************')
print('clustering with AgglomerativeClustering: ',ypre)
##############################################################