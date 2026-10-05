import numpy as np
import sys
import csv
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn import svm

from sklearn.metrics import f1_score, accuracy_score


startTime = time.perf_counter()
chemIndex = int(sys.argv[1])


xVal = []

for batchNum in range(44):
	for sampleNum in range(1,21):
		try:
			with open("experimentalCsvSamples/NMRDec_Batch"+str(batchNum)+"_Sample"+str(sampleNum)+".csv",'r') as f:
				reader=csv.reader(f)
				data = list(reader)
				f.close()
			experimentalData = data

			if(experimentalData[0][0][0]=="["):
				experimentalData[0][0]=experimentalData[0][0][1:]
			if(experimentalData[-1][-1][-1]=="]"):
				experimentalData[0][-1]=experimentalData[0][-1][:-1]
			xVal.append(experimentalData[0])
		except:
            		print("file does not exist")

print("length of exp samples: ",len(xVal))
xVal = np.array(xVal, dtype=float)

with open("experimentalCsvSamples/experimentalSamplesExpectedValues.csv", 'r') as f:
	reader = csv.reader(f)
	data = list(reader)
	f.close

experimentalSampleLabelSet = np.array(data,dtype=float)


yVal = []

for sampleNum in range(len(experimentalSampleLabelSet)):
	if(sampleNum==17):
		continue
	if(experimentalSampleLabelSet[sampleNum][chemIndex]>0):
		yVal.append(1)
	else:
		yVal.append(0)

yVal = np.array(yVal,dtype=float)


# y = y.ravel()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(xVal, yVal, train_size=0.7, shuffle=True)

scaler = StandardScaler()

scaler.fit(X_train_raw)

X_train = scaler.transform(X_train_raw)

X_test = scaler.transform(X_test_raw)


#y_test=np.transpose(np.transpose(y_test)[chemIndex]).ravel()

#y_train=np.transpose(np.transpose(y_train)[chemIndex]).ravel()

print("y_train shape: ",np.shape(y_train))


X_train = X_train[:int(sys.argv[2])*50]
y_train = y_train[:int(sys.argv[2])*50]

print("y_train shape: ",np.shape(y_train))
print("X_train.shape: ",X_train.shape)

print("positive class: ", np.sum(y_train))


y_train = y_train.ravel()



#X_train = xValTrain
#y_train = yValTrain

C_range = np.logspace(0,10,10)

degree_range = (np.linspace(2,10,num=10)).astype(int)


print("degree range: ",degree_range)

kernels = ('linear','rbf','poly')
#kernels = ('linear','poly')

parameters = {'C':C_range, 'kernel':kernels,'degree':degree_range}


svc = svm.SVC()
clf = GridSearchCV(svc,parameters)

print("X_test: ",X_test.shape)
print("y_test: ",y_test.shape)

clf_fit = clf.fit(X_train, y_train)

grid = clf

#for key in clf.cv_results_.keys():
#	print("key: ",key)
#	print("value: ",clf.cv_results_[key])

testPrediction=grid.predict(X_test)
trainPrediction=grid.predict(X_train)
testAcc =  accuracy_score(y_test,testPrediction)
trainAcc = accuracy_score(y_train,trainPrediction)
#valPred = []

#for x in xVal:
#	valPred.append(int(grid.predict(x.reshape(1, -1))))
print("best fittor: ",clf_fit.best_estimator_)
print(clf_fit.best_estimator_)
print("pred: ",testPrediction)
print("y_Test: ",y_test)
print("SVM")
print("testAcc: ",testAcc)
print("test f1 score: ",f1_score(y_test, testPrediction, average="macro"))
print("trainAcc: ",trainAcc)
print("train f1 score: ",f1_score(y_train, trainPrediction, average="macro"))
print("N/A")
print(chemIndex)
endTime = time.perf_counter()
print(endTime-startTime)
