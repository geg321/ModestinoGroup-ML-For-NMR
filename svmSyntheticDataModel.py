import numpy as np
import sys
import csv
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV 
from sklearn import svm

from sklearn.metrics import f1_score, accuracy_score

datasetSize=int(sys.argv[2]) * 100
print()
#print("datasetSize", datasetSize)
multiplier = 0
xVal = []
chemIndex = int(sys.argv[1])
#print("chemIndex: ", chemIndex)
for batchNum in range(0,44):
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
                        error = ("file does not exist")

#print("length of exp samples: ",len(xVal))
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




with open("concSampleArray.csv",'r') as f:
        reader=csv.reader(f)
        data = list(reader)

sampleArray = data

labelSet = []

#sampleArrays = sampleArray[0].split("]")
sampleArray[0][0]=sampleArray[0][0][2:]
sampleArray[0][-1]=sampleArray[0][-1][:-2]
numOfChems=5

for sampleNum in range(datasetSize):
    chemSamples = sampleArray[0][chemIndex+(sampleNum*numOfChems)]

#    chemSamples = sampleArray[0][sampleNum].split(", ")
    if(chemSamples[0:2]==" ["):
        chemSamples = float(chemSamples[2:])

    elif(chemSamples[-1]=="]"):
        chemSamples = float(chemSamples[:-1])
    else:
        chemSamples = float(chemSamples)
#        chemSamples[x] = .2*(int(chemSamples[x])/20)

    labelSet.append(int(0<chemSamples))


labelSet = np.array(labelSet, dtype=float)
#print(labelSet)

Dataset = []


for x in range(datasetSize):

    with open('SyntheticSamples/'+str(x)+'.csv','r') as f:

        reader=csv.reader(f)

        data = list(reader)
        for x in range(len(data[0])):
            data[0][x]=float(data[0][x])
        Dataset.append(data[0])


X = np.array(Dataset, dtype=float)

y= labelSet


#y=np.transpose(np.transpose(y)[chemIndex])


# y = y.ravel()
y=np.reshape(y,(-1))

#print("y.shape: ",y.shape)
#print("X.shape: ",X.shape)

#print(xVal.shape)


xValTest = xVal


yValTest = yVal


X_train = X

y_train = y

tempX_train = X_train
tempY_train = y_train
for x in range(multiplier):
   X_train = tempX_train + X_train

   y_train = tempY_train + y_train


#print("first X_train.shape: ",X_train.shape)

#X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

#y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1).to(device)


#print("X_train GPU Tensor?: ",X_train.is_cuda)
#print("y_train GPU Tensor?: ",y_train.is_cuda)
#print("X_test GPU Tensor?: ",X_test.is_cuda)
#print("y_test GPU Tensor?: ",y_test.is_cuda)
#print(torch.cuda.get_device_name(0))

#print("positive class: ", sum(y_train))

#y_train = y_train.ravel()

#X_train = xValTrain
#y_train = yValTrain
startTime = time.perf_counter()

C_range = np.linspace(0,20,num=20)

#print("C_range: ", C_range)

degree_range = (np.linspace(2,10,num=10)).astype(int)

#print("degree range: ",degree_range)

#kernels = ('linear','rbf','poly')
kernels = ('linear','poly')

parameters = {'C':C_range, 'kernel':kernels,'degree':degree_range}
svc = svm.SVC(kernel='poly',C=12,degree=2)
clf = svc
#clf = GridSearchCV(svc,parameters)
#print("xValTest: ", xValTest.shape)
#print("yValTest: ", yValTest.shape)

clf_fit = clf.fit(X_train, y_train)
#print("cv_results keys: ",sorted(clf.cv_results_.keys()))
#print("best param: ",clf.best_params_)
#print("best score: ",clf.best_score_)
#print("feature_names_in_: ", clf.feature_names_in_)
grid = clf


#valPred = []

#for x in xVal:
#       valPred.append(int(grid.predict(x.reshape(1, -1))))

prediction=grid.predict(xValTest)
acc =  accuracy_score(yValTest,prediction)
#print("xValTest: ",xValTest.shape)

valPred = grid.predict(xVal)
valAcc = accuracy_score(yVal, valPred)

print("datasetSize :",datasetSize)
#print("SVM")
print("chemIndex: ",chemIndex)
endTime = time.perf_counter()
#print(endTime-startTime)

#print("val Pred: ",valPred)
#print("yVal: ", yValTest)
print("val F1 score: ",f1_score(yValTest,valPred,average="macro"))
print("valAcc: ",valAcc)
