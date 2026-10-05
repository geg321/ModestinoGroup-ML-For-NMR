from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import binary_f1_score
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np
import sys
import copy
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import csv

def accuracy_fn(y_true, y_pred):
        correct = torch.eq(torch.ceil(y_true), y_pred).sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred)) * 100
        return acc


def example(rank, world_size):
    startTime = time.perf_counter()

    datasetSize = int(sys.argv[3]) * 2000//world_size
    print("dataset Size: ",datasetSize)
    chemIndex = int(sys.argv[1])

    with open("concSampleArray.csv",'r') as f:
        reader=csv.reader(f)
        data = list(reader)

    sampleArray = data

    labelSet = []
    databaseSize = 500000
    sampleIndices = list(range(databaseSize))
#    sampleIndices= sampleIndices[((datasetSize//world_size)*rank):((datasetSize//world_size)*(rank+1))]
    sampleIndices = sampleIndices[int(rank*(databaseSize//world_size)):int((rank+1)*(databaseSize//world_size))]
    rng = np.random.default_rng(42)


    rng.shuffle(sampleIndices)

    sampleIndices = sampleIndices[:datasetSize]

    sampleArray[0][0]=sampleArray[0][0][2:]
    sampleArray[0][-1]=sampleArray[0][-1][:-2]
    numOfChems=5


    for sampleNum in sampleIndices:
        chemSamples = sampleArray[0][chemIndex+(sampleNum*numOfChems)]

#    chemSamples = sampleArray[0][sampleNum].split(", ")
        if(chemSamples[0:2]==" ["):
            chemSamples = float(chemSamples[2:])

        elif(chemSamples[-1]=="]"):
            chemSamples = float(chemSamples[:-1])
        else:
            chemSamples = float(chemSamples)
#        chemSamples[x] = .2*(int(chemSamples[x])/20)

        labelSet.append((chemSamples))

    labelSet = np.array(labelSet, dtype=float)


    Dataset = []

    for x in sampleIndices:

        with open('SyntheticSamples/'+str(x)+'.csv','r') as f:

            reader=csv.reader(f)

            data = list(reader)
            for x in range(len(data[0])):
                data[0][x]=float(data[0][x])
            Dataset.append(data[0])
    endTime = time.perf_counter()
    print(f"Performance time For Synthetic Data Loading: {endTime - startTime:.6f} seconds")



    l2Reg = float(sys.argv[6])
    mom = 0
    learning_rate=float(sys.argv[5]) * world_size
    print("learning_rate: ",learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    startTime = time.perf_counter()

    chemIndex = int(sys.argv[1])
    print()
    print("chemIndex: ",chemIndex)
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
                    yVal.append(experimentalSampleLabelSet[sampleNum][chemIndex])
            else:
                    yVal.append(0)

    yVal = np.array(yVal,dtype=float)

#    X = torch.from_numpy(X).type(torch.float)
#    y = torch.from_numpy(y).type(torch.float)




    X = np.array(Dataset, dtype=float)

    y= labelSet



    X_train = X
    y_train = y

    X_train = torch.from_numpy(X_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    X_test = torch.from_numpy(xVal).type(torch.float)
    y_test = torch.from_numpy(yVal).type(torch.float)


    trainingSet = TensorDataset(X_train, y_train)
    testSet = TensorDataset(X_test, y_test)
    batchSize = int(sys.argv[4])
    batchSize = datasetSize
    trainDataLoader = DataLoader(trainingSet, batch_size=batchSize, shuffle=False)
    testDataLoader = DataLoader(testSet, batch_size=batchSize, shuffle=False)

    
    epochs = int(sys.argv[2])
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Sequential(

        nn.Linear(2000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 2500),
        nn.ReLU(),
        nn.Linear(2500, 3000),
        nn.ReLU(),
        nn.Linear(3000, 2500),
        nn.ReLU(),
        nn.Linear(2500, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1500),
        nn.ReLU(),
        nn.Linear(1500, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 1)    
    )

    model = model.to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, weight_decay=l2Reg, foreach=True, fused=False)
    #batch_start = torch.arange(0, len(X_train), 1)
#    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
# forward pass
    history = []
    best_Loss = np.inf
    best_weights = None
    trainLossHistory = []
    testLossHistory = []
    for epoch in range(epochs):
        for xTrain, yTrain in trainDataLoader:

            labels = yTrain.to(rank)
            y_pred = ddp_model((xTrain.unsqueeze(1)).to(rank))
            loss = loss_fn(y_pred.squeeze(1).squeeze(1), labels)
                # backward pass

                #optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            loss.backward()

            # update weights
            optimizer.step()
            # print progress

        model.eval()

        y_pred=[]
        train_pred=[]

        for xTest,yTest in testDataLoader:
            pred = ddp_model((xTest.unsqueeze(1)).to(rank))
            y_pred.append(pred.cpu().detach().numpy())

        for xTrain,yTrain in trainDataLoader:
            pred = ddp_model((xTrain.unsqueeze(1)).to(rank))
            train_pred.append(pred.cpu().detach().numpy())


#        y_pred = ddp_model(X_test.unsqueeze(0)).cpu().detach().numpy()[0]
#        train_pred = ddp_model(X_train.unsqueeze(0)).cpu().detach().numpy()[0]

        y_pred = [elem for batch in y_pred for pred in batch for elem in pred]
#        y_pred = torch.from_numpy(np.array(y_pred, dtype=float)).type(torch.float)
        train_pred = [elem for batch in train_pred for pred in batch for elem in pred]
#        train_pred = torch.from_numpy(np.array(train_pred, dtype=float)).type(torch.float)
        y_pred = np.array(y_pred).flatten()
        y_pred = torch.from_numpy(y_pred).type(torch.float)
        train_pred = np.array(train_pred).flatten()

        train_pred = torch.from_numpy(train_pred).type(torch.float)
        trainLoss = loss_fn(train_pred, y_train)
        testLoss = loss_fn(y_pred, y_test)

        MAE = nn.L1Loss()
        trainMAE = MAE(train_pred, y_train)
        testMAE = MAE(y_pred, y_test)

        trainMAPE = []
        valMAPE = []

        for index in range(len(y_test)):
           if(float(y_test[index])==0.0):
               continue
               valMAPE.append(float(y_pred[index]))
           else:
               valMAPE.append( abs( float(y_test[index]) - float(y_pred[index]) ) / abs(float(y_test[index]) ) )
        valMAPE = sum(valMAPE)/len(valMAPE)

        for index in range(len(y_train)):
           if(float(y_train[index])==0.0):
              continue
              trainMAPE.append(float(train_pred[index]))
           else:
              trainMAPE.append( abs( float(y_train[index]) - float(train_pred[index]) ) / abs(float(y_train[index]) ) )
        trainMAPE = sum(trainMAPE)/len(trainMAPE)
        history.append((valMAPE,epoch))

        if testLoss < best_Loss:
             best_Loss = testLoss
             best_weights = copy.deepcopy(model.state_dict())

        trainLossHistory.append((np.sqrt(float(trainLoss)),epoch, trainMAE))
        testLossHistory.append((np.sqrt(float(testLoss)),epoch, testMAE))
        if(epoch%1==0):

            print("epoch: ",epoch,"trainRMSE: %.6f" % np.sqrt(float(trainLoss)),"trainMAE: %.6f" % (float(trainMAE))," trainMAPE: %6f" % trainMAPE, "testRMSE: %.6f" % np.sqrt(float(testLoss)),"testMAE: %.6f" % (float(testMAE))," VAL MAPE: %6F" % (valMAPE) )
    endTime = time.perf_counter()
    print(ddp_model)
    print(f"Performance time: {endTime - startTime:.6f} seconds")
    print("Best MAPE: ", min(history)[0])
    print("MAPE Epoch: ", min(history)[1])
    print("Best train RMSE: ", min(trainLossHistory)[0])
    print("Best train MAE: ", min(trainLossHistory)[2])
    print("train Epoch: ", min(trainLossHistory)[1])
    print("Best test RMSE: ", min(testLossHistory)[0])
    print("Best test MAE: ", min(testLossHistory)[2])
    print("test Epoch: ", min(testLossHistory)[1])
    diffList = []
    model.load_state_dict(best_weights)

    valMAPE = []
    for index in range(len(y_test)):
        if(float(y_test[index])==0.0):

            valMAPE.append(-1)
        else:
            valMAPE.append( abs( float(y_test[index]) - float(y_pred[index]) ) / abs(float(y_test[index]) ) )
    for index in range(len(y_pred)):
        diffList.append( ( index, float("{:.5f}".format(y_test[index])), float("{:.5f}".format(y_pred[index])), float("{:.5f}".format(abs(float(y_test[index] - y_pred[index])))), float("{:.5f}".format( float(valMAPE[index]) ))  ) )
    print("10 max Diff", sorted(diffList, key=lambda item: item[3], reverse=True)[:10])

def main():
    world_size = torch.cuda.device_count()
    print("Num of GPUs: ", world_size)


    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(29500 + int(sys.argv[3]))
    main()

