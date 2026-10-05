from sklearn.model_selection import train_test_split
import math
import tqdm
import numpy as np
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import csv

def example(rank, world_size,):
    batchSize = int(sys.argv[4])
    startTime = time.perf_counter()
    datasetSize = (int(sys.argv[3]) * 500) // world_size
    print("dataset Size: ",datasetSize)
    chemIndex = int(sys.argv[1])

    with open("concSampleArray.csv",'r') as f:
        reader=csv.reader(f)
        data = list(reader)

    sampleArray = data

    labelSet = []
    databaseSize = 500000
    sampleIndices = list(range(databaseSize))
    sampleIndices = sampleIndices[int(rank*(databaseSize//world_size)):int((rank+1)*(databaseSize//world_size))]
#    sampleIndices= sampleIndices[((datasetSize//world_size)*rank):((datasetSize//world_size)*(rank+1))]

    rng = np.random.default_rng()


    rng.shuffle(sampleIndices)

    sampleIndices = sampleIndices[:datasetSize]

    sampleArray[0][0]=sampleArray[0][0][2:]
    sampleArray[0][-1]=sampleArray[0][-1][:-2]
    numOfChems=5


    for sampleNum in sampleIndices:
        chemSamples = sampleArray[0][chemIndex+(sampleNum*numOfChems)]


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

        # Read data
        # loss function and optimize

    endTime = time.perf_counter()
    print(f"Performance Dataload time: {endTime - startTime:.6f} seconds")
    

    l2Reg = 0.00000001 * int(sys.argv[5])
    print("l2Reg: ",l2Reg)
    mom = 0
    learning_rate=float(sys.argv[6]) * world_size
    print("learning_rate: ",learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chemIndex = int(sys.argv[1])
    print()
    print("dataset Size: ",datasetSize)
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

    X = np.array(Dataset, dtype=float)

    y= labelSet


    X_train = X[:(len(X) - len(X)%batchSize)]
    y_train = y[:(len(y) - len(y)%batchSize)]
    xVal = xVal[:(len(xVal) - len(xVal)%batchSize)]
    yVal = yVal[:(len(yVal) - len(yVal)%batchSize)]


    X_train = torch.from_numpy(X_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    X_test = torch.from_numpy(xVal).type(torch.float)
    y_test = torch.from_numpy(yVal).type(torch.float)

    
    epochs = int(sys.argv[2])
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model

    model = nn.Sequential(
        nn.LayerNorm(2000),
        nn.Conv1d(1, 16, (6),stride=3),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(16, 32, (8),stride=4),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 64, (8),stride=4),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(64, 64, (4),stride=4),
        nn.Flatten(start_dim=1),
        nn.LayerNorm(64),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)

        )

    model = model.to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, weight_decay=l2Reg, foreach=True, fused=False)


    history = []
#    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
# forward pass
    startTime = time.perf_counter()

    for epoch in range(epochs):

                # take a batch
        for x in range(math.ceil(X_train.shape[0]/batchSize)):
            batchStart = x*batchSize
            batchEnd = (x+1)*batchSize
            if(batchEnd>X_train.shape[0]):
                batchEnd==X_train.shape[0]
            labels = y_train[batchStart:batchEnd].to(rank)

            # forward pass
            y_pred = ddp_model((X_train[batchStart:batchEnd].unsqueeze(1)).to(rank))
            loss = loss_fn(y_pred, labels)

                # backward pass

                #optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # update weights
            optimizer.step()
            # print progress
        model.eval()
        y_pred=[]
        train_pred=[]
        for index in range(math.ceil(X_test.shape[0]/batchSize)):
            batchStart = index*batchSize
            batchEnd = (index+1)*batchSize
            if(batchEnd>X_test.shape[0]):
                batchEnd==X_test.shape[0]-1
            y_pred.append(ddp_model((X_test[batchStart:batchEnd].unsqueeze(1)).to(rank)).cpu().detach().numpy())

        for index in range(math.ceil(X_train.shape[0]/batchSize)):
            batchStart = index*batchSize
            batchEnd = (index+1)*batchSize
            if(batchEnd>X_train.shape[0]):
                batchEnd==X_train.shape[0]-1
            train_pred.append(ddp_model((X_train[batchStart:batchEnd].unsqueeze(1)).to(rank)).cpu().detach().numpy())

        y_pred=np.array(y_pred).flatten()
        train_pred=np.array(train_pred).flatten()
        y_pred = torch.from_numpy(y_pred).type(torch.float)
        train_pred = torch.from_numpy(train_pred).type(torch.float)
        trainLoss = loss_fn(train_pred, y_train)
        testLoss = loss_fn(y_pred, y_test)

        valMAPE = []
        trainMAPE = []

        for index in range(len(y_test)):
           if(float(y_test[index])==0.0):
               valMAPE.append(0)
           else:
               valMAPE.append( abs( float(y_test[index]) - float(y_pred[index]) ) / abs(float(y_test[index]) ) )
        valMAPE = sum(valMAPE)/len(valMAPE)

        for index in range(len(y_train)):
           if(float(y_train[index])==0.0):
              trainMAPE.append(0)
           else:
              trainMAPE.append( abs( float(y_train[index]) - float(train_pred[index]) ) / abs(float(y_train[index]) ) )
        trainMAPE = sum(trainMAPE)/len(trainMAPE)
        history.append((valMAPE,epoch))

        if(epoch%1==0):

            print("epoch: ",epoch,"trainRMSE: %.30f" % np.sqrt(float(trainLoss))," trainMAPE: %6f" % trainMAPE,"testRMSE: %.30f" % np.sqrt(float(testLoss)), " VAL MAPE: %6F" % (valMAPE) )
    endTime = time.perf_counter()
    print(ddp_model)
    print(f"Performance time: {endTime - startTime:.6f} seconds")
    print("Best MAPE: ", min(history)[0])
    print("Epoch: ", min(history)[1])

def main():
    world_size = torch.cuda.device_count()
    



    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()

