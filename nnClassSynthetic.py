from torcheval.metrics.functional import binary_f1_score
from sklearn.model_selection import train_test_split
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

def accuracy_fn(y_true, y_pred):
        correct = torch.eq(torch.ceil(y_true), y_pred).sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred)) * 100
        return acc


def example(rank, world_size,):
    datasetSize = (int(sys.argv[3]) * 1000)//(world_size)
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



    printEpoch = int(sys.argv[4])
    l2Reg = 0.0000
    mom = 0
    learning_rate=.001 * world_size
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



    X_train = X
    y_train = y

    
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
        nn.Linear(2000,2000),
        nn.ReLU(),
        nn.Linear(2000,2000),
        nn.ReLU(),
        nn.Linear(2000,2000),
        nn.ReLU(),
        nn.Linear(2000,2000),
        nn.ReLU(),
        nn.Linear(2000,1)
    )

    model = model.to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    batchSize = int(sys.argv[5])
    if(batchSize>X_train.shape[0]):
        batchSize=X_train.shape[0]
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, weight_decay=l2Reg, foreach=True, fused=False)

    history = []
    for epoch in range(epochs):
        for batchNum in range((X_train.shape[0])//batchSize):
            batchLabels = y_train[batchNum*batchSize:(batchNum+1)*batchSize]
            trainBatch = X_train[batchNum*batchSize:(batchNum+1)*batchSize]
            batchLabels = (batchLabels).to(rank)
            trainBatch = trainBatch.to(rank)
            # forward pass
            y_pred = ddp_model(trainBatch).squeeze()
            loss = loss_fn(y_pred, batchLabels)

            MAE = nn.L1Loss()
                # backward pass

                #optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            # update weights
            optimizer.step()
            # print progress

        model.eval()
        X_train = X_train.to(rank)
        labels = y_train.to(rank)
        train_pred = torch.round(torch.sigmoid(ddp_model(X_train).squeeze())).to(rank)         
        y_pred = torch.round(torch.sigmoid(ddp_model(X_test).squeeze())).to(rank)         
        y_test=(y_test).to(rank)
        train_acc = accuracy_fn(labels, train_pred)
        test_acc = accuracy_fn(y_test, y_pred)
        test_f1=binary_f1_score(y_pred, torch.ceil(y_test))
        history.append((test_acc,test_f1,epoch))

        if(epoch%1==0 and printEpoch):

            print("epoch: ",epoch," trainAcc: %6f" % (train_acc), " VAL ACC: %6F" % (test_acc), " VAL F1: %6F" %(test_f1) )
    endTime = time.perf_counter()
    print(f"Performance time: {endTime - startTime:.6f} seconds")
    print("Best Acc: ", max(history)[0])            
    print("Best F1: ", max(history)[1])            
    print("Epoch: ", max(history)[2])

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
    os.environ["MASTER_PORT"] = "29500"
    main()

