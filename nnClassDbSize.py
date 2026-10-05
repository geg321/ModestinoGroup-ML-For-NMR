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


def example(rank, world_size):
    datasetSize = 0
        # Read data
        # loss function and optimize
    l2Reg = 0.0000
    mom = 0
    printEpoch = 1
    learning_rate= float(sys.argv[3])
    print("learning_rate: ",learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    startTime = time.perf_counter()

    chemIndex = int(sys.argv[1])
    print()
    print("dataset Size: ",datasetSize)
    print("chemIndex: ",chemIndex)
    X = []
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
                            X.append(experimentalData[0])
                    except:
                            print("file does not exist")

    X = np.array(X, dtype=float)

    with open("experimentalCsvSamples/experimentalSamplesExpectedValues.csv", 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            f.close
    experimentalSampleLabelSet = np.array(data,dtype=float)

    y = []

    for sampleNum in range(len(experimentalSampleLabelSet)):
            if(sampleNum==17):
                    continue
            if(experimentalSampleLabelSet[sampleNum][chemIndex]>0):
                    y.append(experimentalSampleLabelSet[sampleNum][chemIndex])
            else:
                    y.append(0)

    y = np.array(y,dtype=float)


    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    X_train = X_train[:int(sys.argv[5])*50]
    y_train = y_train[:int(sys.argv[5])*50]

    print("X_train shape: ",X_train.shape)
    print("y_train shape: ",y_train.shape)

    X_train = X_train[int(rank*(len(X_train)//world_size)):int((rank+1)*(len(X_train)//world_size))]
    y_train = y_train[int(rank*(len(y_train)//world_size)):int((rank+1)*(len(y_train)//world_size))]
    
    

    X_train = torch.from_numpy(X_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    X_test = torch.from_numpy(X_test).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.float)
    
    epochs = int(sys.argv[2])
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Sequential(
        nn.LayerNorm(2000),
    )

    #arv[4] is number of hidden layers the results presented used 8 hidden layers
    for layer in range(int(sys.argv[4])):
        model.append(nn.Linear(2000,2000))
        model.append(nn.ReLU())
    model.append(nn.Linear(2000,1))

    model = model.to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, weight_decay=l2Reg, foreach=True, fused=False)

    history = []
    for epoch in range(epochs):

        labels = y_train.to(rank)
        X_train = X_train.to(rank)
            # forward pass
        y_pred = ddp_model(X_train).squeeze()
        loss = loss_fn(y_pred, labels)

        MAE = nn.L1Loss()
                # backward pass

                #optimizer.zero_grad()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

            # update weights
        optimizer.step()
            # print progress

        model.eval()
        train_pred = torch.round(torch.sigmoid(ddp_model(X_train).squeeze())).to(rank)         
        y_pred = torch.round(torch.sigmoid(ddp_model(X_test).squeeze())).to(rank)         
        y_test=y_test.to(rank)
        y_train=y_train.to(rank)
        train_acc = accuracy_fn(labels, train_pred)
        test_acc = accuracy_fn(y_test, y_pred)
        test_f1=binary_f1_score(y_pred, torch.ceil(y_test))
        train_f1=binary_f1_score(train_pred, torch.ceil(y_train))
        history.append((train_acc,train_f1,test_acc,test_f1,epoch))
        
        if(epoch%1==0 and printEpoch):

            print("epoch: ",epoch,"train Loss: %.30f" % np.sqrt(float(loss))," trainAcc: %6f" % (train_acc), " VAL ACC: %6F" % (test_acc) )
    endTime = time.perf_counter()
    print(f"Performance time: {endTime - startTime:.6f} seconds")
    print(model)
    print("Best train Acc: ", max(history, key=lambda x:x[2])[0])
    print("Best train F1: ", max(history, key=lambda x:x[2])[1])
    print("Best test Acc: ", max(history, key=lambda x:x[2])[2])
    print("Best test F1: ", max(history, key=lambda x:x[2])[3])
    print("Epoch: ", max(history, key=lambda x:x[2])[4])

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
    os.environ["MASTER_PORT"] = str(29501 + int(sys.argv[5]))
    main()

