import copy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
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
import matplotlib.pyplot as plt
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import csv

def example(rank, world_size):
    datasetSize = 0
        # Read data
        # loss function and optimize
    l2Reg = .1**(float(sys.argv[5]))
    if(float(sys.argv[5])==0):
           l2Reg = 0
    print("l2Reg: ",l2Reg)
    batchSize = int(sys.argv[3])*1
    print("Batch Size: ",batchSize)
    mom = 0
    learning_rate=world_size * float(sys.argv[4])
    print("learning_rate: ", learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    startTime = time.perf_counter()

    chemIndex = int(sys.argv[1])
    print()
    print("dataset Size: ",datasetSize)
    print("chemIndex: ",chemIndex)
    X = []
    for batchNum in range(44):
            if(batchNum==2):
                    continue
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
            if((sampleNum==17) or (sampleNum>39 and sampleNum<60)):
                    continue
            if(experimentalSampleLabelSet[sampleNum][chemIndex]>0):
                    y.append(experimentalSampleLabelSet[sampleNum][chemIndex])
            else:
                    y.append(0)

    y = np.array(y,dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    X_train = X_train[: ( 50*int( sys.argv[7] ) ) ]
    y_train = y_train[: ( 50*int( sys.argv[7] ) ) ]
    print("Train DB Size: ",len(X_train))
    print("Test DB Size: ",len(X_test))

    X_train = X_train[int(rank*(len(X_train)//world_size)):int((rank+1)*(len(X_train)//world_size))]
    y_train = y_train[rank*(len(y_train)//world_size):(rank+1)*(len(y_train)//world_size)]
     

    X_train = torch.from_numpy(X_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    X_test = torch.from_numpy(X_test).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.float)
    trainingSet = TensorDataset(X_train, y_train)
    testSet = TensorDataset(X_test, y_test)

    print(bool(int(sys.argv[6])))

    shuffled=bool(int(sys.argv[6]))

    trainDataLoader = DataLoader(trainingSet, batch_size=batchSize, shuffle=shuffled)
    testDataLoader = DataLoader(testSet, batch_size=batchSize, shuffle=False)
    
    print(X_train.shape)
    epochs = int(sys.argv[2])
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Sequential(
        nn.LayerNorm(2000),

        nn.Conv1d(1, 32, (4),stride=3),
        nn.ReLU(),
        nn.Conv1d(32, 32, (8),stride=4),
        nn.ReLU(),
        nn.Conv1d(32, 64, (8),stride=4),
        nn.ReLU(),
        nn.Conv1d(64, 64, (8),stride=4),
        nn.ReLU(),
        nn.Conv1d(64, 64, (8),stride=4),
        nn.ReLU(),
        nn.Flatten(start_dim=1),        
        nn.LayerNorm(64),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,1),
        
        )

    print(model)

    torch.manual_seed(42)

    model = model.to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, weight_decay=l2Reg, foreach=True, fused=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",factor=0.5,patience=200,threshold=0.0005, min_lr=0.2)
    trainLossHistory = []
    testLossHistory = []
    
    best_Loss = np.inf
    best_MAPE = 0
    best_weights = None
    last_lr = learning_rate
    history = []

    for epoch in range(epochs):
                # take a batch
        for xTrain, yTrain in trainDataLoader:

            labels = yTrain.to(rank)
            # forward pass
            print(xTrain.unsqueeze(1).shape)
            y_pred = ddp_model((xTrain.unsqueeze(1)).to(rank))
            #print(y_pred.squeeze(1).shape)
            loss = loss_fn(y_pred.squeeze(1), labels)
                # backward pass

                #optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
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

        y_pred = [elem for batch in y_pred for pred in batch for elem in pred]

        train_pred = [elem for batch in train_pred for pred in batch for elem in pred]

        y_pred = np.array(y_pred).flatten()
        y_pred = torch.from_numpy(y_pred).type(torch.float)

        train_pred = np.array(train_pred).flatten()

        train_pred = torch.from_numpy(train_pred).type(torch.float)
        #print(train_pred.shape)
        trainLoss = loss_fn(train_pred, y_train)
        testLoss = loss_fn(y_pred, y_test)
        MAE = nn.L1Loss()
        trainMAE = MAE(train_pred, y_train)
        testMAE = MAE(y_pred, y_test)
        
        if(last_lr!=scheduler.get_last_lr()):
           print(scheduler.get_last_lr())
           last_lr=scheduler.get_last_lr()
        scheduler.step(testLoss)

        valMAPE = []
        trainMAPE = []
        for index in range(len(y_test)):
           if(float(y_test[index])==0.0): 
               continue
               valMAPE.append(y_pred[index])
           else:
               valMAPE.append( abs( float(y_test[index]) - float(y_pred[index]) ) / abs(float(y_test[index]) ) )
        valMAPE = sum(valMAPE)/len(valMAPE)

        for index in range(len(y_train)):
           if(float(y_train[index])==0.0):
              continue
              trainMAPE.append(train_pred[index])
           else:
              trainMAPE.append( abs( float(y_train[index]) - float(train_pred[index]) ) / abs(float(y_train[index]) ) )
        trainMAPE = sum(trainMAPE)/len(trainMAPE)

        trainLossHistory.append((np.sqrt(float(trainLoss)),epoch, trainMAE))
        testLossHistory.append((np.sqrt(float(testLoss)),epoch, testMAE))

        history.append((valMAPE,epoch))
        
        if testLoss < best_Loss:

           best_Loss = testLoss

           best_weights = copy.deepcopy(model.state_dict())


        if(epoch%1==0):

            print("epoch: ",epoch,"trainRMSE: %.6f" % np.sqrt(float(trainLoss)),"trainMAE: %.6f" % (float(trainMAE))," trainMAPE: %6f" % trainMAPE,"testRMSE: %.6f" % np.sqrt(float(testLoss)),"testMAE: %.6f" % (float(testMAE)), " VAL MAPE: %6F" % (valMAPE) )
    endTime = time.perf_counter()
    print(ddp_model)
    
    print(f"Performance time: {endTime - startTime:.6f} seconds")
    print("Best MAPE: ", min(history)[0])
    print("MAPE Epoch: ", min(history)[1])
    print("best test Loss: ", min(testLossHistory)[0])
    print("Best test MAE: ", min(testLossHistory)[2])
    print("Test Loss Epoch: ", min(testLossHistory)[1])
    print("best train Loss: ", min(trainLossHistory)[0])
    print("Best train MAE: ", min(trainLossHistory)[2])
    print("Train Loss Epoch: ", min(trainLossHistory)[1])
    diffList = []
    model.load_state_dict(best_weights)

    y_pred=[]
    for xTest,yTest in testDataLoader:
        pred = ddp_model((xTest.unsqueeze(1)).to(rank))
        y_pred.append(pred.cpu().detach().numpy())
    y_pred = [elem for batch in y_pred for pred in batch for elem in pred]
    y_pred = np.array(y_pred).flatten()


    valMAPE = []
    for index in range(len(y_test)):
        if(float(y_test[index])==0.0):
            
            valMAPE.append(-1)
        else:
            valMAPE.append( abs( float(y_test[index]) - float(y_pred[index]) ) / abs(float(y_test[index]) ) )

    torch.save(model.state_dict(), "exp_cnn_quant.pt")



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
    os.environ["MASTER_PORT"] = str(29501) 
    main()

