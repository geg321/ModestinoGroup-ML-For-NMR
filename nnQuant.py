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

def example(rank, world_size):
    datasetSize = 0
        # Read data
        # loss function and optimize
    l2Reg = float(sys.argv[4])
    mom = 0
    learning_rate= float(sys.argv[3])
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

        #print("length of exp samples: ",len(xVal))

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

    #X = torch.from_numpy(X).type(torch.float)
    #y = torch.from_numpy(y).type(torch.float)


    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    X_train = X_train[int(rank*(len(X_train)//world_size)):int((rank+1)*(len(X_train)//world_size))]
    y_train = y_train[rank*(len(y_train)//world_size):(rank+1)*(len(y_train)//world_size)]
    
    X_train = torch.from_numpy(X_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    X_test = torch.from_numpy(X_test).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.float)
    
    print(X_train.shape)
    epochs = int(sys.argv[2])
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model


    model = nn.Sequential(
        nn.LayerNorm(2000),
        #nn.LayerNorm(2000),
        nn.Linear(2000, 5000),
        nn.ReLU(),
        nn.Linear(5000, 3000),
        nn.ReLU(),
        nn.Linear(3000, 3000),
        nn.ReLU(),
        nn.Linear(3000, 3000),
        nn.ReLU(),
        nn.Linear(3000, 1500),
        nn.ReLU(),
        nn.Linear(1500, 1500),
        nn.ReLU(),
        nn.Linear(1500, 750),
        nn.ReLU(),
        nn.Linear(750, 1),

    )

#    for layer in range(int(sys.argv[5])):
#        model.append(nn.Linear(2000,2000))
#        model.append(nn.ReLU())
#    model.append(nn.Linear(2000, 1))

    model = model.to(rank)
    print(model)   
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, weight_decay=l2Reg) #, foreach=True, fused=False)
    batchSize= 1
    history = []
    trainLossHistory = []
    testLossHistory = []
#    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
# forward pass
    for epoch in range(epochs):

        labels = y_train.to(rank)

            # forward pass

        y_pred = ddp_model((X_train).to(rank))

        loss = loss_fn(y_pred.squeeze(1), labels)

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
        
        y_pred = ddp_model(X_test.unsqueeze(0)).cpu().detach().numpy()[0]
        train_pred = ddp_model(X_train.unsqueeze(0)).cpu().detach().numpy()[0]

        trainMAPE = []
        valMAPE = []
        train_pred = torch.from_numpy(train_pred).type(torch.float)
        y_pred = torch.from_numpy(y_pred).type(torch.float)
        
        trainLoss = loss_fn(train_pred, y_train)
        testLoss = loss_fn(y_pred, y_test)
        MAE = nn.L1Loss()
        trainMAE = MAE(train_pred, y_train)
        testMAE = MAE(y_pred, y_test)        


        for index in range(len(y_test)):
           if(float(y_test[index])==0.0):
               continue
               valMAPE.append(0)
           else:
               valMAPE.append( abs( float(y_test[index]) - float(y_pred[index]) ) / abs(float(y_test[index]) ) )
        valMAPE = sum(valMAPE)/len(valMAPE)

        for index in range(len(y_train)):
           if(float(y_train[index])==0.0):
              continue
              trainMAPE.append(0)
           else:
              trainMAPE.append( abs( float(y_train[index]) - float(train_pred[index]) ) / abs(float(y_train[index]) ) )
        trainMAPE = sum(trainMAPE)/len(trainMAPE)
        history.append((valMAPE,epoch))

        trainLossHistory.append((np.sqrt(float(trainLoss)),epoch, trainMAE))
        testLossHistory.append((np.sqrt(float(testLoss)),epoch, testMAE))
        
        if(epoch%1==0):

            print("epoch: ",epoch,"trainRMSE: %.30f" % np.sqrt(float(trainLoss)),"trainMAE: %.6f" % (float(trainMAE))," trainMAPE: %6f" % trainMAPE,"testRMSE: %.30f" % np.sqrt(float(testLoss)), "testMAE: %.6f" % (float(testMAE))," VAL MAPE: %6F" % (valMAPE) )
    endTime = time.perf_counter()
    print(f"Performance time: {endTime - startTime:.6f} seconds")
    print(model)
    print("Best MAPE: ", min(history)[0])
    print("Epoch: ", min(history)[1])
    print("Best train RMSE: ", min(trainLossHistory)[0])
    print("Best train MAE: ", min(trainLossHistory)[2])
    print("Epoch: ", min(trainLossHistory)[1])
    print("Best test RMSE: ", min(testLossHistory)[0])
    print("Best test MAE: ", min(testLossHistory)[2])
    print("Epoch: ", min(testLossHistory)[1])



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
    os.environ["MASTER_PORT"] = "29501"
    main()

