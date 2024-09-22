import os
from copy import deepcopy
import numpy as np
from cec17_functions import cec17_test_func


PopSize = 100
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30
MaxFEs = DimSize * 1000
curFEs = 0

MaxIter = int(MaxFEs / PopSize)
curIter = 0

Pop = np.zeros((PopSize, DimSize))
Off = np.zeros((PopSize, DimSize))

FitPop = np.zeros(PopSize)
FitOff = np.zeros(PopSize)

FuncNum = 0

BestPop = None
BestFit = float("inf")


def fitness(X):
    global DimSize, FuncNum
    f = [0]
    cec17_test_func(X, f, DimSize, 1, FuncNum)
    return f[0]


# initialize the Pop randomly
def Initialization():
    global Pop, FitPop, curFEs, DimSize, BestPop, BestFit
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = fitness(Pop[i])
        curFEs += 1
    BestFit = min(FitPop)
    BestPop = deepcopy(Pop[np.argmin(FitPop)])


def CDEKI():
    global Pop, FitPop, Off, FitOff, curIter, MaxIter, LB, UB, PopSize, DimSize, curFEs, BestPop, BestFit
    for i in range(PopSize):
        IDX = np.random.randint(0, PopSize)
        while IDX == i:
            IDX = np.random.randint(0, PopSize)
        candi = list(range(0, PopSize))
        candi.remove(i)
        candi.remove(IDX)
        r1, r2 = np.random.choice(candi, 2, replace=False)

        F1 = np.random.normal(0.5, 0.3)
        F2 = np.random.normal(0.5, 0.3)
        if FitPop[IDX] < FitPop[i]:  # DE/winner-to-opt/1
            Off = Pop[IDX] + F1 * (BestPop - Pop[IDX]) + F2 * (Pop[r1] - Pop[r2])
            Gap = Pop[IDX] - Pop[i]
        else:
            Off = Pop[i] + F1 * (BestPop - Pop[i]) + F2 * (Pop[r1] - Pop[r2])
            Gap = Pop[i] - Pop[IDX]

        for j in range(DimSize):
            Cr = np.random.normal(0.5, 0.3)
            if np.random.rand() < Cr:
                Off[j] = Off[j] + np.random.rand() * Gap[j]
            else:
                Off[j] = BestPop[j] + np.random.rand() * Gap[j]

        for j in range(DimSize):
            if Off[j] < LB[j] or Off[j] > UB[j]:
                if np.random.rand() < 0.5:
                    Off[j] = BestPop[j]
                else:
                    Off[j] = np.random.uniform(LB[j], UB[j])
        FitOff = fitness(Off)
        if FitOff < FitPop[i]:
            Pop[i] = Off.copy()
            FitPop[i] = FitOff
            if FitOff < BestFit:
                BestFit = FitOff
                BestPop = Off.copy()


def RunCDEKI():
    global curFEs, curIter, MaxIter, MaxFEs, TrialRuns, Pop, FitPop, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curFEs = 0
        curIter = 0
        np.random.seed(2024 + 88 * i)
        Initialization()
        Best_list.append(min(FitPop))
        while curIter < MaxIter:
            CDEKI()
            curIter += 1
            Best_list.append(min(FitPop))
        All_Trial_Best.append(Best_list)
    np.savetxt("./CDEKI_Data/CEC2017/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, Pop, MaxFEs, MaxIter, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize)
    LB = [-100] * dim
    UB = [100] * dim
    FuncNum = 1
    for i in range(1, 31):
        FuncNum = i
        if i == 2:
            continue
        RunCDEKI()


if __name__ == "__main__":
    if os.path.exists('./CDEKI_Data/CEC2017') == False:
        os.makedirs('./CDEKI_Data/CEC2017')
    Dims = [30, 50]
    for Dim in Dims:
        main(Dim)
