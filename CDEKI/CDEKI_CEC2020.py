import os
from opfunu.cec_based.cec2020 import *
from copy import deepcopy
import numpy as np
import warnings

warnings.filterwarnings("ignore")

PopSize = 100
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize

TrialRuns = 30
MaxFEs = 1000
curFEs = 0

MaxIter = int(MaxFEs / PopSize)
curIter = 0

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

FuncNum = 0

BestFit = float("inf")
BestPop = None


# initialize the Pop randomly
def Initialization(func):
    global Pop, FitPop, curFEs, DimSize, BestPop, BestFit
    Pop = np.zeros((PopSize, DimSize))
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])
    best_idx = np.argmin(FitPop)
    BestPop = deepcopy(Pop[best_idx])
    BestFit = FitPop[best_idx]


def CDEKI(func):
    global Pop, FitPop, curIter, MaxIter, LB, UB, PopSize, DimSize, BestPop, BestFit, curIter, MaxIter

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
        FitOff = func(Off)
        if FitOff < FitPop[i]:
            Pop[i] = Off.copy()
            FitPop[i] = FitOff
            if FitOff < BestFit:
                BestFit = FitOff
                BestPop = Off.copy()


def main(Dim):
    global FuncNum, DimSize, Pop, MaxFEs, curIter, MaxIter, LB, UB, BestFit
    CEC2020 = [F12020(Dim), F22020(Dim), F32020(Dim), F42020(Dim), F52020(Dim),
               F62020(Dim), F72020(Dim), F82020(Dim), F92020(Dim), F102020(Dim)]
    for i in range(len(CEC2020)):
        DimSize = Dim
        Pop = np.zeros((PopSize, Dim))
        MaxFEs = 1000 * Dim
        MaxIter = int(MaxFEs / PopSize)
        LB = [-100] * DimSize
        UB = [100] * DimSize
        FuncNum = i + 1
        All_Trial_Best = []
        for time in range(TrialRuns):
            Best_list = []
            curIter = 0
            np.random.seed(2024 + 88 * time)
            Initialization(CEC2020[i].evaluate)
            Best_list.append(BestFit)
            while curIter <= MaxIter:
                CDEKI(CEC2020[i].evaluate)
                curIter += 1
                Best_list.append(BestFit)
            All_Trial_Best.append(Best_list)
        np.savetxt("./CDEKI_Data/CEC2020/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


if __name__ == "__main__":
    if os.path.exists('./CDEKI_Data/CEC2020') == False:
        os.makedirs('./CDEKI_Data/CEC2020')
    Dims = [30, 50]
    for dim in Dims:
        main(dim)
