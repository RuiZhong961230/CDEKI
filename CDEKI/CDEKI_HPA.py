import os
from copy import deepcopy
import numpy as np
from hpa.problem import HPA101, HPA102, HPA103, HPA131, HPA142, HPA143


PopSize = 20
DimSize = 10
LB = [0] * DimSize
UB = [1] * DimSize

TrialRuns = 20
MaxFEs = 1000
curFEs = 0

MaxIter = int(MaxFEs / PopSize)
curIter = 0

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

FuncNum = "HPA131"

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
                Off[j] = np.random.uniform(LB[j], UB[j])
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


def mainCons(ndiv, level):
    global FuncNum, DimSize, Pop, MaxFEs, curIter, MaxIter, LB, UB, BestFit
    Probs = [HPA131(n_div=ndiv, level=level), HPA142(n_div=ndiv, level=level), HPA143(n_div=ndiv, level=level)]
    Name = ["HPA131", "HPA142", "HPA143"]
    for i in range(len(Probs)):
        DimSize = Probs[i].nx
        Pop = np.zeros((PopSize, DimSize))
        MaxFEs = 1000
        MaxIter = int(MaxFEs / PopSize)
        LB = [0] * DimSize
        UB = [1] * DimSize
        FuncNum = Name[i]
        All_Trial_Best = []
        for time in range(TrialRuns):
            Best_list = []
            curIter = 0
            np.random.seed(2024 + 88 * time)
            Initialization(Probs[i])
            Best_list.append(BestFit)
            while curIter <= MaxIter:
                CDEKI(Probs[i])
                curIter += 1
                Best_list.append(BestFit)
            All_Trial_Best.append(Best_list)
        np.savetxt("./CDEKI_Data/HPA_Cons/" + FuncNum + "_" + str(ndiv) + "_" + str(level) + ".csv", All_Trial_Best,
                   delimiter=",")


def mainUncons(ndiv, level):
    global FuncNum, DimSize, Pop, MaxFEs, curIter, MaxIter, LB, UB, BestFit
    Probs = [HPA101(n_div=ndiv, level=level), HPA102(n_div=ndiv, level=level), HPA103(n_div=ndiv, level=level)]
    Name = ["HPA101", "HPA102", "HPA103"]
    for i in range(len(Probs)):
        DimSize = Probs[i].nx
        Pop = np.zeros((PopSize, DimSize))
        MaxFEs = 1000
        MaxIter = int(MaxFEs / PopSize)
        LB = [0] * DimSize
        UB = [1] * DimSize
        FuncNum = Name[i]
        All_Trial_Best = []
        for time in range(TrialRuns):
            Best_list = []
            curIter = 0
            np.random.seed(2024 + 88 * time)
            Initialization(Probs[i])
            Best_list.append(BestFit)
            while curIter <= MaxIter:
                CDEKI(Probs[i])
                curIter += 1
                # print("curIter: ", curIter, "min: ", BestFit)
                Best_list.append(BestFit)
            All_Trial_Best.append(Best_list)
        np.savetxt("./CDEKI_Data/HPA_Uncons/" + FuncNum + "_" + str(ndiv) + "_" + str(level) + ".csv", All_Trial_Best,
                   delimiter=",")


if __name__ == "__main__":
    if os.path.exists('./CDEKI_Data/HPA_Uncons') == False:
        os.makedirs('./CDEKI_Data/HPA_Uncons')
    if os.path.exists('./CDEKI_Data/HPA_Cons') == False:
        os.makedirs('./CDEKI_Data/HPA_Cons')
    for ndiv in range(3, 6):
        for level in range(0, 3):
            mainUncons(ndiv, level)
            mainCons(ndiv, level)
