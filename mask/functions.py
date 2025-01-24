from pandas import DataFrame
import random
from . import utility
from numpy.linalg import inv
import math
import itertools

def MASK_Distortion(dataset: DataFrame, p: float):
    '''
    MASK
    
    choose a probability p

    2 event
    P(x = true) -> 1-p % -> we add or remove an element in the transaction
    P(x = false) -> p% -> the transaction remains the same
    '''
    distorted_dataset = dataset.copy(deep=True)
    for i in range(0,len(distorted_dataset)):
        for column in distorted_dataset.columns:
            event = random.random()
            if event > p:
                distorted_dataset.loc[i, column] = int(not dataset.loc[i, column])

    return distorted_dataset




def MASK_Support(dataset:DataFrame, rule, p:float, M_inv = None):
    if M_inv is None:
        M_inv = inv(
            utility.computeM(
                size=int(math.pow(2,len(rule))),
                p=p
            )
        )

    linC_D = utility.computeLinC_D(dataset,rule)

    #print(linC_D)
    #print(M_inv)

    C_T_11 = utility.vectormatrixProdMod(linC_D,M_inv)

    #print(C_T_11)

    return C_T_11/len(dataset)


def MASK_Apriori(dataset: DataFrame, p: float, min_sup,levels: int = None):
    if levels is None:
        levels = len(dataset.columns)
    rules = [
        [],
        [ [item] for item in dataset.columns ]
    ]

    for i in range(2, levels+1):
        print(f"Constructing Level: {i}")
        combinations = list(itertools.combinations(dataset.columns,i))
        rule_i = [list(c) for c in combinations]
        rules.append(rule_i)
        

    for i in range(1,levels+1):
        print(f"Analysing Level: {i}")
        size = int(math.pow(2,i))
        M_inv = inv(
            utility.computeM(size,p)
        )
        
        rules[i] = [
            rule
            for rule in rules[i]
            if MASK_Support(dataset,rule,p,M_inv) >= min_sup
        ]

    return rules