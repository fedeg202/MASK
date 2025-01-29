from pandas import DataFrame
import math
import numpy as np
from numpy.linalg import inv
from mask.utility.classes import AprioriRule
from mlxtend.frequent_patterns import apriori


def support(T: DataFrame ,X_U_Y: list | dict):
    '''
    Parameters:
    T (dataframe)
    X_U_Y (list | dict) : name of the attributes considered (X U Y)
    Return:
    float: support of the attributes in the dataset
    '''
    for attribute in X_U_Y:
        if attribute not in T.columns:
             return ValueError
    count = 0
    for tuple in T.itertuples(False):
        contained = True
        for attribute in X_U_Y:
            try: 
                if getattr(tuple,attribute) == 0:
                    contained = False
                    break
            except AttributeError:
                contained = False,
                break

        if contained:
            count += 1 
    return count/len(T)




def confidence(T: DataFrame, X: list | dict, Y: list | dict):
    '''
    Parameters:
    T (dataframe)
    X ( list | dict): X part of X ==> Y rule
    X (list | dict): Y part of X ==> Y rule
    '''
    if len(X)+len(Y) > len(T.columns):
        return ValueError
    for item in X:
        if Y.__contains__(item):
            return ValueError # XY = empty set
    
    countX=0
    countY=0
    for tuple in T.itertuples():
        containedX = True
        for attribute in X:
            if getattr(tuple,attribute) == 0:
                containedX = False
                break
        if containedX:
            countX += 1
            containedY=True
            for attribute in Y:
                if attribute == '':
                    containedY = False
                    break
                if getattr(tuple,attribute) == 0:
                    containedY = False
                    break
            if containedY:
                countY += 1
    return countY/countX    





def Apriori(dataset: DataFrame, min_sup, levels: int = None):
    if levels is None:
        levels = len(dataset.columns)
    frequent_itemsets = apriori(dataset, min_support=min_sup, use_colnames=True)
    frequent_itemsets['size'] = frequent_itemsets['itemsets'].apply(len)
    rules = [[]]
    for i in range(1,levels):
        itemsets_size = frequent_itemsets[frequent_itemsets['size'] == i]
        if len(itemsets_size) == 0: break
        supports_size = itemsets_size['support'].values
        itemsets = itemsets_size['itemsets'].values
        rules.append([])
        for itemset, support in zip(itemsets, supports_size):
            new_rule = AprioriRule(list(itemset),support)
            rules[i].append(new_rule)
    return rules



def hammingDistanceBitwise(a:int,b:int):
    return (a^b).bit_count()



def computeM(size: int, p: float):
    max_exp = int(math.log2(size))
    M = np.diag([math.pow(p,max_exp) for i in range(size)])
    for i in range(0,size):
        for j in range(i+1,size):
            difference = hammingDistanceBitwise(i,j)
            M[i][j] = math.pow(1-p,difference)*math.pow(p,max_exp-difference)

    for i in range(1,size):
        for j in range(i,-1,-1):
            M[i][j] = M[j][i]

    return M


def computeLinC_D(dataset: DataFrame,rule: list | dict):
    linC_D = np.zeros(len(rule)+1,dtype=int)
    reducted_ds = dataset[rule]
    for tuple in reducted_ds.itertuples(False):
        linC_D[np.sum(tuple,axis=0)] += 1
    
    return linC_D


def vectormatrixProdMod(linC_D,matrix):
    size = matrix.shape[0]
    row = matrix[size-1]
    sum=0
    for j in range(len(linC_D)):
        index = int(math.pow(2,j))-1
        sum += row[index]*linC_D[j]
    return sum


def MASK_Support(dataset: DataFrame,rule,M_inv = None,p = None):
    if M_inv is None and p is None: return ValueError
    if M_inv is None:
        M_inv = inv(
            computeM(
                size=int(math.pow(2,len(rule)-1)),
                p=p
            )
        )

    linC_D = computeLinC_D(dataset,rule)

    C_T_11 = vectormatrixProdMod(linC_D,M_inv)

    return C_T_11/len(dataset)


def R_1(s_0,p):
    return ((s_0 * math.pow(p,2))/((s_0*p)+(1-s_0)*(1-p)))+((s_0 * math.pow(1-p,2))/((s_0*(1-p))+(1-s_0)*p))

def R_0(s_0,p):
    return (((1-s_0) * math.pow(p,2))/(((1-s_0)*p)+s_0*(1-p)))+(((1-s_0) * math.pow(1-p,2))/((s_0*p)+(1-s_0)*(1-p)))

def R(s_0,p,a):
    return a*R_1(s_0,p)+(1-a)*R_0(s_0,p)

def mean_support(inventory, dataset):
    support_vector = []
    for item in inventory:
        support_vector.append(support(dataset,[item]))
    return np.mean(support_vector)
