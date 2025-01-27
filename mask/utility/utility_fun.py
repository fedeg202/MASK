from pandas import DataFrame
import math
import numpy as np
from numpy.linalg import inv
from mask.utility.classes import AprioriRule


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





def Apriori(items, dataset, min_sup, levels: int = None):
    if levels is None:
        levels = len(items)
    '''
    rules[0] = empty set
    rules[1] = rules of length 1
    rules[2] = rules of length 2


    rules[len(items)] = items
    ...
    '''
    rules = [
        [],
        [ AprioriRule([item]) for item in items ]
    ]



    # iterate over all the possible rules length from 1 to len(items)
    for i in range(1, levels+1):
        print(f"Apriori Level: {i}")
        # remove all the rules in rules[i]
        # that don't have a support of at least min_sup

        #print(f"RULES[{i}] BEFORE", rules[i])
        for j in range(len(rules[i])-1,-1,-1):
            sup = support(dataset,rules[i][j].itemset)
            if sup >= min_sup:
                rules[i][j].support = sup
            else:
                rules[i].remove(rules[i][j])
    
        '''
        print(f"RULES[{i}] SUPPORT", [
            support(dataset, rule)
            for rule in rules[i]
        ]
        )'''


        #print(f"RULES[{i}] AFTER", rules[i])

        if len(rules[i]) == 0:
            break


        # generate all the possibile 
        # rules with i+1 elements
        rules.append([]) # create the element rules[i+1]

        #print("RULE[i]", rules[i])
        #print("RULE[i+1]", rules[i+1])
        for rule in rules[i]:
            for j in range(0,len(rules[1])):

                # skip if item is already inside the rule
                if rules[1][j].itemset[0] in rule.itemset:
                    continue
                
                itemset = rule.itemset + rules[1][j].itemset
                itemset.sort()

                new_rule = AprioriRule(itemset)
                if new_rule not in rules[i+1]:

                    rules[i+1].append(new_rule)
                # create a new rule composed of rule + [item]


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
    reducted_ds = dataset[rule] # posso prendere solo le colonne interessanti
    for tuple in reducted_ds.itertuples(False):
        '''sum = 0
        for attribute in rule:
            sum += getattr(tuple,attribute) # o farlo qua con il for
            linC_D[sum] += 1
        '''
        
        linC_D[np.sum(tuple,axis=0)] += 1
    
    return linC_D


def vectormatrixProdMod(linC_D,matrix):
    size = matrix.shape[0]
    row = matrix[size-1]
    sum=0
    for j in range(len(linC_D)):
        index = int(math.pow(2,j))-1
        #print(str(row[index])+" * "+str(linC_D[j]))
        sum += row[index]*linC_D[j]
    return sum


def MASK_Support(linC_D,db_cardinality,M_inv = None,p = None):
    if M_inv is None and p is None: return ValueError
    if M_inv is None:
        M_inv = inv(
            computeM(
                size=int(math.pow(2,len(linC_D)-1)),
                p=p
            )
        )

    C_T_11 = vectormatrixProdMod(linC_D,M_inv)

    return C_T_11/db_cardinality


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
