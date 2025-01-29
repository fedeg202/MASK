from pandas import DataFrame
import random
from .utility import utility_fun,classes
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



def MASK_frequent_itemsets(dataset: DataFrame, p: float, min_sup: float,levels: int = None):


    def _support(linC_D,db_cardinality,M_inv):
        if M_inv is None and p is None: return ValueError
        if M_inv is None:
            M_inv = inv(
                utility_fun.computeM(
                    size=int(math.pow(2,len(linC_D)-1)),
                    p=p
                )
            )

        C_T_11 = utility_fun.vectormatrixProdMod(linC_D,M_inv)

        return C_T_11/db_cardinality


    if levels is None:
        levels = len(dataset.columns)
    rules = [
        [],
        [classes.MASKRule([item]) for item in dataset.columns]
    ]

    infrequent_itemsets = []

    for i in range(1,levels+1):
        print(f"Mask Rule Mining level: {i}")


        for tuple in dataset.itertuples():

            item_list = []

            for item in dataset.columns:
                if getattr(tuple,item) == 1 and item not in infrequent_itemsets:
                    item_list.append(item)
           
            for rule in rules[i]:
                bit_counter = 0
                for item in rule.itemset:
                    if item in item_list:
                        bit_counter += 1
                
                rule.counters[bit_counter]+=1
            
        
        

        for j in range(len(rules[i])-1,-1,-1):

            size = int(math.pow(2,i))
            M_inv = inv(
                utility_fun.computeM(size,p)
            )

            sup = _support(rules[i][j].counters,len(dataset),M_inv)

            if sup >= min_sup:
                rules[i][j].support = sup

            else:
                if i == 1 and rules[i][j].itemset[0] not in infrequent_itemsets:
                    infrequent_itemsets.append(item)         
                rules[i].remove(rules[i][j])
            
        if len(rules[i]) == 0:
            break

        rules.append([])

        for rule in rules[i]:
            for r in rules[1]:
                if r.itemset[0] not in rule.itemset:
                    itemset =  rule.itemset + r.itemset
                    itemset.sort()
                    new_rule = classes.MASKRule(itemset)
                    if new_rule not in rules[i+1]:
                        rules[i+1].append(new_rule)
    
    return rules
