from pandas import DataFrame
import random
from MASK.mask.utility import utility_fun,classes
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



def MASK_Rule_Mining(dataset: DataFrame, p: float, min_sup: float,levels: int = None):
    if levels is None:
        levels = len(dataset.columns)
    rules = [
        [],
    ]
    frequent_itemsets = [[]]
    infrequent_itemsets = [[]]

    for i in range(1,levels+1):
        print(f"Mask Rule Mining level: {i}")

        combinations = list(itertools.combinations(dataset.columns,i))
        rule_i = [
            classes.MASKRule(sorted(list(c)))for c in combinations]
        rules.append(rule_i)

        frequent_itemsets.append([])
        infrequent_itemsets.append([])


        for tuple in dataset.itertuples():

            item_list = []
            complement_list=[]

            for item in dataset.columns:
                if getattr(tuple,item) == 1 and item not in infrequent_itemsets[i-1]:
                    item_list.append(item)
            for item in frequent_itemsets[i-1]:
                if item not in item_list:
                    complement_list.append(item)

           
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

            sup = utility_fun.MASK_Support(rules[i][j].counters,len(dataset),M_inv)

            if sup >= min_sup:
                rules[i][j].support = sup
                for item in rules[i][j].itemset:
                    if item not in frequent_itemsets[i]:
                        frequent_itemsets[i].append(item)

            else:
                rules[i].remove(rules[i][j])
                for item in rule.itemset:
                    if item not in infrequent_itemsets[i]:
                        infrequent_itemsets[i].append(item)
            
        if len(rules[i]) == 0:
            break
    
    return rules
