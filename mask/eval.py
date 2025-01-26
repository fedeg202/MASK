from MASK.mask.utility import utility_fun
import numpy as np

def support_error(AprioriRuleslevel,MASKRuleslevel):
    
    sum = 0
    cnt = 0

    for apriorirule in AprioriRuleslevel:
        if apriorirule is not None and apriorirule in MASKRuleslevel:
            cnt += 1
            index = MASKRuleslevel.index(apriorirule)
            
            rec_sup = MASKRuleslevel[index].support
            act_sup = apriorirule.support
            
            sum += abs(rec_sup-act_sup)/act_sup
    if cnt != 0:
        return (sum/cnt)*100
    else:
        return 0

def identity_error(AprioriRulesLevel,MASKRulesLevel):

    false_positive_cnt = 0
    false_negative_cnt = 0

    for rule in AprioriRulesLevel:
        if rule not in MASKRulesLevel:
            false_positive_cnt += 1
    
    for rule in MASKRulesLevel:
        if rule not in AprioriRulesLevel:
            false_negative_cnt += 1
    
    F = len(AprioriRulesLevel)

    if F != 0:
        false_positive = false_positive_cnt/F
        false_negative = false_negative_cnt/F
        return false_positive, false_negative
    else: return 0,0

def P(s_0,p,a):
    return (1-utility_fun.R(s_0,p,a))*100

def mean_support(inventory, dataset):
    support_vector = []
    for item in inventory:
        support_vector.append(utility_fun.support(dataset,item))
    return np.mean(support_vector)