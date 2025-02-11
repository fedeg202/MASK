from mask.utility import utility_fun
import numpy as np

def support_error(AprioriRuleslevel, MASKRuleslevel):
    """
    Computes the percentage error between Apriori and MASK support values.
    
    Parameters
    ----------
    AprioriRuleslevel : list
        List of Apriori rule objects.
    MASKRuleslevel : list
        List of MASK rule objects.
    
    Returns
    -------
    float
        The average percentage error in support values.
    """
    total_error = 0
    count = 0
    
    for apriorirule in AprioriRuleslevel:
        if apriorirule and apriorirule in MASKRuleslevel:
            count += 1
            index = MASKRuleslevel.index(apriorirule)
            rec_sup = MASKRuleslevel[index].support
            act_sup = apriorirule.support
            total_error += abs(rec_sup - act_sup) / act_sup
    
    return (total_error / count) * 100 if count else 0

def identity_error(AprioriRulesLevel, MASKRulesLevel):
    """
    Computes false positive and false negative rates for MASK rules.
    
    Parameters
    ----------
    AprioriRulesLevel : list
        List of Apriori rule objects.
    MASKRulesLevel : list
        List of MASK rule objects.
    
    Returns
    -------
    tuple
        (false_positive_rate, false_negative_rate)
    """
    false_positive_cnt = sum(1 for rule in MASKRulesLevel if rule not in AprioriRulesLevel)
    false_negative_cnt = sum(1 for rule in AprioriRulesLevel if rule not in MASKRulesLevel)
    F = len(AprioriRulesLevel)
    
    if F:
        return false_positive_cnt / F, false_negative_cnt / F
    return (100, 0) if false_positive_cnt > 0 else (0, 0)


def P(s_0, p, a):
    """
    Computes the privacy level based on the reconstruction probability.
    
    Parameters
    ----------
    s_0 : float
        A parameter related to the initial state.
    p : float
        The probability factor used in distortion.
    a : float
        An additional parameter affecting reconstruction.
    
    Returns
    -------
    float
        The privacy percentage obtained.
    """
    return (1 - utility_fun.R(s_0, p, a)) * 100
