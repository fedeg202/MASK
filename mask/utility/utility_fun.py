from pandas import DataFrame
import math
import numpy as np
from numpy.linalg import inv
from mask.utility.classes import AprioriRule
from mlxtend.frequent_patterns import apriori


def support(T: DataFrame ,X_U_Y: list | dict) -> float:
    """
    Computes the support of given attributes in the dataset.

    Parameters
    ----------
    T : pandas DataFrame
        The dataset containing transactions.
    X_U_Y : list or dict
        The names of the attributes considered (X ∪ Y).

    Returns
    -------
    float
        The support of the attributes in the dataset.
    """
    for attribute in X_U_Y:
        if attribute not in T.columns:
            raise ValueError("Attribute not found in dataset.")
    
    count = sum(all(getattr(tuple, attr) == 1 for attr in X_U_Y) for tuple in T.itertuples(False))
    return count / len(T)

def confidence(T: DataFrame, X: list | dict, Y: list | dict) -> float:
    """
    Computes the confidence of rule X ⇒ Y.

    Parameters
    ----------
    T : pandas DataFrame
        The dataset containing transactions.
    X : list or dict
        The antecedent (X) of the rule.
    Y : list or dict
        The consequent (Y) of the rule.

    Returns
    -------
    float
        The confidence value of the rule X ⇒ Y.
    """
    if len(X) + len(Y) > len(T.columns):
        raise ValueError("Invalid rule: More attributes than dataset columns.")
    if any(item in Y for item in X):
        raise ValueError("Invalid rule: X and Y must be disjoint.")
    
    countX = sum(all(getattr(tuple, attr) == 1 for attr in X) for tuple in T.itertuples(False))
    countXY = sum(all(getattr(tuple, attr) == 1 for attr in X + Y) for tuple in T.itertuples(False))
    
    return countXY / countX if countX else 0

def Apriori(dataset: DataFrame, min_sup: float, levels: int = None):
    """
    Implements the Apriori algorithm to find frequent itemsets.

    Parameters
    ----------
    dataset : pandas DataFrame
        The dataset containing transactions.
    min_sup : float
        The minimum support threshold.
    levels : int, optional
        Maximum length of itemsets to consider. Defaults to all itemsets.

    Returns
    -------
    list
        A list of frequent itemsets with their support values.
    """
    if levels is None:
        levels = len(dataset.columns)
    frequent_itemsets = apriori(dataset, min_support=min_sup, use_colnames=True)
    frequent_itemsets['size'] = frequent_itemsets['itemsets'].apply(len)
    
    rules = [[]]
    for i in range(1, levels):
        itemsets_size = frequent_itemsets[frequent_itemsets['size'] == i]
        if itemsets_size.empty:
            break
        rules.append([AprioriRule(list(itemset), support) for itemset, support in zip(itemsets_size['itemsets'], itemsets_size['support'])])
    return rules



def hammingDistanceBitwise(a:int,b:int):
    return (a^b).bit_count()


def computeM(size: int, p: float) -> np.ndarray:
    """
    Computes the transformation matrix M for MASK algorithm.

    Parameters
    ----------
    size : int
        The size of the matrix (based on itemsets count).
    p : float
        Probability factor used in MASK.

    Returns
    -------
    np.ndarray
        The transformation matrix.
    """
    max_exp = int(math.log2(size))
    M = np.diag([math.pow(p, max_exp) for _ in range(size)])
    for i in range(size):
        for j in range(i+1, size):
            difference = bin(i ^ j).count('1')
            M[i][j] = math.pow(1-p, difference) * math.pow(p, max_exp - difference)
    
    for i in range(1, size):
        for j in range(i):
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
