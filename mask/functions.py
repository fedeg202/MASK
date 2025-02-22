from pandas import DataFrame
import random
from .utility import utility_fun, classes
from numpy.linalg import inv
import math

def MASK_Distortion(dataset: DataFrame, p: float) -> DataFrame:
    """
    Applies MASK distortion to a dataset by randomly flipping item presence based on probability `p`.

    Parameters
    ----------
    dataset : pandas DataFrame
        A one-hot encoded DataFrame representing transactions.
    p : float
        The probability that an item in a transaction remains unchanged.
        If `event > p`, the item's presence is flipped (1 → 0 or 0 → 1).

    Returns
    -------
    distorted_dataset : pandas DataFrame
        The distorted version of the original dataset.
    """
    distorted_dataset = dataset.copy(deep=True)
    for i in range(len(distorted_dataset)):
        for column in distorted_dataset.columns:
            event = random.random()
            if event > p:
                distorted_dataset.loc[i, column] = int(not dataset.loc[i, column])

    return distorted_dataset


from pandas import DataFrame
from mask.utility import utility_fun,classes
from numpy.linalg import inv
import math

def MASK_frequent_itemsets(dataset: DataFrame, p: float, min_sup: float, levels: int = None):
    """
    Finds frequent itemsets in a distorted transaction dataset using the MASK algorithm.

    Parameters
    ----------
    dataset : pandas DataFrame
        A one-hot encoded DataFrame representing transactions.
    p : float
        The probability that an item in a transaction remains unchanged during distortion.
    min_sup : float
        The minimum support threshold for frequent itemsets.
    levels : int, optional (default=None)
        The maximum length of itemsets to be considered. If None, it is set to the number of columns.

    Returns
    -------
    rules : list
        A list of MASKRule objects representing the frequent itemsets.
    """
    def _support(linC_D, db_cardinality, M_inv):
        """
        Computes the estimated true support by correcting distorted counts.
        
        Parameters
        ----------
        linC_D : array-like
            Distorted frequency counts for an itemset.
        db_cardinality : int
            The number of transactions in the dataset.
        M_inv : numpy.ndarray
            The inverse of the distortion matrix.
        
        Returns
        -------
        float
            The corrected support value.
        """
        C_T_11 = utility_fun.vectormatrixProdMod(linC_D, M_inv)
        return C_T_11 / db_cardinality

    if levels is None:
        levels = len(dataset.columns)
    
    column_map = {i: item for i, item in enumerate(dataset.columns)}


    rules = [
        [],
        [classes.MASKRule([item]) for item in column_map.keys()]
    ]
    
    dataset_list = []

    for transaction in dataset.itertuples():
        item_list = [item for item in column_map.keys() if getattr(transaction, column_map[item]) == 1]
        dataset_list.append(item_list)


    infrequent_itemsets = []
    
    for i in range(1, levels + 1):
        print(f"Mask Rule Mining level: {i}")
        
        for transaction in dataset_list:
            transaction = [item for item in transaction if item not in infrequent_itemsets]
            for rule in rules[i]:
                bit_counter = sum(1 for item in rule.itemset if item in transaction)
                rule.counters[bit_counter] += 1
        
        size = int(math.pow(2, i))
        M_inv = inv(utility_fun.computeM(size, p))
        # Computing support for each itemset
        for j in range(len(rules[i]) - 1, -1, -1):
            sup = _support(rules[i][j].counters, len(dataset), M_inv)
            if sup >= min_sup:
                rules[i][j].support = sup
            else:
                if i == 1 and rules[i][j].itemset[0] not in infrequent_itemsets:
                    infrequent_itemsets.append(rules[i][j].itemset[0])
                rules[i].remove(rules[i][j])
        
        if not rules[i]:  # Stop if no more frequent itemsets are found
            break
        
        # Generate candidate itemsets for the next level
        rules.append([])
        for rule in rules[i]:
            for r in rules[1]:
                if r.itemset[0] not in rule.itemset:
                    itemset = sorted(rule.itemset + r.itemset)
                    new_rule = classes.MASKRule(itemset)
                    if new_rule not in rules[i + 1]:
                        rules[i + 1].append(new_rule)
    
    for i in range(1,len(rules)):
        for j in range(len(rules[i])):
            for k in range(i):
                rules[i][j].itemset[k] = column_map[rules[i][j].itemset[k]]

    return rules

