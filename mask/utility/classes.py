import numpy as np

class Rule:
    def __init__(self,itemset: list | dict,support: float = 0, confidence: float = 0):
        self.itemset = itemset
        self.support = support
        self.confidence = confidence

    def __str__(self):
        return f"Rule('{self.itemset}', '{self.support}')"
    
    def __repr__(self):
        return f"Rule('{self.itemset}', '{self.support}')"
    
    def __eq__(self, other):
        if isinstance(other, Rule):
            return set(self.itemset) == set(other.itemset)
        return False
    
    def __iter__(self):
        return iter([self.itemset, self.support, self.confidence])
    

class AprioriRule(Rule):
    def __init__(self,itemset: list | dict,support: float = 0):
        super().__init__(itemset,support)

class MASKRule(Rule):
    def __init__(self,itemset: list | dict,support: float = 0):
        super().__init__(itemset,support)
        self.counters = np.zeros(len(itemset)+1)