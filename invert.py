'''Represent a word by a row in the inverted indexing of the concept-to-term matrix'''

from collections import Counter
import scipy
import numpy as np
from nltk import probability
from math import log
from scipy.spatial.distance import cosine
from math import pow

class Matrix:
    def __init__(self, data, concepts):
        self.data = data
        self.all_words = []
        for v in data.values():
            self.all_words += v
        self.voc = set(self.all_words)
        self.concepts = concepts
        self.matrix = np.zeros((len(self.voc), len(self.concepts)), dtype=float)

    def fill_matrix(self):
        words_counts = Counter(self.all_words)
        words_probs = {w: words_counts[w]/len(self.all_words) for w in self.voc}
        concept_prob = 1/30
        #print(sum(words_probs.values()))
        for i in range(len(self.concepts)):
            c = self.concepts[i]
            all_j = 0
            ppmi = []
            counts = Counter(self.data[c])
            for w in self.voc:
                if w in counts.keys():
                    joint_p = counts[w]/len(self.data[c])
                    all_j += joint_p
                    pmi = log(joint_p/(words_probs[w]*concept_prob),2)
                    ppmi.append(max(pmi,0.0))
                else:
                    ppmi.append(0.0)
            ppmi = np.array(ppmi)
            self.matrix[:,i] = ppmi

    def get_concepts(self,w):
        all = dict()
        w_i = list(self.voc).index(w)
        for i in range(len(self.concepts)):
            if self.matrix[w_i,i] != 0.0:
                all[i]=self.matrix[w_i,i]
        return all
        #return sorted(all.items(),key=lambda x:x[1], reverse=True)