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
        self.voc = list(set(self.all_words))
        self.concepts = concepts
        #co-occurrence matrix with counts for computing ppmi
        self.occur_matrix = np.zeros((len(self.voc), len(self.concepts)), dtype=int)
        #ppmi matrix
        self.matrix = np.zeros((len(self.voc), len(self.concepts)), dtype=float)

    def fill_counts(self):
        for i in range(len(self.concepts)):
            c = self.concepts[i]
            counts = Counter(self.data[c])
            for w in range(len(self.voc)):
                if self.voc[w] in counts.keys():
                    self.occur_matrix[w,i]=counts[self.voc[w]]

    def fill_matrix(self):
        for i in range(len(self.concepts)):
            nonzero = np.nonzero(self.occur_matrix[:,i])
            p_c = np.sum(self.occur_matrix[:,i])/len(self.all_words)
            for n in np.nditer(nonzero):
                p_joint = self.occur_matrix[n,i]/len(self.all_words)
                p_w = sum(self.occur_matrix[n,:])/len(self.all_words)
                pmi = log(p_joint/(p_c*p_w),2)
                #print(pmi)
                self.matrix[n,i] = max(pmi, 0.0)


    #Laplace smoothing (use instead of fill_counts)
    def smooth(self, k):
        for i in range(len(self.concepts)):
            c = self.concepts[i]
            counts = Counter(self.data[c])
            for w in range(len(self.voc)):
                if self.voc[w] in counts.keys():
                    self.occur_matrix[w,i]=counts[self.voc[w]]+k
                else:
                    self.occur_matrix[w,i]=k
    # def fill_matrix(self):
    #     words_counts = Counter(self.all_words)
    #     words_probs = {w: words_counts[w]/len(self.all_words) for w in self.voc}
    #     concept_prob = 1/30
    #     #print(sum(words_probs.values()))
    #     for i in range(len(self.concepts)):
    #         c = self.concepts[i]
    #         all_j = 0
    #         ppmi = []
    #         counts = Counter(self.data[c])
    #         for w in self.voc:
    #             if w in counts.keys():
    #                 joint_p = counts[w]/len(self.data[c])
    #                 all_j += joint_p
    #                 pmi = log(joint_p/(words_probs[w]*concept_prob),2)
    #                 ppmi.append(max(pmi,0.0))
    #             else:
    #                 ppmi.append(0.0)
    #         ppmi = np.array(ppmi)
    #         self.matrix[:,i] = ppmi

    def get_concepts(self,w):
        all = dict()
        w_i = list(self.voc).index(w)
        for i in range(len(self.concepts)):
            if self.matrix[w_i,i] != 0.0:
                all[i]=self.matrix[w_i,i]
        #return all
        #print(sorted(all.items(),key=lambda x:x[1], reverse=True))
        return sorted(all.items(),key=lambda x:x[1], reverse=True)

    def get_words(self,c):
        all = dict()
        #print(np.nonzero(self.matrix[:, c]))
        for index in np.nditer(np.nonzero(self.matrix[:,c])):
            all[(list(self.voc)[index])]=self.matrix[index,c]
        return sorted(all.items(),key=lambda x:x[1], reverse=True)[:35]
