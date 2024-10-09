import pickle as pk
import pandas as pd

class Toolkit_bm25:
    def __init__(self, term_document_matrix, vocabulary, k = 1.2, b = 0.8):     
        self.term_document_matrix = term_document_matrix
        self.vocabulary = vocabulary
        self.k = k
        self.b = b

    def avgdl(self):
        s = 0
        for v in self.vocabulary:
            s = s + self.term_document_matrix[v].sum()

        return s / len(self.vocabulary)

    def c(self, w, d):
        count_w = self.term_document_matrix[d][w]
        return count_w

    def bm25(self, w, d, avgdl):
        """BM25 function"""
        dl_l = len(self.term_document_matrix[d])
        bm_25 = ((1+self.k)*self.c(w,d))/(self.c(w,d) + self.k*(1 - self.b + self.b*(dl_l / avgdl)))
        return bm_25
    
    def matrix_bm25(self):
        m = {}
        avgdl = self.avgdl()
        for v1 in self.vocabulary:
            m[v1] = []
            for v2 in self.vocabulary:
                m[v1].append(self.bm25(v2, v1, avgdl))
        mbm25 = pd.DataFrame.from_dict(m, dtype=float)
        mbm25.index = self.vocabulary

        with open('Corpus/bm25Pickle', 'wb') as bm25file:
            pk.dump(mbm25, bm25file)