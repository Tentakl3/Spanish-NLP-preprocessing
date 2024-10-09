from bm25 import Toolkit_bm25
import pandas as pd
import pickle as pk
import numpy as np

class Normalization:
    def __init__(self,term_document_matrix, vocabulary) -> None:
        self.term_document_matrix = term_document_matrix
        self.vocabulary = vocabulary
        self.k = 1.2
        self.b = 0.8

    def term_document_normalize(self):
        """Normalize the term document matrix"""
        col_sums = self.term_document_matrix.sum(axis=0)
        tdmn = self.term_document_matrix.div(col_sums, axis=1)

        with open('Corpus/tdmbm25Pickle', 'wb') as tdmnfile:
            pk.dump(tdmn, tdmnfile)

    def term_document_bm25(self):
        """Create bm25 normalized matrix"""
        mbm25 = pd.read_pickle('Corpus/bm25Pickle')
        m = {}
        for v1 in self.vocabulary:
            m[v1] = []
            s = mbm25[v1].sum()
            for v2 in self.vocabulary:
                m[v1].append(mbm25[v1][v2] / s)
                
        tdmbm25 = pd.DataFrame.from_dict(m, dtype=float)
        tdmbm25.index = self.vocabulary

        tdmbm25.to_csv('Corpus/tdmbm25Pickle.csv', index = True)

        with open('Corpus/tdmbm25Pickle', 'wb') as tdmbm25file:
            pk.dump(tdmbm25, tdmbm25file)

    def update_matrix_bm25(self):
        """Update bm25 value matrix (not the normalized matrix)"""
        tlbm25 = Toolkit_bm25(self.term_document_matrix, self.vocabulary)
        tlbm25.matrix_bm25()