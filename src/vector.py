from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from prepro import Prepro
import pickle as pk
import pandas as pd
import numpy as np


class Vector:
    """Class to represent the vectorization of text"""
    def __init__(self, text, vocabulary):
        self.text = text
        self.vocabulary = vocabulary
        self.contexts = {}
        self.col = []
        self.df = pd.DataFrame()

    def check_context(self):
        contextfile = open('contextPickle', 'rb')    
        contextdb = pk.load(contextfile)
        for keys in contextdb:
            print(keys, '=>', contextdb[keys])
        contextfile.close()

    def context(self):
        """collect contexts for each word of the vocabulary"""
        window=8
        for word in self.vocabulary:
            context = []
            for i in range(len(self.text)):
                if self.text[i] == word:
                    for j in range(i-int(window/2), i): #left context
                        if j >= 0: 
                            context.append(self.text[j])
                    try:
                        for j in range(i+1, i+(int(window/2)+1)): #right context
                            context.append(self.text[j])
                    except IndexError:
                        pass
            self.contexts[word] = context

        with open('Corpus/contextPickle', 'wb') as contextfile:
            pk.dump(self.contexts, contextfile)

    def term_document(self):
        with open('Corpus/contextPickle', 'rb') as contextfile:    
            self.contexts = pk.load(contextfile)

        contexts_list = self.contexts.values()

        contexts_strings = [' '.join(x) for x in contexts_list]

        vec = CountVectorizer()
        X = vec.fit_transform(contexts_strings)
        self.col = vec.get_feature_names_out()
        self.df = pd.DataFrame(X.todense(), columns = self.col)
        print(self.df)

    def dot_product(self):
        results = {}
        for v1 in self.col[:10]:
            results[v1] = {}
            for v2 in self.col[:10]:
                results[v1][v2] = np.dot(self.df[v1], self.df[v2])
                
        return pd.DataFrame(results)

    def cos_similarity(self):
        return pd.DataFrame(cosine_similarity(self.df, self.df))

if __name__ == "__main__":
    prepro = Prepro("Corpus/e990519_mod.htm", "Corpus/stopwords.txt", "vocabulary")
    res = prepro.main()
    vector = Vector(res[0], res[1])
    #vector.context()
    vector.term_document()
    print(vector.dot_product())
    print(vector.cos_similarity())