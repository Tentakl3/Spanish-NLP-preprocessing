import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from prepro import Prepro
import pickle as pk
import pandas as pd
import numpy as np


class Vector:
    """Class to represent the vectorization of text"""
    def __init__(self, text, vocabulary, word):
        self.text = text
        self.vocabulary = vocabulary
        self.word = word

    def main(self):
        contexts = self.context()
        term_document_matrix = self.term_document(contexts)
        tdmn = self.term_document_normalize(term_document_matrix)
        #self.cos_similarity(tdmn)
        #self.dot_similarity(tdmn)
        self.euclidean_similarity(tdmn)
        

    def context(self):
        """collect contexts for each word of the vocabulary"""
        contexts = {}
        window = 8
        for word in self.vocabulary:
            context = []
            for i in range(len(self.text)):
                if self.text[i] == word:
                    for j in range(i-int(window/2), i): #left context
                        if j >= 0: 
                            context.append(self.text[j])
                    for j in range(i+1, i+(int(window/2)+1)): #right context
                        if j < len(self.text):
                            context.append(self.text[j])

            contexts[word] = context

        with open('Corpus/contextPickle', 'wb') as contextfile:
            pk.dump(contexts, contextfile)

        return contexts
    
    def term_document(self, contexts):
        """Generate de term-document matrix"""
        contexts_list = contexts.values()
        contexts_strings = [' '.join(x) for x in contexts_list]
        vec = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
        X = vec.fit_transform(contexts_strings)
        vocab = vec.get_feature_names_out()
        term_document_matrix = pd.DataFrame(X.toarray(), columns=vocab, index=vocab)
        self.vocabulary = vocab
        return term_document_matrix
    
    def term_document_normalize(self, term_document_matrix):
        """Normalize the term document matrix"""
        col_sums = term_document_matrix.sum(axis=0)
        tdmn = term_document_matrix.div(col_sums, axis=1)
        tdmn.to_csv("Corpus/term_document_matrix.csv", header=True, index=True, encoding="utf-8")
        return tdmn
    
    def cos_similarity(self, tdmn):
        """Calculation of cosine similarity between vectors"""
        vec1 = np.array(tdmn[self.word])
        norm_vec1 = np.linalg.norm(vec1)
        res = {}
        for v in self.vocabulary:
            vec2 = np.array(tdmn[v])
            norm_vec2 = np.linalg.norm(vec2)
            res[v] = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
        
        res_series = pd.Series(res)
        sorted_voc = res_series.sort_values(ascending=False)

        with open(f'Corpus/{self.word}_cos.txt', 'w', encoding="utf-8") as file:
            for voc, similarity in sorted_voc.items():
                file.write(f"{voc} : {similarity}\n")

    def dot_similarity(self, tdmn):
        """Calculation of dot product similarity betwen vectors"""
        vec1 = np.array(tdmn[self.word])
        vec1 = np.linalg.norm(vec1)
        res = tdmn.dot(vec1)
        sorted_voc = res.sort_values(ascending=False)
        
        with open(f'Corpus/{self.word}_dot.txt', 'w', encoding = "utf-8") as file:
            for voc, similarity in sorted_voc.items():
                file.write(f"{voc} : {similarity}\n")

    def euclidean_similarity(self, tdmn):
        """Calculation of the euclidean distance between vectors"""
        vec1 = np.array(tdmn[self.word])
        res = {}
        for v in self.vocabulary:
            vec2 = np.array(tdmn[v])
            res[v] = np.linalg.norm(vec1 - vec2)

        res_series = pd.Series(res)
        sorted_voc = res_series.sort_values(ascending=True)

        with open(f'Corpus/{self.word}_euclidean.txt', 'w', encoding="utf-8") as file:
            for voc, similarity in sorted_voc.items():
                file.write(f"{voc} : {similarity}\n")

if __name__ == "__main__":
    prepro = Prepro("Corpus/e990519_mod.htm", "Corpus/stopwords.txt", "vocabulary")
    text, vocabuary = prepro.main()
    vector = Vector(text, vocabuary, "organización")
    vector.main()