from sklearn.feature_extraction.text import CountVectorizer
from normalization import Normalization
from prepro import Prepro
import pickle as pk
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Vector:
    """Class to represent the vectorization of text"""
    def __init__(self, text, vocabulary, word):
        self.text = text
        self.vocabulary = vocabulary
        self.word = word

    def main(self):
        term_document_matrix = self.term_document()
        norm = Normalization(term_document_matrix, self.vocabulary)
        #norm.term_document_normalize()
        #norm.term_document_bm25()
        #self.cos_similarity()
        self.dot_similarity()
        #self.euclidean_similarity()
        #self.bm25_similarity(term_document_matrix)
        

    def context(self):
        """collect contexts for each word of the vocabulary"""
        contexts = {}
        window = 8
        for word in self.vocabulary:
            context = []
            for i in enumerate(self.text):
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
    
    def term_document(self):
        """Generate de term-document matrix"""
        contexts = pd.read_pickle('Corpus/contextPickle')
        contexts_list = contexts.values()
        contexts_strings = [' '.join(x) for x in contexts_list]
        vec = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
        X = vec.fit_transform(contexts_strings)
        vocab = vec.get_feature_names_out()
        term_document_matrix = pd.DataFrame(X.toarray(), columns=vocab, index=vocab)
        self.vocabulary = vocab

        return term_document_matrix

    def idf(self, word, term_document_matrix):
        N_q = (term_document_matrix.loc[word] > 0).sum()  # count how many documents contain term w
        n = term_document_matrix.shape[1]  # total number of documents
        idf_value = np.log(((n - N_q + 0.5) / (N_q + 0.5)) + 1)
        
        return idf_value
    
    def cos_similarity(self):
        """Calculation of cosine similarity between vectors"""
        tdmn = pd.read_pickle('Corpus/tdmnPickle')
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

    def dot_similarity(self):
        """Calculation of dot product similarity between vectors"""
        tdmn = pd.read_pickle('Corpus/tdmnPickle')
        vec1 = np.array(tdmn[self.word])
        vec1_n = np.linalg.norm(vec1)
        res = tdmn.dot(vec1/vec1_n)
        sorted_voc = res.sort_values(ascending=False)
        
        with open(f'Corpus/{self.word}_dot.txt', 'w', encoding = "utf-8") as file:
            for voc, similarity in sorted_voc.items():
                file.write(f"{voc} : {similarity}\n")
        
    def euclidean_similarity(self):
        """Calculation of the euclidean distance between vectors"""
        tdmn = pd.read_pickle('Corpus/tdmnPickle')
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

    def bm25_similarity(self, term_document_matrix):
        """Calculation of the bm25 similarity betwen vectors"""
        tdmbm25 = pd.read_pickle('Corpus/tdmbm25Pickle')
        
        idf_cache = {}
        for word in self.vocabulary:
            idf_cache[word] = self.idf(word, term_document_matrix)
        
        res = {}
        tdmbm25_np = tdmbm25.to_numpy()
        word_index = self.vocabulary.tolist().index(self.word)
        
        for v1 in self.vocabulary:
            v1_idx = self.vocabulary.tolist().index(v1)
            
            idf_vector = np.array([idf_cache[v2] for v2 in self.vocabulary])
            bm25_word_vector = tdmbm25_np[word_index]
            bm25_v1_vector = tdmbm25_np[v1_idx]
            
            res[v1] = np.sum(idf_vector * bm25_word_vector * bm25_v1_vector)

        res_series = pd.Series(res)
        sorted_voc = res_series.sort_values(ascending=False)

        with open(f'Corpus/{self.word}_bm25.txt', 'w', encoding="utf-8") as file:
            for voc, similarity in sorted_voc.items():
                file.write(f"{voc} : {similarity}\n")

if __name__ == "__main__":
    prepro = Prepro("Corpus/e990519_mod.htm", "Corpus/stopwords.txt", "vocabulary")
    text, vocabuary = prepro.main()
    vector = Vector(text, vocabuary, "organización")
    vector.main()