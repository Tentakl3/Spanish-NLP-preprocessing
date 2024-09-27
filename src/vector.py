import pickle as pk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class Vector:
    """Class to represent the vectorization of text"""
    def __init__(self, text, vocabulary):
        self.text = text
        self.vocabulary = vocabulary
        self.contexts = {}

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

        contextfile = open('Corpus/contextPickle', 'ab')
        pk.dump(self.contexts, contextfile)
        contextfile.close()

    def term_document(self):
        contextfile = open('Corpus/contextPickle', 'rb')    
        self.contexts = pk.load(contextfile)
        contextfile.close()

        contexts_list = self.contexts.values()

        #docs = ['why hello there', 'omg hello pony', 'she went there? omg']
        contexts_strings = [' '.join(x) for x in contexts_list]
        #print(contexts_strings)

        vec = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
        X = vec.fit_transform(contexts_strings)
        df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())