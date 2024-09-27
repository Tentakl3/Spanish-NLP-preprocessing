from vector import Vector
from bs4 import BeautifulSoup
import spacy
import nltk

class Prepro:
    """Class represent preprocesing of text"""
    def __init__(self, file_name, file_stopwords, file_out):
        self.file_name = file_name
        self.file_stopwords = file_stopwords
        self.file_out = file_out
        self.tokens = []
        self.vocabulary = []
    
    def main(self):
        """Method to read the html archive."""
        f = open(self.file_name, encoding="utf-8")
        text_string = f.read()
        f.close()
        text_string = text_string.lower()
        self.normalize(text_string)
        return [self.tokens, self.vocabulary]
    
    def normalize(self, text_string):
        """Normalize the lxml text from Corpus."""
        soup = BeautifulSoup(text_string, "html.parser")
        norm_string = soup.get_text()
        self.raw_tokens(norm_string)

    def raw_tokens(self, norm_string):
        """Reciebes prepoces text and return raw tokens."""
        raw_tokens=nltk.Text(nltk.word_tokenize(norm_string))
        self.clean_tokens(raw_tokens)
        

    def clean_tokens(self, raw_tokens):
        """Receives a list of raw tokens and returns tokens of letters only."""
        clean_tokens=[]
        for tok in raw_tokens:
            t=[]
            for char in tok:
                if char.isalpha():
                    t.append(char)
            letter_token=''.join(t)
            if letter_token !='':
                clean_tokens.append(letter_token)

        self.delete_stopwords(clean_tokens)

    def delete_stopwords(self, clean_tokens):
        """Receives a list of tokens and eliminates stopwords using a file of stopwords."""
        f=open(self.file_stopwords, encoding='utf-8')
        words=f.read()
        stopwords=words.split()
        f.close()
        
        tokens_without_stopwords=[]
        for tok in clean_tokens:
            if tok not in stopwords:
                tokens_without_stopwords.append(tok)
        
        self.lemmatize(tokens_without_stopwords)

    def lemmatize(self, tokens):
        """Lemmatize the list of tokens"""
        nlp = spacy.load("es_core_news_sm")
        doc = nlp(" ".join(tokens))
        lemmatized_tokens = [token.lemma_ for token in doc if " " not in token.lemma_]
        self.tokens = lemmatized_tokens
        self.vocabulary = sorted(list(set(self.tokens)))
    """
    def pos_tag(self, lemmatized_tokens):
        nlp = spacy.load("es_core_news_sm")
        doc = nlp(" ".join(lemmatized_tokens))
        pos_tokens = [[token.text, token.pos_] for token in doc]
        print(pos_tokens[:100])
    """

if __name__ == "__main__":
    prepro = Prepro("Corpus/e990519_mod.htm", "Corpus/stopwords.txt", "vocabulary")
    res = prepro.main()
    vector = Vector(res[0], res[1])
    #vector.context()
    vector.term_document()