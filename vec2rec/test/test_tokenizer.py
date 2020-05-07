from ..preprocess.tools import Tokenizer
from nltk import word_tokenize
import re
from string import punctuation
from nltk.corpus import stopwords
from krovetzstemmer import Stemmer

class A:
    tok = Tokenizer()

a = A()
text = "She even shows-me her boobs and I like it.\nHello world!"
print(A.tok(text))

print(list(token.lower() for token in word_tokenize(re.sub("\n", "", text.translate(str.maketrans(punctuation, " " * len(punctuation))))) if token.isalnum ))
print(list(token.lower() for token in word_tokenize(re.sub("\n", "", text.translate(str.maketrans(punctuation, " " * len(punctuation))))) if token.isalnum and token.lower() not in stopwords.words("english")))
stemmer = Stemmer()
print(list(stemmer.stem(token.lower()) for token in word_tokenize(re.sub("\n", "", text.translate(str.maketrans(punctuation, " " * len(punctuation))))) if token.isalnum and token.lower() not in stopwords.words("english")))
