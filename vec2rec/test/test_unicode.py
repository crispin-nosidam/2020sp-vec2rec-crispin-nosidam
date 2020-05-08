import nltk
import krovetzstemmer
import unicodedata
import posixpath

#print(posixpath.basename("s3://csci-e29/project/asldk/abc.txt"))


"""
a = "  \n \n a \n \n b"
print(nltk.word_tokenize(a))
"""
s = 'Kästner'
def ud():
    return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore')
print(ud())
s = ud().decode("utf-8")

if (s.isalnum() and s.lower() not in nltk.corpus.stopwords.words("english")):
    word = krovetzstemmer.Stemmer().stem(s)
print(word)

