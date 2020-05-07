import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from krovetzstemmer import Stemmer
import pandas as pd
import os
import re
from glob import glob
from PyPDF4 import PdfFileReader

DATA_DIR = os.path.abspath("vec2rec/data/resume")


def extract_pdf_text(path, format="string"):
    with open(path, "rb") as fileobj:
        pfr = PdfFileReader(fileobj)
        text = "" if format == "string" else []
        for pg in range(pfr.getNumPages()):
            if format == "string":
                text += pfr.getPage(pg).extractText()
            else:
                text.append(pfr.getPage(pg).extractText())
        return text


text = extract_pdf_text(os.path.join(DATA_DIR, "1Amy.pdf"))
print(text)


def clean_text(text):
    stemmer = Stemmer()
    return [
        stemmer.stem(token.lower())
        for token in nltk.word_tokenize(re.sub("[ ]+", " ", re.sub("\n", "", text)))
        if (token.isalnum() and token not in stopwords.words("english"))
    ]

tokens = clean_text(text)
print(tokens)
