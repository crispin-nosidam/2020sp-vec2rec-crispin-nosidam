import nltk
import os
import pandas as pd
import re
from glob import glob
from krovetzstemmer import Stemmer
from nltk.corpus import stopwords
from string import punctuation
from PyPDF4 import PdfFileReader


def extract_pdf_text(path, format="string"):
    with open(path, "rb") as fileobj:
        pfr = PdfFileReader(fileobj, strict=False)
        text = "" if format == "string" else []
        for pg in range(pfr.getNumPages()):
            if format == "string":
                text += pfr.getPage(pg).extractText()
            else:
                text.append(pfr.getPage(pg).extractText())
        return text


def tokenize(text):
    stemmer = Stemmer()
    return [
        stemmer.stem(token.lower())
        for token in nltk.word_tokenize(re.sub("\n", "", text.translate(str.maketrans(punctuation, " "*len(punctuation)))))
        if (token.isalnum() and token.lower() not in stopwords.words("english") and len(token) > 1)
    ]


def pdf_to_dataframe(dir, file_glob="*.pdf", type="resume"):
    df = pd.DataFrame(columns=["filename", "tokens", "length"]).astype({"length": "int"})
    for file in glob(os.path.join(dir, file_glob)):
        tokens = tokenize(extract_pdf_text(os.path.join(dir, file)))
        df = df.append(
            {
                "type": type,
                "filename": os.path.basename(file),
                "tokens": tokens,
                "length": len(tokens),
            },
            ignore_index=True,
        )
    return df
