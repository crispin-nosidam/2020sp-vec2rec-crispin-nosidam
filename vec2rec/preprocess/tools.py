import nltk
import os
import pandas as pd
import re
import unicodedata
from glob import glob
from krovetzstemmer import Stemmer
from nltk.corpus import stopwords
from string import punctuation
from PyPDF4 import PdfFileReader


class TokenData:
    df_schema = columns=["doc_type", "filename", "ID", "tokens", "length"]

    def __init__(self):
        self.res_df = pd.DataFrame(columns=self.df_schema).astype({"length": "int"})
        self.job_df = pd.DataFrame(columns=self.df_schema).astype({"length": "int"})
        self.train_df = pd.DataFrame(columns=self.df_schema).astype({"length": "int"})

    @staticmethod
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

    @staticmethod
    def tokenize(text):
        stemmer = Stemmer()
        return [
            stemmer.stem(token.lower())
            for token in nltk.word_tokenize(re.sub("\n", "", text.translate(str.maketrans(punctuation, " "*len(punctuation)))))
            if (token.isalnum() and token.lower() not in stopwords.words("english") and len(token) > 1)
        ]

    def pdf_to_df(self, parent_dir, file_glob="*.pdf", df_type="resume"):
        # TODO: Change to Dask for parallel processing and to S3
        df = pd.DataFrame(columns=self.df_schema).astype({"length": "int"})
        for file in glob(os.path.join(parent_dir, file_glob)):
            tokens = self.tokenize(self.extract_pdf_text(os.path.join(parent_dir, file)))
            df = df.append(
                {
                    "doc_type": df_type,
                    "filename": file,
                    "ID": os.path.basename(file),
                    "tokens": tokens,
                    "length": len(tokens),
                },
                ignore_index=True,
            )
        self._update_df(df, df_type)
        return df

    def xls_to_df(self, parent_dir, file_glob="*.xlsx", df_type="train"):
        # TODO: Change to Dask for parallel processing and to S3
        df = pd.DataFrame(columns=self.df_schema).astype({"length": "int"})
        for file in glob(os.path.join(parent_dir, file_glob)):
            df_part = pd.read_excel(os.path.join(file))
            df_part["Description"] = df_part["Description"].apply(
                lambda text: unicodedata.normalize("NFKD", text)
                    .encode("ASCII", "ignore")
                    .decode("utf-8")
            )
            df_part["doc_type"] = "train"
            df_part["filename"] = file
            df_part["tokens"] = df_part["Description"].apply(TokenData.tokenize)
            df_part["length"] = df_part["tokens"].apply(len)
            df_part = df_part[["doc_type", "filename", "ID", "tokens", "length"]].astype({"length": "int"})
            df = pd.concat([df, df_part])
        self._update_df(df, df_type)
        return df

    def _update_df(self, df, df_type):
        if df_type == "resume":
            self.res_df = pd.concat([self.res_df, df])
            return
        if df_type == "job":
            self.job_df = pd.concat([self.job_df, df])
            return
        if df_type == "train":
            self.train_df = pd.concat([self.train_df, df])
            return
        raise ValueError(f"df_type has an invalid value of {df_type}")


