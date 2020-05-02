import nltk
import os
import posixpath
import pandas as pd
import re
import unicodedata
import dask.dataframe as dd
import dask.delayed
from glob import glob
from krovetzstemmer import Stemmer
from nltk.corpus import stopwords
from string import punctuation
from PyPDF4 import PdfFileReader


class TokenData:
    df_schema = ["doc_type", "filename", "ID", "tokens", "length"]
    default_fp = {
        "job": "job.parquet",
        "resume": "resume.parquet",
        "train": "train.parquet",
    }
    res_df = dd.from_pandas(
        pd.DataFrame(columns=df_schema).astype({"length": "int"}), chunksize=20
    )
    job_df = dd.from_pandas(
        pd.DataFrame(columns=df_schema).astype({"length": "int"}), chunksize=20
    )
    train_df = dd.from_pandas(
        pd.DataFrame(columns=df_schema).astype({"length": "int"}), chunksize=20
    )

    @staticmethod
    @dask.delayed
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
    @dask.delayed
    def tokenize(text):
        stemmer = Stemmer()
        return [
            stemmer.stem(token.lower())
            for token in nltk.word_tokenize(
                re.sub(
                    "\n",
                    "",
                    text.translate(str.maketrans(punctuation, " " * len(punctuation))),
                )
            )
            if (
                token.isalnum()
                and token.lower() not in stopwords.words("english")
                and len(token) > 1
            )
        ]

    def pdf_to_df(self, parent_dir, file_glob="*.pdf", df_type="resume"):
        df = pd.DataFrame(columns=self.df_schema).astype({"length": "int"})
        df = dd.from_pandas(df, chunksize=20)
        for file in glob(os.path.join(parent_dir, file_glob)):
            tokens = self.tokenize(
                self.extract_pdf_text(os.path.join(parent_dir, file))
            )
            df_part = pd.DataFrame({
                "doc_type": df_type,
                "filename": file,
                "ID": os.path.basename(file),
                "tokens": [tokens.compute()],  # dask.dataframe somehow interfere with nltk
                "length": 0},
                ).astype({"length": "int"})
            df_part = dd.from_pandas(df_part, chunksize=20)
            df = dd.concat([df, df_part], interleave_partitions=True)
        df["length"] = df["tokens"].apply(len, meta="int")
        self._update_df(df, df_type)
        return df

    def xls_to_df(self, parent_dir, file_glob="*.xlsx", df_type="train"):
        df = pd.DataFrame(columns=self.df_schema).astype({"length": "int"})
        df = dd.from_pandas(df, chunksize=20)
        for file in glob(os.path.join(parent_dir, file_glob)):
            df_part = dd.from_delayed(dask.delayed(pd.read_excel)(os.path.join(file)))
            df_part["Description"] = df_part["Description"].apply(
                lambda text: unicodedata.normalize("NFKD", text)
                .encode("ASCII", "ignore")
                .decode("utf-8"),
                meta="str",
            )
            df_part["doc_type"] = "train"
            df_part["filename"] = file
            df_part["tokens"] = df_part["Description"].apply(
                TokenData.tokenize, meta="list"
            )
            df_part["length"] = df_part["tokens"].apply(len, meta="int")
            df_part = df_part[
                ["doc_type", "filename", "ID", "tokens", "length"]
            ].astype({"length": "int"})
            df = dd.concat([df, df_part], interleave_partitions=True)
        self._update_df(df, df_type)
        return df

    def _update_df(self, df, df_type):
        if df_type == "resume":
            self.res_df = dd.concat([self.res_df, df], interleave_partitions=True)
            return
        if df_type == "job":
            self.job_df = dd.concat([self.job_df, df], interleave_partitions=True)
            return
        if df_type == "train":
            self.train_df = dd.concat([self.train_df, df], interleave_partitions=True)
            return
        raise ValueError(f"df_type has an invalid value of {df_type}")

    @classmethod
    def read_parquet(cls, parent_dir, file_path=default_fp, df_type="all"):
        join = posixpath.join if parent_dir.startswith("s3://") else os.path.join
        if df_type in ["all", "resume"]:
            cls.res_df = dd.read_parquet(join(parent_dir, file_path["resume"]))
        if df_type in ["all", "job"]:
            cls.job_df = dd.read_parquet(join(parent_dir, file_path["job"]))
        if df_type in ["all", "train"]:
            cls.train_df = dd.read_parquet(join(parent_dir, file_path["train"]))

    def to_parquet(self, parent_dir, file_path=default_fp, df_type="all"):
        join = posixpath.join if parent_dir.startwith("s3://") else os.path.join
        if df_type in ["all", "resume"]:
            # encountered bugs here
            # self.res_df.to_parquet(os.path.join(parent_dir, file_path["resume"]), compute=True)
            self.res_df.compute().to_parquet(join(parent_dir, file_path["resume"]), compression="gzip")
        if df_type in ["all", "job"]:
            # self.job_df.to_parquet(os.path.join(parent_dir, file_path["job"]), compute=True)
            self.job_df.compute().to_parquet(join(parent_dir, file_path["job"]), compression="gzip")
        if df_type in ["all", "train"]:
            # self.train_df.to_parquet(os.path.join(parent_dir, file_path["train"]), compute=True)
            self.train_df.compute().to_parquet(join(parent_dir, file_path["train"]), compression="gzip")
