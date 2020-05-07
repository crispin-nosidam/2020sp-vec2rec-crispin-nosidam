import boto3
import nltk
import os
import posixpath
import pandas as pd
import re
import s3fs
import unicodedata
import dask.dataframe as dd
import dask.delayed
from glob import glob as os_glob
from krovetzstemmer import Stemmer
from logging import getLogger, StreamHandler
from nltk.corpus import stopwords
from string import punctuation
from urllib.parse import urlparse
from PyPDF4 import PdfFileReader

logger = getLogger(__name__)
logger.addHandler(StreamHandler())
if "LOGGING" in os.environ:
    logger.setLevel(os.environ["LOGGING"])

class Tokenizer:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance

    def __call__(self, text):
        return self.tokenize(text)

    @staticmethod
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


class PDFReader:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance

    def __call__(self, *args, **kwargs):
        return self.extract_pdf_text(*args, **kwargs)

    @staticmethod
    def extract_pdf_text(path, fmt="string"):
        def extract(fileobj):
            pfr = PdfFileReader(fileobj, strict=False)
            text = "" if fmt == "string" else []
            for pg in range(pfr.getNumPages()):
                if fmt == "string":
                    text += pfr.getPage(pg).extractText()
                else:
                    text.append(pfr.getPage(pg).extractText())
            return text
        if path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            up = urlparse(path)
            with s3.open(up.netloc+up.path) as fileobj:
                return extract(fileobj)
        else:
            with open(path, "rb") as fileobj:
                return extract(fileobj)


class TokenData:
    df_schema = ["doc_type", "filename", "ID", "tokens", "length"]
    default_fp = {
        "job": "job.parquet",
        "resume": "resume.parquet",
        "train": "train.parquet",
    }

    def __init__(self, chunk_size=20):
        self.chunk_size = chunk_size
        self.res_df = dd.from_pandas(
            pd.DataFrame(columns=self.df_schema).astype({"length": "int"}),
            chunksize=chunk_size,
        )
        self.job_df = dd.from_pandas(
            pd.DataFrame(columns=self.df_schema).astype({"length": "int"}),
            chunksize=chunk_size,
        )
        self.train_df = dd.from_pandas(
            pd.DataFrame(columns=self.df_schema).astype({"length": "int"}),
            chunksize=chunk_size,
        )

    extract_pdf_text = staticmethod(dask.delayed(PDFReader()))

    tokenize = staticmethod(dask.delayed(Tokenizer()))

    @staticmethod
    def s3_glob(url):
        up = urlparse(url)
        bucket_name = up.netloc
        prefix, suffix = posixpath.split(up.path[1:])
        suffix = posixpath.splitext(suffix)[1]
        s3_client = boto3.client("s3")
        objs = s3_client.list_objects_v2(Bucket=bucket_name)["Contents"]
        return [
            "s3://" + bucket_name + "/" + obj["Key"]
            for obj in objs
            if obj["Key"].startswith(prefix) and obj["Key"].endswith(suffix)
        ]

    def pdf_to_df(self, parent_dir, file_glob="*.pdf", df_type="resume"):
        df = pd.DataFrame(columns=self.df_schema).astype({"length": "int"})
        df = dd.from_pandas(df, chunksize=self.chunk_size)
        join = posixpath.join if parent_dir.startswith("s3://") else os.path.join
        basename = (
            posixpath.basename if parent_dir.startswith("s3://") else os.path.basename
        )
        glob = self.s3_glob if parent_dir.startswith("s3://") else os_glob
        logger.debug("__________ In pdf_to_df __________")
        logger.debug(join(parent_dir, file_glob))
        logger.debug(glob(join(parent_dir, file_glob)))
        for file in glob(join(parent_dir, file_glob)):
            #tokens = self.tokenize(self.extract_pdf_text(join(parent_dir, file)))
            tokens = self.tokenize(self.extract_pdf_text(file))
            df_part = pd.DataFrame(
                {
                    "doc_type": df_type,
                    "filename": file,
                    "ID": basename(file),
                    "tokens": [
                        tokens.compute()
                    ],  # dask.dataframe somehow interfere with nltk
                    "length": 0,
                },
            ).astype({"length": "int"})
            df_part = dd.from_pandas(df_part, chunksize=self.chunk_size)
            df = dd.concat([df, df_part], interleave_partitions=True)
        df["length"] = df["tokens"].apply(len, meta="int")
        self._update_df(df, df_type)
        return df

    def xls_to_df(self, parent_dir, file_glob="*.xlsx", df_type="train"):
        df = pd.DataFrame(columns=self.df_schema).astype({"length": "int"})
        df = dd.from_pandas(df, chunksize=self.chunk_size)
        join = posixpath.join if parent_dir.startswith("s3://") else os.path.join
        glob = self.s3_glob if parent_dir.startswith("s3://") else os_glob
        print(glob(join(parent_dir, file_glob)))
        for file in glob(join(parent_dir, file_glob)):
            # df_part = dd.from_delayed(dask.delayed(pd.read_excel)(os.path.join(file)))
            df_part = dd.from_delayed(dask.delayed(pd.read_excel)(file))
            df_part["Description"] = df_part["Description"].apply(
                lambda text: unicodedata.normalize("NFKD", text)
                .encode("ASCII", "ignore")
                .decode("utf-8"),
                meta="str",
            )
            df_part["doc_type"] = df_type
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

    def read_parquet(self, parent_dir, file_path=default_fp, df_type="all"):
        join = posixpath.join if parent_dir.startswith("s3://") else os.path.join
        if df_type in ["all", "resume"]:
            self.res_df = dd.read_parquet(join(parent_dir, file_path["resume"]))
        if df_type in ["all", "job"]:
            self.job_df = dd.read_parquet(join(parent_dir, file_path["job"]))
        if df_type in ["all", "train"]:
            self.train_df = dd.read_parquet(join(parent_dir, file_path["train"]))

    def to_parquet(self, parent_dir, file_path=default_fp, df_type="all"):
        join = posixpath.join if parent_dir.startswith("s3://") else os.path.join
        if df_type in ["all", "resume"]:
            # encountered bugs here
            # self.res_df.to_parquet(os.path.join(parent_dir, file_path["resume"]), compute=True)
            self.res_df.compute().to_parquet(
                join(parent_dir, file_path["resume"]), compression="gzip"
            )
        if df_type in ["all", "job"]:
            # self.job_df.to_parquet(os.path.join(parent_dir, file_path["job"]), compute=True)
            self.job_df.compute().to_parquet(
                join(parent_dir, file_path["job"]), compression="gzip"
            )
        if df_type in ["all", "train"]:
            # self.train_df.to_parquet(os.path.join(parent_dir, file_path["train"]), compute=True)
            self.train_df.compute().to_parquet(
                join(parent_dir, file_path["train"]), compression="gzip"
            )
