import os
import pandas as pd
from ..preprocess.tools import TokenData

#TRAIN_DATA_DIR = os.path.abspath("vec2rec/data/train")
TRAIN_DATA_DIR = "s3://csci-e29-kwc271/project/train"
#RES_DATA_DIR = os.path.abspath("vec2rec/data/resume")
RES_DATA_DIR = "s3://csci-e29-kwc271/project/resume"
#JOB_DATA_DIR = os.path.abspath("vec2rec/data/job")
JOB_DATA_DIR = "s3://csci-e29-kwc271/project/job"
FILE_NAME = "2017-18Hotchkiss_CourseCatalog (simplepdf.com) (edited).xlsx"
pd.set_option("display.max_columns", None)
pd.set_option("max_colwidth", 30)

td = TokenData()
"""
td.xls_to_df(TRAIN_DATA_DIR, df_type="train")
print(td.train_df)
print(td.train_df.compute())
# td.to_parquet("s3://csci-e29-kwc271/project/parquet", df_type="train")
# td.to_parquet("vec2rec/data/parquet", df_type="train")

"""
td.xls_to_df(JOB_DATA_DIR, df_type="job")
print(td.job_df)
td_jobdf = td.job_df.compute()
print(td_jobdf)
# td_jobdf.to_parquet("vec2rec/data/parquet/job1.parquet", compression="gzip")
#td.job_df.to_parquet("vec2rec/data/parquet/job1.parquet", compression="gzip", object_encoding={"tokens": "bytes"})
td.to_parquet("s3://csci-e29-kwc271/project/parquet", df_type="job")

"""
td.pdf_to_df(RES_DATA_DIR)
print(td.res_df)
print("------------------------------")
print(td.res_df.compute())

# td.to_parquet("s3://csci-e29-kwc271/project/parquet", df_type="job")
td.read_parquet("s3://csci-e29-kwc271/project/parquet", df_type="job")
# td.to_parquet("vec2rec/data/parquet")
print(td.job_df.compute())

td.read_parquet("vec2rec/data/parquet")
print(td.job_df.compute())
print(td.res_df.compute())
print(td.train_df.compute())
"""
