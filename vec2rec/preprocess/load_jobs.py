import os
import pandas as pd
from vec2rec.preprocess.tools import tokenize

DATA_DIR = os.path.abspath("vec2rec/data/job")
FILE_NAME = "Updated-Job-Descriptions-2018 (edited).xlsx"

df = pd.read_excel(os.path.join(DATA_DIR, FILE_NAME))
df["Job Description"] = df["Job Description"].apply(tokenize)
pd.set_option('display.max_columns', None)
pd.set_option("max_colwidth", 30)
print(df)
print(df.iloc[0]["Job Description"])
print(df["Job Description"].apply(len).mean())
