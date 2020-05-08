import os
import pandas as pd
import unicodedata
from vec2rec.preprocess.tools import TokenData

DATA_DIR = os.path.abspath("vec2rec/data/train")
FILE_NAME = "2017-18Hotchkiss_CourseCatalog (simplepdf.com) (edited).xlsx"

df = pd.read_excel(os.path.join(DATA_DIR, FILE_NAME))
df["Description"] = df["Description"].apply(
    lambda text: unicodedata.normalize("NFKD", text)
    .encode("ASCII", "ignore")
    .decode("utf-8")
)
df["tokens"] = df["Description"].apply(TokenData.tokenize)
pd.set_option("display.max_columns", None)
pd.set_option("max_colwidth", 30)
df["df_type"]="models"
df["filename"]=os.path.join(DATA_DIR, FILE_NAME)
df = df[["df_type", "filename", "ID", "tokens"]]
df["length"] = df["tokens"].apply(len)
print(df)
print(df.iloc[0]["tokens"])
print(df["length"].mean())
