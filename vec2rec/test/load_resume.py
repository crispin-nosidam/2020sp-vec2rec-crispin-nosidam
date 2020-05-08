import pandas as pd
import os
from vec2rec.preprocess.tools import pdf_to_dataframe

DATA_DIR = os.path.abspath("vec2rec/data/resume")

df = pdf_to_dataframe(DATA_DIR)
pd.set_option('display.max_columns', None)
pd.set_option("max_colwidth", 30)
print(df)
print(df.dtypes)
print(df.length.mean())
