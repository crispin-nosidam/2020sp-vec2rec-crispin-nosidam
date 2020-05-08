import dask.dataframe as dd
from vec2rec.preprocess.tools import TokenData
import os.path
import pandas as pd

td = TokenData()
tokens = [td.tokenize( td.extract_pdf_text("vec2rec/data/resume/1Amy.pdf")), td.tokenize( td.extract_pdf_text("vec2rec/data/resume/1Amy.pdf"))]

df = dd.from_delayed(tokens, meta=("data", "O"))
# df["hello"] = 1
# df.to_delayed().compute()
print(df.compute())
"""
df = pd.DataFrame(columns=["a", "b"])
ddf = dd.from_pandas()
"""
