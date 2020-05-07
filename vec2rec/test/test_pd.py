import pandas as pd
import os
pd.set_option("display.max_columns", None)
pd.set_option("max_colwidth", 30)
df_type="resume"
file="C:/Users/kerkermui/PycharmProjects/2020sp-vec2rec-crispin-nosidam/vec2rec/data/resume/1Amy.pdf"
tokens = ["a", "b", "c"]
df_schema = columns = ["doc_type", "filename", "ID", "tokens", "length"]

df_part = pd.DataFrame([{
    "doc_type": df_type,
    "filename": file,
    "ID": os.path.basename(file),
    "tokens": None,
    "length": "0"}],
    ).astype({"length": "int"})
print(df_part.dtypes)
#df_part.at[0, "tokens"] = tokens
df_part["tokens"] = [tokens]
print(df_part)
df = pd.DataFrame({'A': [12, 23], 'B': [['a', 'b'], ['c', 'd']]})
df.at[1, 'B'] = tokens
print(df)
