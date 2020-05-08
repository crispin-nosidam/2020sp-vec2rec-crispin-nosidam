print(not True ^ True)
print(not True ^ False)
print(not False ^ False)

a = []
b = "a"
if type(a) is list:
    print("list")

import posixpath

url = "s3://host/haha/hehe.ext"

print(posixpath.splitext(url))
url = "C:/host/haha/hehe.ext"
print(posixpath.splitext(url))
url = "C:\\host\\haha\\hehe.xlsaskjasd"
print(posixpath.splitext(url))
print(posixpath.splitext(url)[1][0:4])


import pandas as pd

df = pd.DataFrame({"a": ["haha", "hehe"], "b": ["boo","bleh"]})
print(df)
print(df.apply(lambda x: x + " ").agg("sum"))
