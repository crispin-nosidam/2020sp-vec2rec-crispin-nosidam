import posixpath
import pandas as pd

S3_BUCKET_BASE = "s3://csci-e29-kwc271/project"

print(posixpath.join(S3_BUCKET_BASE, "resume", "abc.txt"))

df = pd.DataFrame({"a": [10], "b":["c"]})
# df["Description"]


class A:
    b=1

c="b"

a = A()
print(A.__dict__[c])
