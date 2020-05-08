from ..preprocess.tools import TokenData

td=TokenData()
print(td.s3_glob("s3://csci-e29-kwc271/project/resume/*.pdf"))