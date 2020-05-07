from ..models.nlpmodels import D2VModel
import pandas as pd
import numpy as np
import s3fs
import tempfile
from gensim.models import Doc2Vec

d2v = D2VModel(vector_size=100, epochs=200)
pd.set_option("display.max_columns", None)
pd.set_option("max_colwidth", 100)
# d2v.build_corpus("s3://csci-e29-kwc271/project/parquet", "job.parquet")
d2v.build_corpus("C:/Users/kerkermui/PycharmProjects/2020sp-vec2rec-crispin-nosidam/vec2rec/data/parquet","job.parquet")
print(len(d2v.train_corpus))
#print(d2v.train_corpus[0])
print(len(d2v.test_corpus))
# d2v.train()
epochs=1000
for epoch in range(0, epochs, 50):
    print("Training epoch %d" % (epoch))
    d2v.train(epochs)

    v2 = d2v.model.infer_vector("I feel good".split())
    if epoch == 0:
        v1 = v2
    if epoch % 10 == 0:
        print(np.linalg.norm(v1-v2))
    v1 = v2
#print(list(d2v.model.docvecs.vectors_docs))
#print(d2v.model.docvecs[0])
print(len(d2v.model.docvecs.vectors_docs))

"""
#sims = d2v.lookup("computer engineer")
#print(f"text lookup {sims}")
#sims = d2v.lookup(filepath="vec2rec/data/resume/1Amy.pdf")
#print(f"pdf lookup {sims}")
sims = d2v.lookup(filepath="vec2rec/data/train/2017-18Hotchkiss_CourseCatalog (simplepdf.com) (edited).xlsx")
print(f"xls lookup {sims}")
for idx, sim in sims:
    print(f"similiarity = {sim} tagdoc = {d2v.train_corpus[idx]}")
    print(f"original df = {d2v.df_train.iloc[idx]}")
    
#print({sim: d2v.df_train.iloc[idx].to_dict() for idx, sim in sims})
#d2v.test()
#s3 = s3fs.S3FileSystem(anon=False)

#d2v.save_model("s3://csci-e29-kwc271/project/models", "job.model")
#d2v.load_model("s3://csci-e29-kwc271/project/models", "job.model")

with tempfile.TemporaryDirectory() as temp_dir:
    d2v.model.save(temp_dir+"job.model")
    s3.put(temp_dir+"job.model", "s3://csci-e29-kwc271/project/models/job.model")

with tempfile.TemporaryDirectory() as temp_dir:
    s3.get("s3://csci-e29-kwc271/project/models/job.model", temp_dir+"job.model")
    d2v.model = Doc2Vec.load(temp_dir+"job.model")
"""
#d2v.test()
