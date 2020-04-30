import os
from logging import getLogger, StreamHandler
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.model_selection import train_test_split
from ..preprocess.tools import pdf_to_dataframe

logger = getLogger(__name__)
logger.addHandler(StreamHandler())
if "LOGGING" in os.environ:
    logger.setLevel(os.environ["LOGGING"])


def read_corpus(df, token_only=False):
    n = 0
    for _, row in df.iterrows():
        if token_only:
            yield row.tolist()[0]
        else:
            yield TaggedDocument(row.tolist()[0], [n])
            n += 1


def build_corpus(line=None, job=None, resume=None, training=None, test_ratio=1 / 3):
    # TODO: refactor load_*.py into functions to be used here
    df_res = pdf_to_dataframe(resume).tokens

    df_res_train, df_res_test = train_test_split(
        df_res, test_size=test_ratio, random_state=42
    )

    train_corpus = list(read_corpus(df_res_train.to_frame()))
    test_corpus = list(read_corpus(df_res_test.to_frame(), token_only=True))

    return train_corpus, test_corpus


def train_doc2vec(train_corpus, file_path, to_file=False, vector_size=75, min_count=2, epochs=40):
    model = Doc2Vec( vector_size=vector_size, min_count=min_count, epochs=epochs )

    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
