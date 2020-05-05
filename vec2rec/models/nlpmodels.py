import os
import pandas as pd
import posixpath
from logging import getLogger, StreamHandler
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.model_selection import train_test_split

logger = getLogger(__name__)
logger.addHandler(StreamHandler())
if "LOGGING" in os.environ:
    logger.setLevel(os.environ["LOGGING"])


# TODO: Is it Correct to Change NLPModel to Abstract Class?
class NLPModel:
    def build_corpus(self, parent_dir, path):
        raise NotImplementedError

    def train(self, epochs=None):
        raise NotImplementedError

    def test(self, top_n=3):
        raise NotImplementedError

    def load_model(self, parent_dir, file_path):
        raise NotImplementedError

    def save_model(self, parent_dir, file_path):
        raise NotImplementedError

    def lookup(self, text, filepath, top_n, return_type="text"):
        raise NotImplementedError


class D2VModel(NLPModel):
    def __init__(self, vector_size=75, min_count=2, epochs=40):
        self.train_corpus = None
        self.test_corpus = None
        self.model = Doc2Vec(
            vector_size=vector_size, min_count=min_count, epochs=epochs
        )

    @staticmethod
    def read_corpus(df, token_only=False):
        n = 0
        for _, row in df.iterrows():
            row_list = row.tolist()
            if token_only:
                yield row_list[3]
            else:
                yield TaggedDocument(row_list[3], [n] + row_list[0:3])
                n += 1

    # classmethod?
    def build_corpus(self, parent_dir, file_path, test_ratio=1 / 3):
        # TODO: refactor load_*.py into functions to be used here
        join = posixpath.join if parent_dir.startswith("s3://") else os.path.join
        df = pd.read_parquet(join(parent_dir, file_path))

        df_train, df_test = train_test_split(
            df, test_size=test_ratio, random_state=42
        )

        self.train_corpus = list(self.read_corpus(df_train))
        self.test_corpus = list(
            self.read_corpus(df_test, token_only=True)
        )
        print(self.train_corpus)
        print(self.test_corpus)

    def train(self, epochs=None):
        if epochs is None:
            epochs = self.model.epochs

        self.model.build_vocab(self.train_corpus)
        self.model.train(
            self.train_corpus, total_examples=self.model.corpus_count, epochs=epochs
        )

    def test(self, top_n=3):
        pass

    # from files load to self.corpus
    def load_model(self, parent_dir, file_path):
        pass

    def save_model(self, parent_dir, file_path):
        pass

    def lookup(self, text, filepath, top_n, return_type="text"):
        pass

