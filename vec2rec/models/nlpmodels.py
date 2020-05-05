import collections
import os
import pandas as pd
import posixpath
import random
import s3fs
import unicodedata
from logging import getLogger, StreamHandler
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.model_selection import train_test_split
from tempfile import TemporaryDirectory
from ..preprocess.tools import Tokenizer, PDFReader

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

    def test(self, sample=1, top_n=2):
        raise NotImplementedError

    def load_model(self, parent_dir, file_path):
        raise NotImplementedError

    def save_model(self, parent_dir, file_path):
        raise NotImplementedError

    def lookup(self, text, filepath, top_n):
        raise NotImplementedError


class D2VModel(NLPModel):
    def __init__(self, vector_size=75, min_count=2, epochs=40):
        self.train_corpus = None
        self.test_corpus = None
        self.df_train = None
        self.df_test = None
        self.trained = False
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
                # avoid using multi-tag
                # yield TaggedDocument(row_list[3], [n] + row_list[0:3])
                yield TaggedDocument(row_list[3], [n])
                n += 1

    # classmethod?
    def build_corpus(self, parent_dir, file_path, test_ratio=1 / 3):
        # Load from Parquet files with all preprocessed data
        join = posixpath.join if parent_dir.startswith("s3://") else os.path.join
        df = pd.read_parquet(join(parent_dir, file_path))

        self.df_train, self.df_test = train_test_split(
            df, test_size=test_ratio, random_state=42
        )

        self.train_corpus = list(self.read_corpus(self.df_train))
        self.test_corpus = list(self.read_corpus(self.df_test, token_only=True))

    def train(self, epochs=None):
        # class method defaults cannot use class var inline
        if epochs is None:
            epochs = self.model.epochs

        self.model.build_vocab(self.train_corpus)
        self.model.train(
            self.train_corpus, total_examples=self.model.corpus_count, epochs=epochs
        )
        self.trained = True

    def test(self, sample=1, top_n=2):
        # Testing code from https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#assessing-the-model
        ranks = []
        for doc_id in range(len(self.train_corpus)):
            inferred_vector = self.model.infer_vector(self.train_corpus[doc_id].words)
            sims = self.model.docvecs.most_similar(
                [inferred_vector], topn=len(self.model.docvecs)
            )
            rank = [docid for docid, sim in sims].index(doc_id)
            ranks.append(rank)

        counter = collections.Counter(ranks)
        print("---------- Verifying with Training Corpus ----------")
        print(f"Out of {len(self.train_corpus)} samples in the training corpus")
        for rank, count in counter.most_common():
            print(
                f"{count} is classfied as {'most' if rank is 0 else 'top '+str(rank+1)+' sample'} similar to itself"
            )

        print(f"\n---------- Similarity with Testing Corpus ----------")
        doc_id = random.randint(0, len(self.test_corpus) - 1)
        inferred_vector = self.model.infer_vector(self.test_corpus[doc_id])
        sims = self.model.docvecs.most_similar(
            [inferred_vector], topn=len(self.model.docvecs)
        )

        # Compare and print the most/median/least similar documents from the models corpus
        print(
            "Test Document ({}): «{}»\n".format(
                doc_id, " ".join(self.test_corpus[doc_id])
            )
        )
        print("SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n" % self.model)
        for label, index in [
            ("MOST", 0),
            ("MEDIAN", len(sims) // 2),
            ("LEAST", len(sims) - 1),
        ]:
            print(
                "%s %s: «%s»\n"
                % (
                    label,
                    sims[index],
                    " ".join(self.train_corpus[sims[index][0]].words),
                )
            )

    # from files load to self.corpus
    def load_model(self, parent_dir, file_name="model"):
        if parent_dir.startswith("s3://"):
            with TemporaryDirectory() as temp_dir:
                s3 = s3fs.S3FileSystem(anon=False)
                s3.get(
                    posixpath.join(parent_dir, file_name),
                    os.path.join(temp_dir, file_name),
                )
                self.model = Doc2Vec.load(os.path.join(temp_dir, file_name))
        else:
            self.model.save(os.path.join(parent_dir, file_name))

    def save_model(self, parent_dir, file_name):
        if parent_dir.startswith("s3://"):
            with TemporaryDirectory() as temp_dir:
                s3 = s3fs.S3FileSystem(anon=False)
                self.model.save(os.path.join(temp_dir, file_name))
                s3.put(
                    os.path.join(temp_dir, file_name),
                    posixpath.join(parent_dir, file_name),
                )
        else:
            self.model.save(os.path.join(parent_dir, file_name))

    # file path can be a list
    def lookup(self, text=None, filepath=None, top_n=3):
        # valid options: text only - convert to list to lookup
        # valid options: file only - convert to list to lookup
        # valid options: multiple files - concat convert to list to lookup
        # if excel, must have a column called "Description".
        # Text will be concat-ed if there are more 1 row in excel
        if not ((text is None) ^ (filepath is None)):
            raise ValueError(
                "Must specify one of text or filepath param, but not both."
            )

        tokens = []
        if text is not None:
            tokens = Tokenizer.tokenize(text)
        else:
            if type(filepath) is not list:
                filepath = [filepath]
            for file in filepath:
                ext = posixpath.splitext(file)[1][0:4]
                if ext not in [".pdf", ".xls"]:
                    raise TypeError("Only pdf or xls files are supported")
                if ext == ".pdf":
                    raw_line = PDFReader.extract_pdf_text(file)
                if ext == ".xls":
                    # expects a column called "Description", convert char into ASCII
                    # add a space at the end of each line, then concat together
                    raw_line = (
                        pd.read_excel(file)["Description"]
                        .apply(
                            lambda txt: unicodedata.normalize("NFKD", txt)
                            .encode("ASCII", "ignore")
                            .decode("utf-8")
                        )
                        .apply(lambda x: x + " ")
                        .agg("sum")
                    )
                tokens = tokens + Tokenizer.tokenize(raw_line)
        if len(tokens) == 0:
            raise ValueError(
                "Words provided are too generic and are all screened by preprocessing, cannot find similarity"
            )

        line_vec = self.model.infer_vector(tokens)
        sims = self.model.docvecs.most_similar([line_vec], topn=top_n)
        # with the index in sims, join back the metadata and return
        return {sim: self.df_train.iloc[idx].to_dict() for idx, sim in sims}
