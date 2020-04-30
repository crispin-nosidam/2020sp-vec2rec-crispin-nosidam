"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will
  cause problems: the code will get executed twice:

  - When you run `python -m vec2rec` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``vec2rec.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``vec2rec.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import os
import argparse
import collections
import random

# from logging import getLogger, StreamHandler, DEBUG, WARN, INFO
from logging import getLogger, StreamHandler

from gensim.models.doc2vec import Doc2Vec

from .preprocess.tools import tokenize
from .train.tools import build_corpus

logger = getLogger(__name__)
logger.addHandler(StreamHandler())
if "LOGGING" in os.environ:
    logger.setLevel(os.environ["LOGGING"])

parser = argparse.ArgumentParser(description="Command description.")
parser.add_argument(
    "-s", "--vector_size", metavar="VECSIZE", default=75, help="Vector size for doc2vec"
)
parser.add_argument(
    "-m",
    "--min_count",
    metavar="MINCNT",
    default=2,
    help="Min occurrence for words to be included",
)
parser.add_argument(
    "-e", "--epochs", metavar="EPOCH", default=40, help="# epochs to train"
)
parser.add_argument(
    "-tr",
    "--test_ratio",
    metavar="TEST_RATIO",
    default=1 / 3,
    help="% sample used for test case",
)
parser.add_argument("-r", "--resume", action="store_true", help="add resume in model")
parser.add_argument("-j", "--job", action="store_true", help="add jobs desc in model")
parser.add_argument(
    "-t", "--train", action="store_true", help="add training desc in model"
)
mxgroup = parser.add_mutually_exclusive_group(required=True)
mxgroup.add_argument("-l", "--line", metavar="LINE", help="Line for similarity search")
mxgroup.add_argument(
    "-f", "--file", metavar="FILENAME", help="Filename for similarity search"
)

DATA_DIR = os.path.abspath("vec2rec/data/resume")
# TODO: put these back to preprocess.load_*.py
RESUME_FILE_PATH = DATA_DIR
JOB_FILE_PATH = ""
TRAIN_FILE_PATH = ""


def main(args=None):
    args = parser.parse_args(args=args)

    file_params = {}
    if args.resume:
        file_params.update({"resume": TRAIN_FILE_PATH})
    if args.job:
        file_params.update({"job": JOB_FILE_PATH})
    if args.resume:
        file_params.update({"resume": RESUME_FILE_PATH})
    if file_params == {}:
        raise ValueError(
            "Please specify at least one file type to be included in model."
        )

    train_corpus, test_corpus = build_corpus(test_ratio=args.test_ratio, **file_params)

    # TODO: Should save training/testing corpus to allow retrain with different parameters

    model = Doc2Vec(
        vector_size=args.vector_size, min_count=args.min_count, epochs=args.epochs
    )

    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    print(f"model.corpus_count = {model.corpus_count}")
    print(f"len(model.docvecs) = {len(model.docvecs)}")

    # TODO: Should save file and load file here, split into 2 functions

    processed_line = tokenize(args.line)
    if len(processed_line) == 0:
        raise ValueError(
            "All words provided are screened by preprocessing, cannot find similarity"
        )
    line_vec = model.infer_vector(processed_line)
    sims = model.docvecs.most_similar([line_vec], topn=len(model.docvecs))
    print(f"top_three = {sims[0:3]}")
    print(f"len(train_corpus) = {len(train_corpus)}")
    print(f"len(test_corpus) = {len(test_corpus)}")
    print(f"len(sims) = {len(sims)}")
    print(f"len(model.docvecs) = {len(model.docvecs)}")
    print(f"type(model.docvecs) = {type(model.docvecs)}")

    for doc_id, _ in sims[0:3]:
        print(f"Document {doc_id}")
        print([tag_doc for tag_doc in train_corpus if tag_doc.tags[0] == doc_id][0])

    print("---------Verifying Training Corpus-------------------")
    # TODO: Move the below to testing, also refactor code
    ranks = []
    second_ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])

    counter = collections.Counter(ranks)
    print(counter)

    print("Document ({}): «{}»\n".format(doc_id, " ".join(train_corpus[doc_id].words)))
    print("SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n" % model)
    for label, index in [
        ("MOST", 0),
        ("SECOND-MOST", 1),
        ("MEDIAN", len(sims) // 2),
        ("LEAST", len(sims) - 1),
    ]:
        print(
            "%s %s: «%s»\n"
            % (label, sims[index], " ".join(train_corpus[sims[index][0]].words))
        )

    # Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(test_corpus) - 1)
    inferred_vector = model.infer_vector(test_corpus[doc_id])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    # Compare and print the most/median/least similar documents from the train corpus
    print("Test Document ({}): «{}»\n".format(doc_id, " ".join(test_corpus[doc_id])))
    print("SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n" % model)
    for label, index in [
        ("MOST", 0),
        ("MEDIAN", len(sims) // 2),
        ("LEAST", len(sims) - 1),
    ]:
        print(
            "%s %s: «%s»\n"
            % (label, sims[index], " ".join(train_corpus[sims[index][0]].words))
        )
