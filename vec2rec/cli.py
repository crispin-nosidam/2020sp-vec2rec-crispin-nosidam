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

import argparse
import os
import pandas as pd
import posixpath
from gensim.models import Doc2Vec
from logging import getLogger, StreamHandler
from .frontend.vec2rec import Vec2Rec, S3_BUCKET_BASE
from .models.nlpmodels import D2VModel
from .preprocess.tools import TokenData

LOCAL_BASE = "/tmp/data"

logger = getLogger(__name__)
logger.addHandler(StreamHandler())
if "LOGGING" in os.environ:
    logger.setLevel(os.environ["LOGGING"])

# need to perform 4 tasks
# Docker 1 - load files from s3, build preprocessed parquet
# Docker 2 - get preprocessed parquet, train model and upload
# Docker 3 - load_model from file, run tests
# CLI - from text or file, lookup from correct repo, display
#     - upload / download file to repo

parser = argparse.ArgumentParser(
    prog=__name__.split(".")[0],
    description="Vec2Rec - Similarity matching among resumes, jobs and training descriptions.",
)
parser.add_argument(
    "-p", "--parent_dir", default=S3_BUCKET_BASE, help="Parent Dir for repo"
)
"""
psr_mx = parser.add_mutually_exclusive_group(required=True)
psr_mx.add_argument("-r", "--resume", action="store_true", help="Action on resumes")
psr_mx.add_argument("-j", "--job", action="store_true", help="Action on jobs")
psr_mx.add_argument("-t", "--train", action="store_true", help="Action on training")
psr_mx.add_argument("-a", "--all", action="store_true", help="Action on all 3 file types")
"""
subparsers = parser.add_subparsers(dest="cmd")
psr_pp = subparsers.add_parser(
    "preprocess", help="Preprocess PDF and Excel for NLP model Training"
)
psr_pp.add_argument(
    "-c", "--chunk", default=20, type=int, help="Chunksize for Dask processing"
)
psr_pp.add_argument(
    "-t",
    "--type",
    choices=["resume", "job", "train", "all"],
    default="all",
    help="Doctype for action",
)
psr_pp.add_argument(
    "-l", "--local", action="store_true", help="Store a copy in local dir"
)
psr_pp.add_argument("-ld", "--local_dir", default=LOCAL_BASE, help="Local dir path")

psr_tn = subparsers.add_parser("train", help="Train NLP Models")
psr_tn.add_argument(
    "-s", "--vector_size", default=75, type=int, help="Vector size for doc2vec"
)
psr_tn.add_argument(
    "-m", "--min_cnt", default=2, type=int, help="Min word occurrence included"
)
psr_tn.add_argument("-e", "--epochs", default=100, type=int, help="# epochs to models")
psr_tn.add_argument(
    "-tr",
    "--test_ratio",
    default=1 / 3,
    type=float,
    help="percentage of  sample used for test case",
)
psr_tn.add_argument(
    "-t",
    "--type",
    choices=["resume", "job", "train", "all"],
    default="all",
    help="Doctype for action",
)
psr_tn.add_argument(
    "-l", "--local", action="store_true", help="Store a copy in local dir"
)
psr_tn.add_argument("-ld", "--local_dir", default=LOCAL_BASE, help="Local dir path")

psr_tt = subparsers.add_parser("test", help="Run tests on NLP Models")
psr_tt.add_argument(
    "-s", "--sample", default=1, type=int, help="Sample size from test_corpus used"
)
psr_tt.add_argument(
    "-n", "--top_n", default=2, type=int, help="Top N similar docs returned"
)
psr_tt.add_argument(
    "-t",
    "--type",
    choices=["resume", "job", "train", "all"],
    default="all",
    help="Doctype for action",
)

psr_lk = subparsers.add_parser("lookup", help="Lookup models for similar documents")
psr_lk.add_argument(
    "-n", "--top_n", default=3, type=int, help="Top N similar docs returned"
)
psr_lk.add_argument(
    "-t",
    "--type",
    choices=["resume", "job", "train"],
    default="job",
    help="Doctype for action",
)
psr_lkmx = psr_lk.add_mutually_exclusive_group(required=True)
psr_lkmx.add_argument("-s", "--string", help="Text string to lookup")
psr_lkmx.add_argument(
    "-f", "--files", nargs="+", help="One or more files concat-ed for lookup"
)

psr_add = subparsers.add_parser("add_doc", help="Add new document to doc repo")
psr_add.add_argument("-f", "--filename", help="Name of file to be added in repo")
psr_add.add_argument(
    "-t",
    "--type",
    choices=["resume", "job", "train"],
    default="job",
    help="Doctype for action",
)

psr_del = subparsers.add_parser("del_doc", help="Delete document from doc repo")
psr_del.add_argument("-f", "--filename", help="Name of file to be deleted from repo")
psr_del.add_argument(
    "-t",
    "--type",
    choices=["resume", "job", "train"],
    default="job",
    help="Doctype for action",
)


def main(args=None):
    args = parser.parse_args(args=args)
    print(args)
    exit()
    if args.cmd is "preprocess":
        to_do = ["resume", "job", "train"] if args.type is "all" else [args.type]
        join = (
            posixpath.join if args.parent_dir.startswith("s3 = s3://") else os.path.join
        )
        td = TokenData()
        for doc_type in to_do:
            # REPO_BASE/<doc_type>/
            raw_doc_dir = join(args.parent_dir, doc_type)

            if doc_type in ["job", "train"]:
                td.xls_to_df(raw_doc_dir, df_type=doc_type)
            if doc_type is "resume":
                td.pdf_to_df(raw_doc_dir, df_type=doc_type)

            # REPO_BASE/parquet/<doc_type>/
            parquet_dir = join(args.parent_dir, "parquet", doc_type)
            td.to_parquet(parquet_dir, df_type=doc_type)

            if args.local:
                # LOCAL_BASE/parquet/<doc_type>/<doc_type>.parquet
                td.to_parquet(
                    os.path.join(args.local_dir, "parquet", doc_type), df_type=doc_type
                )

    if args.cmd is "train":
        to_do = ["resume", "job", "train"] if args.type is "all" else [args.type]
        join = (
            posixpath.join if args.parent_dir.startswith("s3 = s3://") else os.path.join
        )
        for doc_type in to_do:
            Vec2Rec.__dict__[doc_type + "_model"].model = Doc2Vec(
                vector_size=args.vector_size, min_count=args.min_cnt, epochs=args.epochs
            )
            # REPO_BASE/parquet/<doc_type>/
            parquet_dir = join(args.parent_dir, "parquet", doc_type)
            # REPO_BASE/parquet/<doc_type>/<doc_type>.parquet
            Vec2Rec.__dict__[doc_type + "_model"].model.build_corpus(
                parquet_dir, doc_type + ".parquet"
            )
            Vec2Rec.__dict__[doc_type + "_model"].model.train()

            # REPO_BASE/models/<doc_type>/
            model_dir = join(args.parent_dir, "models", doc_type)
            # REPO_BASE/models/<doc_type>/<doc_type>_model
            Vec2Rec.__dict__[doc_type + "_model"].model.save_model(
                model_dir, doc_type + "_model"
            )
            # REPO_BASE/parquet/<doc_type>/<doc_type>_train.parquet
            Vec2Rec.__dict__[doc_type + "_model"].df_train.to_parquet(
                join(parquet_dir, doc_type + "_train.parquet")
            )
            # REPO_BASE/parquet/<doc_type>/<doc_type>_test.parquet
            Vec2Rec.__dict__[doc_type + "_model"].df_test.to_parquet(
                join(parquet_dir, doc_type + "_test.parquet")
            )
            if args.local:
                join = os.path.join
                # LOCAL_BASE/models/<doc_type>/<doc_type>_model
                Vec2Rec.__dict__[doc_type + "_model"].model.save_model(
                    join(args.local_dir, "models", doc_type), doc_type + "_model",
                )
                # LOCAL_BASE/parquet/<doc_type>/<doc_type>_train.parquet
                Vec2Rec.__dict__[doc_type + "_model"].df_train.to_parquet(
                    join(parquet_dir, "parquet", doc_type, doc_type + "_train.parquet")
                )
                # LOCAL_BASE/parquet/<doc_type>/<doc_type>_test.parquet
                Vec2Rec.__dict__[doc_type + "_model"].df_test.to_parquet(
                    join(parquet_dir, "parquet", doc_type, doc_type + "_test.parquet")
                )

    if args.cmd is "test":
        to_do = ["resume", "job", "train"] if args.type is "all" else [args.type]
        join = (
            posixpath.join if args.parent_dir.startswith("s3 = s3://") else os.path.join
        )
        for doc_type in to_do:
            # REPO_BASE/models/<doc_type>/
            model_dir = join(args.parent_dir, "models", doc_type)
            # REPO_BASE/models/<doc_type>/<doc_type>_model
            Vec2Rec.__dict__[doc_type + "_model"].model.load_model(
                model_dir, doc_type + "_model"
            )
            # REPO_BASE/parquet/<doc_type>/
            parquet_dir = join(args.parent_dir, "parquet", doc_type)
            # REPO_BASE/parquet/<doc_type>/<doc_type>_train.parquet
            Vec2Rec.__dict__[
                doc_type + "_model"
            ].model.train_corpus = D2VModel.read_corpus(
                pd.read_parquet(join(parquet_dir, doc_type + "_train.parquet"))
            )
            Vec2Rec.__dict__[
                doc_type + "_model"
            ].model.test_corpus = D2VModel.read_corpus(
                pd.read_parquet(join(parquet_dir, doc_type + "_test.parquet"))
            )
            Vec2Rec.__dict__[doc_type + "_model"].model.test(args.sample, args.top_n)

    if args.cmd is "lookup":
        join = (
            posixpath.join if args.parent_dir.startswith("s3 = s3://") else os.path.join
        )
        # REPO_BASE/models/<doc_type>/
        model_dir = join(args.parent_dir, "models", args.type)
        # REPO_BASE/models/<doc_type>/<doc_type>_model
        Vec2Rec.__dict__[args.type + "_model"].model.load_model(
            model_dir, args.type + "_model"
        )
        if args.string is not None:
            print(
                Vec2Rec.__dict__[args.type + "_model"].model.lookup(
                    text=args.string, top_n=args.top_n
                )
            )
        if args.fileis is not None:
            print(
                Vec2Rec.__dict__[args.type + "_model"].model.lookup(
                    filepath=args.files, top_n=args.top_n
                )
            )

    if args.cmd is "add_doc":
        Vec2Rec.add_doc(args.parent_dir, args.filename, args.type)

    if args.cmd is "del_doc":
        Vec2Rec.del_doc(args.filename, args.type)
