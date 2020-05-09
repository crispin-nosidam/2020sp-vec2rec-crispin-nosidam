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
    "-p", "--parent_dir", default=S3_BUCKET_BASE, help="Parent dir for repo / file to be processed"
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
    logger.debug(args)
    # exit()
    to_do = ["resume", "job", "train"] if args.type == "all" else [args.type]
    join = posixpath.join if args.parent_dir.startswith("s3://") else os.path.join
    if not args.parent_dir.startswith("s3://"):
        args.parent_dir = os.path.abspath(args.parent_dir)
    if args.cmd == "preprocess":
        logger.debug("__________ in preprocess __________")
        td = TokenData()
        for doc_type in to_do:
            logger.debug(f"doc_type = {doc_type}")
            # REPO_BASE/<doc_type>/
            raw_doc_dir = join(args.parent_dir, doc_type)
            logger.debug(f"raw_doc_dir = {raw_doc_dir}")

            if doc_type in ["job", "train"]:
                td.xls_to_df(raw_doc_dir, df_type=doc_type)
            if doc_type == "resume":
                td.pdf_to_df(raw_doc_dir, df_type=doc_type)

            # REPO_BASE/parquet/
            parquet_dir = join(args.parent_dir, "parquet")
            logger.debug(f"parquet_dir = {parquet_dir}")
            td.to_parquet(parquet_dir, df_type=doc_type)

            if args.local:
                # LOCAL_BASE/parquet/<doc_type>.parquet
                args.local_dir = os.path.abspath(args.local_dir)
                local_parquet_dir = os.path.join(args.local_dir, "parquet")
                if not os.path.exists(local_parquet_dir):
                    os.makedirs(local_parquet_dir)
                logger.debug(local_parquet_dir)
                td.to_parquet(local_parquet_dir, df_type=doc_type)

    if args.cmd == "train":
        for doc_type in to_do:
            logger.debug("__________ in train __________")
            logger.debug(f"doc_type = {doc_type}")
            logger.debug(Vec2Rec.__dict__[doc_type + "_model"].model)
            Vec2Rec.__dict__[doc_type + "_model"].model = Doc2Vec(
                vector_size=args.vector_size, min_count=args.min_cnt, epochs=args.epochs
            )
            # REPO_BASE/parquet/
            parquet_dir = join(args.parent_dir, "parquet")
            logger.debug(f"parquet_dir = {parquet_dir}")
            # REPO_BASE/parquet/<doc_type>.parquet
            Vec2Rec.__dict__[doc_type + "_model"].build_corpus(
                parquet_dir, doc_type + ".parquet"
            )
            Vec2Rec.__dict__[doc_type + "_model"].train()
            logger.debug(f'trained = {Vec2Rec.__dict__[doc_type + "_model"].trained}')

            # REPO_BASE/models/
            model_dir = join(args.parent_dir, "models")
            logger.debug(f"model_dir = {model_dir}")
            # REPO_BASE/models/<doc_type>_model
            Vec2Rec.__dict__[doc_type + "_model"].save_model(
                model_dir, doc_type + "_model"
            )
            # REPO_BASE/parquet/<doc_type>_train.parquet
            Vec2Rec.__dict__[doc_type + "_model"].df_train.to_parquet(
                join(parquet_dir, doc_type + "_train.parquet"), compression="gzip"
            )
            # REPO_BASE/parquet/<doc_type>_test.parquet
            Vec2Rec.__dict__[doc_type + "_model"].df_test.to_parquet(
                join(parquet_dir, doc_type + "_test.parquet"), compression="gzip"
            )
            if args.local:
                args.local_dir = os.path.abspath(args.local_dir)
                local_model_dir = os.path.join(args.local_dir, "models")
                # LOCAL_BASE/models/<doc_type>_model
                logger.debug(f"local_model_dir = {local_model_dir}")
                if not os.path.exists(local_model_dir):
                    os.makedirs(local_model_dir)
                Vec2Rec.__dict__[doc_type + "_model"].save_model(
                    local_model_dir, doc_type + "_model",
                )
                # LOCAL_BASE/parquet/<doc_type>_train.parquet
                logger.debug(
                    f'local_parquet_dir = {os.path.join(args.local_dir, "parquet")}'
                )
                Vec2Rec.__dict__[doc_type + "_model"].df_train.to_parquet(
                    os.path.join(
                        args.local_dir, "parquet", doc_type + "_train.parquet"
                    ),
                    compression="gzip",
                )
                # LOCAL_BASE/parquet/<doc_type>_test.parquet
                Vec2Rec.__dict__[doc_type + "_model"].df_test.to_parquet(
                    os.path.join(args.local_dir, "parquet", doc_type + "_test.parquet"),
                    compression="gzip",
                )

    if args.cmd == "test":
        for doc_type in to_do:
            print(f"---------- For Doc Type {doc_type} ----------")
            logger.debug("__________ in test __________")
            logger.debug(f"doc_type = {doc_type}")
            # REPO_BASE/models/
            model_dir = join(args.parent_dir, "models")
            logger.debug(f"model_dir = {model_dir}")
            # REPO_BASE/models/<doc_type>_model
            Vec2Rec.__dict__[doc_type + "_model"].load_model(
                model_dir, doc_type + "_model"
            )
            # REPO_BASE/parquet/
            parquet_dir = join(args.parent_dir, "parquet")
            logger.debug(f"parquet_dir = {parquet_dir}")
            # REPO_BASE/parquet/<doc_type>_train.parquet
            Vec2Rec.__dict__[doc_type + "_model"].train_corpus = list(
                D2VModel.read_corpus(
                    pd.read_parquet(join(parquet_dir, doc_type + "_train.parquet"))
                )
            )
            Vec2Rec.__dict__[doc_type + "_model"].test_corpus = list(
                D2VModel.read_corpus(
                    pd.read_parquet(join(parquet_dir, doc_type + "_test.parquet")),
                    token_only=True,
                )
            )
            Vec2Rec.__dict__[doc_type + "_model"].test(args.sample, args.top_n)

    if args.cmd == "lookup":
        # REPO_BASE/models/
        model_dir = join(args.parent_dir, "models")
        # REPO_BASE/models/<doc_type>_model
        Vec2Rec.__dict__[args.type + "_model"].load_model(
            model_dir, args.type + "_model"
        )
        # REPO_BASE/parquet/
        parquet_dir = join(args.parent_dir, "parquet")
        logger.debug(f"parquet_dir = {parquet_dir}")
        # REPO_BASE/parquet/<doc_type>_train.parquet
        Vec2Rec.__dict__[args.type + "_model"].df_train = pd.read_parquet(
            join(parquet_dir, args.type + "_train.parquet")
        )

        if args.string is not None:
            param = {"text": args.string}
        if args.files is not None:
            param = {"filepath": args.files}
        n = 0
        for sim, metadata in (
            Vec2Rec.__dict__[args.type + "_model"]
            .lookup(top_n=args.top_n, **param)
            .items()
        ):
            print(f"---------- Top {n+1} similarity ----------")
            print(f"Similarity: {sim}")
            for key, value in metadata.items():
                if key != "tokens":
                    print(f"{key} = {value}")
                else:
                    print(f"{key} = {' '.join(value)}")

            n += 1

    if args.cmd == "add_doc":
        Vec2Rec.add_doc(args.parent_dir, args.filename, args.type)

    if args.cmd == "del_doc":
        Vec2Rec.del_doc(args.filename, args.type)
