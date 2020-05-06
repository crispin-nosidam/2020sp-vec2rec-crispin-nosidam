import os
import pandas as pd
import posixpath
import s3fs
from ..preprocess.tools import PDFReader
from ..models.nlpmodels import D2VModel

S3_BUCKET_BASE = "s3://csci-e29-kwc271/project"


# TODO: Implement Factory?
class Vec2Rec:
    job_model = D2VModel()
    resume_model = D2VModel()
    train_model = D2VModel()

    # TODO: Test this
    @staticmethod
    def add_doc(parent_dir, file_name, type="resume"):
        ext = posixpath.splitext(file_name)[1][0:4]
        if ext not in [".pdf", ".xls"]:
            raise TypeError("Only pdf or xls files are supported")
        # Test reading the files to see if error is thrown
        if ext == ".pdf":
            PDFReader.extract_pdf_text(file_name)
        if ext == ".xls":
            # excel files must have the column Description
            try:
                pd.read_excel(file_name)["Description"]
            except KeyError:
                raise KeyError("Input file in Excel format must have a column called 'Description'")
        s3 = s3fs.S3FileSystem(anon=False)
        s3.put(
            os.path.join(parent_dir, file_name),
            posixpath.join(S3_BUCKET_BASE, type, file_name),
        )

    # TODO: Test this
    @staticmethod
    def del_doc(file_name, type="resume"):
        s3 = s3fs.S3FileSystem(anon=False)
        s3.rm(posixpath.join(S3_BUCKET_BASE, type, file_name))
