# vec2rec
#### A Recommendation Engine for Job Seekers, Headhunters and Training Professionals Alike!
###### by Michael Choi Kwong Chak

## Goal
All of us looked for a job before. We also search training courses to equip ourselves for the job market. Some of us also try to find suitable candidates for their team or try to train up their staff. What challenges do these tasks have in common? With limited time to search, we have:
1.	An explosive amount of choices
2.	Choices that are mostly similar to each other
3.	Data, while somewhat categorized, the differentiators are in words, i.e.: natural languages
4.	Different preferences with different background – the best paid Java programmer position may be easy to search, but are you up to it?

A description of yourself, i.e.: a resume, is ultimately what gets you into an interview, after the broad-stroke categorizations like “Java programmer”. Keyword search engines like google are common, but what if you can search with your resume?
Similarity, a search with job description for the best candidate, or the relevancy of a training to a job we want would be nice.
Vec2rec is a recommendation engine that, with natural language processing, enable us to search with a description, be it a resume, a job description or a training description for most relevant results.
In addition, we can also run **what-if scenarios**, e.g.: with my current resume, how much closer would it be for me to a dream job if I take this training course? 

## Use Cases
* Job Seekers
  * Find the most suitable jobs
* Headhunters
  * Find the most suitable candidates for a job
* Training Professionals / Managers / Job Seekers
  * Most relevant trainings to a job
  * What jobs a training can enable
  * Increment of a candidate’s suitability for a job after a training

## Architecture
![Architecture Diagram](/vec2rec/images/arch_diag.png)

## Technology Stack
* Gensim Doc2vec
* Python – Descriptors, Iterators
* Kubflow Pipeline and Docker
* Dask Dataframe, Dask Delay, Pandas and Parquet
* Natural Language toolkit (NLTK) and Krovetz Stemmer, PyPDF4
* S3
* Argparse (Future: Flask)

## Design Choices & Implementation Considerations
### 1) Preserve computation steps to allow easy reruns under different config / data updates
* Kubeflow pipeline allows easy reruns for each step.
* The serialization of the 3 generated artifact types minimize the number of reruns needed
  * For activities such as
    * Addition/Remove of data
    * Retrain with different parameters
    * Restarts
  * Features
    * Generic processed data can be updated incrementally
    * Doc2Vec formatted data cannot be updated incrementally but left provision for future enhancement for serialization method such as pickle
    * Segregated model avoid total retrain for doc types on changes
    * Saved Models avoid total recalculation of models over restarts
### 2) Modularize components and enable future enhancement / replacement
* Docker phases in Kubeflow allow replacement for whole phases
* Usage of Descriptor in PDF scrapper, Stemmer, data cleaning modules, even the main engine Doc2vec, allow the easy replacement of these modules
### 3) Enhance parallelization on computation and Memory Efficiency
* Use of Dask Dataframe and Dask Delayed increases parallelization
* Some attempts are made to reduce memory footprint during preprocessing by using iterators, but Doc2vec requires whole corpus to be in memory during training
* Temporary training data in Doc2vec model is deleted after training to reduce memory full print for lookup engine
### 4) Allow incremental growth to lookup database
* Interface is added to modify raw data of lookup database
* Allow Incremental updates up to generic preprocessed data
* Gensim Doc2vec does not allow incremental update of models
### 5) Centralized Repository
* S3 being the most easily implemented repository for document-type raw data
* Database may have better performance as interim data storage, but still cannot store saved models
## Components
### Batch processing – to produce Gensim Doc2Vec models for similarity lookups
* There are 4 types of artifacts, raw data, preprocessed generic data, Gensim Doc2vec formatted training and testing data, and trained model(s). All of these are stored on S3.
* The Kubeflow Pipeline enables modularization and reruns. Each stage is a docker which can be replaced, even with non-python dockers as long as it produces results with correct format

#### Kubeflow Pipeline Phases
The following are all dockers images uploaded to DockerHub. The job definitions are written in Python which are compiled with the Kubeflow domain specific compiler (DSL) into a yaml file, which can be uploaded into the Kubeflow cluster for job definition. Each step can either pass “small” variables, usually int, str, float, bool. For larger data will have to serialize and Kubeflow will help to copy to the correct path location to be retrieved by the next phase. 
* **Generic preprocessing phase** converts raw data, such as Excel and PDF into Pandas Dataframe, which is computed with Dask Dataframe / Dask Delayed, stored as Parquet
  * File format converter is implemented as descriptor to be easily replaceable
  * Preprocessing includes conversion to lower case, UTF-8 charset, removal of NLTK stopwords and punctuations, and the use of Krovetz Stemmer, which seems to have better conversion results than some others like the NLTK stemming or lemmatization; dropping of short and infrequent words
  * Preprocessing is implemented as a descriptor to be easily replaceable
* **Gensim Doc2vec preprocessing phase** split training/testing set in dataframes on Parquet and convert into Gensim Doc2Vec Training and Testing Corpus
  * Specifically detached from the generic phase to allow Doc2vec engine to be replaced
  * Final point where incremental updates are possible as Doc2vec models are not.
  * Currently merged w/ training phase as corpus serialization is not implemented.
* **Gensim Doc2vec training phase** builds the doc2vec model from training corpus
  * Currently, incremental update is not supported by Doc2Vec the model needs to be retrained whenever there are additions/removal to corpus
  * 4 models are built
    * Models from each data type – resumes, job desc, train desc
      * Better retrain performance
    * Model with all data meshed together (Future Enhancement)
      * Larger sample size, more complete vocabulary
* **Gensim Doc2vec testing phase** uses the both the training data and testing data to evaluate the model performance. This phase is not exposed to the user.
  * Training data: should have best similarity to itself
  * Testing data: in this project, eyeball verification is employed though more sophisticated methods are available
 
![Preprocess Phase](/vec2rec/images/preprocess.png)

![Training Phase](/vec2rec/images/train.png)

![Testing Phase](/vec2rec/images/test.png)

### Front end – for Similarity Queries
* Includes
  * CLI Python Module with argparse
  * Flask API (Future Enhancement)

#### Functions of CLI:
##### Top Level Options
```text
usage: vec2rec [-h] [-p PARENT_DIR]
               {preprocess,train,test,lookup,add_doc,del_doc} ...

Vec2Rec - Similarity matching among resumes, jobs and training descriptions.

positional arguments:
  {preprocess,train,test,lookup,add_doc,del_doc}
    preprocess          Preprocess PDF and Excel for NLP model Training
    train               Train NLP Models
    test                Run tests on NLP Models
    lookup              Lookup models for similar documents
    add_doc             Add new document to doc repo
    del_doc             Delete document from doc repo

optional arguments:
  -h, --help            show this help message and exit
  -p PARENT_DIR, --parent_dir PARENT_DIR
                        Parent dir for repo / file to be processed
```
##### Sub command options
```text

usage: vec2rec preprocess [-h] [-c CHUNK] [-t {resume,job,train,all}] [-l]
                          [-ld LOCAL_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -c CHUNK, --chunk CHUNK
                        Chunksize for Dask processing
  -t {resume,job,train,all}, --type {resume,job,train,all}
                        Doctype for action
  -l, --local           Store a copy in local dir
  -ld LOCAL_DIR, --local_dir LOCAL_DIR
                        Local dir path

usage: vec2rec train [-h] [-s VECTOR_SIZE] [-m MIN_CNT] [-e EPOCHS]
                     [-tr TEST_RATIO] [-t {resume,job,train,all}] [-l]
                     [-ld LOCAL_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -s VECTOR_SIZE, --vector_size VECTOR_SIZE
                        Vector size for doc2vec
  -m MIN_CNT, --min_cnt MIN_CNT
                        Min word occurrence included
  -e EPOCHS, --epochs EPOCHS
                        # epochs to models
  -tr TEST_RATIO, --test_ratio TEST_RATIO
                        percentage of sample used for test case
  -t {resume,job,train,all}, --type {resume,job,train,all}
                        Doctype for action
  -l, --local           Store a copy in local dir
  -ld LOCAL_DIR, --local_dir LOCAL_DIR
                        Local dir path

usage: vec2rec test [-h] [-s SAMPLE] [-n TOP_N] [-t {resume,job,train,all}]

optional arguments:
  -h, --help            show this help message and exit
  -s SAMPLE, --sample SAMPLE
                        Sample size from test_corpus used
  -n TOP_N, --top_n TOP_N
                        Top N similar docs returned
  -t {resume,job,train,all}, --type {resume,job,train,all}
                        Doctype for action

usage: vec2rec add_doc [-h] [-f FILENAME] [-t {resume,job,train}]

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        Name of file to be added in repo
  -t {resume,job,train}, --type {resume,job,train}
                        Doctype for action

usage: vec2rec del_doc [-h] [-f FILENAME] [-t {resume,job,train}]

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        Name of file to be deleted from repo
  -t {resume,job,train}, --type {resume,job,train}
                        Doctype for action
```

Finding Jobs from Text Entry

![Finding Job from Text Entry](/vec2rec/images/job_from_text.png)

Finding People from Text Entry

![Finding People from Text Entry](/vec2rec/images/resume_from_text.png)

![Finding People from Text Entry](/vec2rec/images/resume_from_text2.png)

Finding Training from Text Entry

![Finding Training from Text Entry](/vec2rec/images/train_from_text.png)

Finding Jobs from Resume

![Finding Jobs from Resume](/vec2rec/images/job_from_doc.png)

What-if scenario from a Resume and a Training

![What-if scenario from Resume + Training](/vec2rec/images/job_from_multi_doc.png)

## Package Structure
##### vec2rec.preprocess.tools
* class TokenData - Preprocess raw data and store
* class Tokenizer - Descriptor of a tokenizer for data cleaning and tokenization
* class PDFReader - Descriptor of a PDFReader
```python
class TokenData: # only highlights are shown here
    def __init__(self, chunksize=20):
        ... # set Dask chunksize
    # descriptors w/ dask.delayed
    extract_pdf_text = staticmethod(dask.delayed(PDFReader()))
    tokenize = staticmethod(dask.delayed(Tokenizer())) 

    def pdf_to_df(self, parent_dir, file_glob="*.pdf", df_type="resume"):
        ... # with Dask, convert pdf to df, add to class var train/test df 
    def xls_to_df(self, parent_dir, file_glob="*.xlsx", df_type="train"):
        ... # with Dask, convert xls to df, add to class variable train/test df
    def read_parquet(self, parent_dir, file_path=default_fp, df_type="all"):
        ... # read parquet from local/S3 to df and add to class var train/test df
    def to_parquet(self, parent_dir, file_path=default_fp, df_type="all"):
        ... # write class variable train/test df to parquet on local/S3

# class PDFReader is a PDF2Text descriptor
class PDFReader: # only highlights are shown here
    @staticmethod
    def extract_pdf_text(path, fmt="string"):
        ...

# class Tokenizer is a descriptor that perform tokenizing
# stemming, and various data cleaning tasks from the doc
class Tokenizer: # only highlights are shown here
    @staticmethod
    def tokenize(text):
        ...         

```
##### vec2rec.models.nlpmodels
* Descriptor classes to store model specific data from preprocessed data, and the Gensim Doc2Vec model itself
* NLPModel can be inherited and implemented with other models if available
```python
class NLPModel:
    ... # Abstract class with all functions implemented in D2VModel

class D2VModel(NLPModel):
    def __init__(self, vector_size=75, min_count=2, epochs=40, test_ratio=0.3):
        ... # Initialize model in class var
    def build_corpus(self, parent_dir, file_path, test_ratio=1 / 3):
        ... # reads in a parquet file from local/S3 and build corpus
    def train(self, epochs=None):
        ... # builds vocab and train model in class var
    def test(self, sample=1, top_n=2):
        ... # calculate accuracy with training data – doc itself should have
        # highest similarity. The print out the topn similarity with the testing data
    def load_model(self, parent_dir, file_path):
        ... # load saved model from local/S3 & store in class var
    def save_model(self, parent_dir, filepath):
        ... # save model into file and upload to local/S3
    def lookup(self, text=None, filepath=None, top_n=3):
        ... # lookup with text or filepath local or S3. Filepath can be a list
        # top_n returns to top N similar records from the repo
```
##### vec2rec.frontend.vec2rec
* class Vec2Rec - main class for the CLI or UI - models used can be replaced
```python
class Vec2Rec: # the class used by the front end
    # each of these models has a lookup() function which will download the required
    # model from S3 to perform the similarity check if not already downloaded
    # These are current Gensim models but can be others, see class NLPModel
    job_model = D2VModel()
    res_model = D2VModel()
    train_model = D2VModel()

    def add_doc(self, parent_dir, file_glob):
        ... # upload doc to S3 repository

    def del_doc(self, parent_dir, file_glob):
        ... # delete doc from S3 repository 
```
##### vec2rec.kfp.vec2rec_pipeline
Functions in this file is used to generate the definition file in yaml.
Each step returns a dsl.ContainerOp object which will ultimately be a runnable
docker container in the pipeline.

The decorator @dsl.pipeline is used to define the pipeline where results are
passed to the next phase. Care must be taken to remember these are not real
python functions but dockers. All variables passed between the phases needs
to be small (< 256k), or be serialized into files. It could either be remote
storage like S3, or passing locally. Using the options file_outputs and
artifact_argument_path, Kubeflow will help you to copy the files from one
container to another.

Each phases will need to be built as a docker and uploaded into docker hub
or GCR and downloaded as the pipeline runs.

This kind of structure forces you to restructure your code and can be very
cumbersome as all variables are not shared.

Python functions can also be directly converted into container phases without
uploading containers, tho these are still functions which does not share variables
with other steps.

```python
import kfp
from kfp import dsl

def preprocess_op(parent_dir=S3_BUCKET_BASE, chunk=20, doc_type="all", local_dir=LOCAL_BASE):
    ...
    return dsl.ContainerOp(
        name="preprocess",
        image=f"crispinnosidam/vec2rec:{img_ver}",
        command=["pipenv", "run", "python", "-W", "ignore", "-m", "vec2rec",],
        # ...
        file_outputs=[...], # file path to be passed to next step
    )

def train_op( resume_path, job_path, train_path, parent_dir=S3_BUCKET_BASE,
    vector_size=75, min_cnt=2, epochs=100, test_ratio=1 / 3, doc_type="all", local_dir=LOCAL_BASE, ):
    ...
    return dsl.ContainerOp(
        name="train",
        image=f"crispinnosidam/vec2rec:{img_ver}",
        artifact_argument_paths=[
            dsl.InputArgumentPath(argument=resume_path),
            dsl.InputArgumentPath(argument=job_path),
            dsl.InputArgumentPath(argument=train_path),
        ], # file paths to this step
        command=["pipenv", "run", "python", "-W", "ignore", "-m", "vec2rec",]
        + ["-p", parent_dir, "train", ],
        # ... other params
        file_outputs=[...], # file paths to be passed to next step
    )

def test_op(input_path, parent_dir=LOCAL_BASE, sample=1, top_n=2, doc_type="all"):
    return dsl.ContainerOp(
        name="test",
        image=f"crispinnosidam/vec2rec:{img_ver}",
        artifact_argument_paths=[...], # file paths to this step
        command=["pipenv", "run", "python", "-W", "ignore", "-m", "vec2rec",]
        + ["-p", parent_dir, "test", ]
        # ... other params
    )

@dsl.pipeline(name="my testing pipeline", description="my testing pipeline description")
def vec2rec_pipeline(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY):
    ...
    pp_op = preprocess_op(S3_BUCKET_BASE)
    tr_op = train_op(
        pp_op.outputs["resume"], pp_op.outputs["job"], pp_op.outputs["train"]
    )
    tt_op = test_op(tr_op.outputs)
    ...

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(vec2rec_pipeline, __file__ + ".yaml") # compile defn to yaml
```
# Future Enhancements
* Weighted importance based on
  * record age – with decaying importance
  * user preferences based on past search and selections
  * add user profile for matching, if available
* Migration of S3 to database for better retrieval performance
* Visualization with Roadmaps with Resume being a starting point and jobs being goals (can be multiple), with each training being steps in the middle
* Serialization of Gensim Training / Testing Corpus with pickle to reduce repeated computation after adding samples.
* Another training module which allows incremental sample addition without total retrain.
* Support conversion from MS Word for resume.
* Front end replaced by a chatbot

# References
* [Doc2Vec](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html)
* [Kubeflow](https://www.kubeflow.org/)
* [MiniKF](https://www.kubeflow.org/docs/started/workstation/getting-started-minikf/)
* [Dask DataFrame](https://docs.dask.org/en/latest/dataframe.html)
* [Dask Delayed](https://docs.dask.org/en/latest/delayed.html)
* [Pandas](https://pandas.pydata.org/)
* [NLTK](https://www.nltk.org/)
* [Krovetz Stemmer](https://sourceforge.net/p/lemur/wiki/KrovetzStemmer/)
* [PyPDF4](https://github.com/claird/PyPDF4)
* [AWS S3](https://aws.amazon.com/s3/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
* [Training Data - Hotchkiss](https://www.hotchkiss.org/uploaded/documents/Academics/2017-18Hotchkiss_CourseCatalog.pdf)
* [Job Data - Nevada Pay Survey](https://www.nevadaemployers.org/wp-content/uploads/Updated-Job-Descriptions-2018.pdf)
* [Resume Data](https://github.com/JAIJANYANI/Automated-Resume-Screening-System)

