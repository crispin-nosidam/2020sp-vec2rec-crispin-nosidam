import kfp
import os
from kfp import dsl
from kubernetes.client.models import V1EnvVar
from ..frontend.vec2rec import S3_BUCKET_BASE
from ..cli import LOCAL_BASE

# img_ver = "1.4"
img_ver = "2.2"


def preprocess_op(
    parent_dir=S3_BUCKET_BASE, chunk=20, doc_type="all", local_dir=LOCAL_BASE
):
    # if not parent_dir[0].startswith("s3://"):
    #     parent_dir = os.path.dirname(parent_dir[0])
    file_outputs = (
        {
            "resume": f"{LOCAL_BASE}/parquet/resume.parquet",
            "job": f"{LOCAL_BASE}/parquet/job.parquet",
            "train": f"{LOCAL_BASE}/parquet/train.parquet",
        }
        if doc_type == "all"
        else {doc_type: f"{LOCAL_BASE}/parquet/{doc_type}.parquet"}
    )
    return dsl.ContainerOp(
        name="preprocess",
        image=f"crispinnosidam/vec2rec:{img_ver}",
        command=["pipenv", "run", "python", "-W", "ignore", "-m", "vec2rec",]
        + ["-p", parent_dir, "preprocess",]
        + ["-c", chunk,]
        + ["-t", doc_type,]
        + ["-l", "-ld", local_dir,],
        file_outputs=file_outputs,
    )


def train_op(
    resume_path,
    job_path,
    train_path,
    # parent_dir=S3_BUCKET_BASE,
    parent_dir=LOCAL_BASE,
    vector_size=75,
    min_cnt=2,
    epochs=100,
    test_ratio=1 / 3,
    doc_type="all",
    local_dir=LOCAL_BASE,
):
    file_outputs = (
        {
            "resume_train_parquet": f"{LOCAL_BASE}/parquet/resume_train.parquet",
            "resume_test_parquet": f"{LOCAL_BASE}/parquet/resume_test.parquet",
            "resume_model": f"{LOCAL_BASE}/models/resume_model",
            "job_train_parquet": f"{LOCAL_BASE}/parquet/job_train.parquet",
            "job_test_parquet": f"{LOCAL_BASE}/parquet/job_test.parquet",
            "job_model": f"{LOCAL_BASE}/models/job_model",
            "train_train_parquet": f"{LOCAL_BASE}/parquet/train_train.parquet",
            "train_test_parquet": f"{LOCAL_BASE}/parquet/train_test.parquet",
            "train_model": f"{LOCAL_BASE}/models/train_model",
        }
        if doc_type == "all"
        else (
            {
                doc_type
                + "_train_parquet": f"{LOCAL_BASE}/parquet/{doc_type}_train.parquet",
                doc_type
                + "_test_parquet": f"{LOCAL_BASE}/parquet/{doc_type}_test.parquet",
                doc_type + "_model": f"{LOCAL_BASE}/models/{doc_type}_model",
            }
        )
    )
    input_arg_path_args = {}
    if doc_type == "resume":
        input_arg_path_args = {
            "argument": resume_path,
            "path": LOCAL_BASE + f"/parquet/resume.parquet",
        }
    if doc_type == "job":
        input_arg_path_args = {
            "argument": job_path,
            "path": LOCAL_BASE + f"/parquet/job.parquet",
        }
    if doc_type == "train":
        input_arg_path_args = {
            "argument": train_path,
            "path": LOCAL_BASE + f"/parquet/train.parquet",
        }
    return dsl.ContainerOp(
        name="train",
        image=f"crispinnosidam/vec2rec:{img_ver}",
        artifact_argument_paths=[
            dsl.InputArgumentPath(
                argument=resume_path, path=LOCAL_BASE + "/parquet/resume.parquet"
            ),
            dsl.InputArgumentPath(
                argument=job_path, path=LOCAL_BASE + "/parquet/job.parquet"
            ),
            dsl.InputArgumentPath(
                argument=train_path, path=LOCAL_BASE + "/parquet/train.parquet"
            ),
        ]
        if doc_type == "all"
        else [dsl.InputArgumentPath(**input_arg_path_args)],
        command=["pipenv", "run", "python", "-W", "ignore", "-m", "vec2rec",]
        + ["-p", parent_dir, "train",]
        + ["-s", vector_size,]
        + ["-m", min_cnt,]
        + ["-e", epochs,]
        + ["-tr", test_ratio,]
        + ["-t", doc_type,]
        + ["-l", "-ld", local_dir,],
        file_outputs=file_outputs,
    )


def test_op(input_path, parent_dir=LOCAL_BASE, sample=1, top_n=2, doc_type="all"):
    return dsl.ContainerOp(
        name="test",
        image=f"crispinnosidam/vec2rec:{img_ver}",
        artifact_argument_paths=[
            dsl.InputArgumentPath(
                argument=input_path["resume_train_parquet"],
                path=LOCAL_BASE + "/parquet/resume_train.parquet",
            ),
            dsl.InputArgumentPath(
                argument=input_path["resume_test_parquet"],
                path=LOCAL_BASE + "/parquet/resume_test.parquet",
            ),
            dsl.InputArgumentPath(
                argument=input_path["resume_model"],
                path=LOCAL_BASE + "/models/resume_model",
            ),
            dsl.InputArgumentPath(
                argument=input_path["job_train_parquet"],
                path=LOCAL_BASE + "/parquet/job_train.parquet",
            ),
            dsl.InputArgumentPath(
                argument=input_path["job_test_parquet"],
                path=LOCAL_BASE + "/parquet/job_test.parquet",
            ),
            dsl.InputArgumentPath(
                argument=input_path["job_model"], path=LOCAL_BASE + "/models/job_model"
            ),
            dsl.InputArgumentPath(
                argument=input_path["train_train_parquet"],
                path=LOCAL_BASE + "/parquet/train_train.parquet",
            ),
            dsl.InputArgumentPath(
                argument=input_path["train_test_parquet"],
                path=LOCAL_BASE + "/parquet/train_test.parquet",
            ),
            dsl.InputArgumentPath(
                argument=input_path["train_model"],
                path=LOCAL_BASE + "/models/train_model",
            ),
        ]
        if doc_type == "all"
        else [
            dsl.InputArgumentPath(
                argument=input_path[f"{doc_type}_train_parquet"],
                path=LOCAL_BASE + f"/parquet/{doc_type}_train.parquet",
            ),
            dsl.InputArgumentPath(
                argument=input_path[f"{doc_type}_test_parquet"],
                path=LOCAL_BASE + f"/parquet/{doc_type}_test.parquet",
            ),
            dsl.InputArgumentPath(
                argument=input_path[f"{doc_type}_model"],
                path=LOCAL_BASE + f"/models/{doc_type}_model",
            ),
        ],
        command=["pipenv", "run", "python", "-W", "ignore", "-m", "vec2rec",]
        + ["-p", parent_dir, "test", ]
        + ["-s", sample, ]
        + ["-n", top_n, ]
    )


@dsl.pipeline(name="vec2rec", description="Vec2Rec Model Building Pipeline")
def vec2rec_pipeline(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY):
    # def vec2rec_pipeline():
    env_var1 = V1EnvVar(name="AWS_ACCESS_KEY_ID", value=AWS_ACCESS_KEY_ID)
    env_var2 = V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value=AWS_SECRET_ACCESS_KEY)
    pp_op = (
        preprocess_op(parent_dir="/app/vec2rec/data")
        .add_env_variable(env_var1)
        .add_env_variable(env_var2)
    )
    tr_op = (
        train_op(pp_op.outputs["resume"], pp_op.outputs["job"], pp_op.outputs["train"])
        .add_env_variable(env_var1)
        .add_env_variable(env_var2)
    )
    tt_op = test_op(tr_op.outputs)
    """
    pp_op = preprocess_op(S3_BUCKET_BASE)
    tr_op = train_op(
        pp_op.outputs["resume"], pp_op.outputs["job"], pp_op.outputs["train"]
    )
    # pp_op.container.set_image_pull_policy("Always")
    # tr_op.container.set_image_pull_policy("Always")
    """


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        vec2rec_pipeline, os.path.splitext(__file__)[0] + ".yaml"
    )
