import kfp
import os
from kfp import dsl
from kubernetes.client.models import V1EnvVar
from ..frontend.vec2rec import S3_BUCKET_BASE
from ..cli import LOCAL_BASE

img_ver = "1.0"


def preprocess_op(
    parent_dir=S3_BUCKET_BASE, chunk=20, doc_type="all", local_dir=LOCAL_BASE
):
    # if not parent_dir[0].startswith("s3://"):
    #     parent_dir = os.path.dirname(parent_dir[0])
    file_outputs = (
        {
            "resume": f"{LOCAL_BASE}/parquet/resume.parquet",
            "job": f"{LOCAL_BASE}/parquet/job.parquet",
            "train": f"{LOCAL_BASE}/paruqet/train.parquet",
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
    parent_dir=S3_BUCKET_BASE,
    vector_size=75,
    min_cnt=2,
    epochs=100,
    test_ratio=1 / 3,
    doc_type="all",
    local_dir=LOCAL_BASE,
):
    """
    if type(parent_dir == dsl.InputArgumentPath):
        if not parent_dir[0].startswith("s3://"):
            parent_dir = os.path.dirname(parent_dir[0])
    else:
        if not parent_dir.startswith("s3://"):
            parent_dir = os.path.dirname(parent_dir[0])
    """
    return dsl.ContainerOp(
        name="train",
        image=f"crispinnosidam/vec2rec:{img_ver}",
        artifact_argument_paths=[
            dsl.InputArgumentPath(argument=resume_path),
            dsl.InputArgumentPath(argument=job_path),
            dsl.InputArgumentPath(argument=train_path),
        ],
        command=["pipenv", "run", "python", "-W", "ignore", "-m", "vec2rec",]
        + ["-p", parent_dir, "train",]
        + ["-s", vector_size,]
        + ["-m", min_cnt,]
        + ["-e", epochs,]
        + ["-tr", test_ratio,]
        + ["-t", doc_type,]
        + ["-l", "-ld", local_dir,],
    )


# command=["pipenv", "run", "python", "/app/read_file.py", dsl.InputArgumentPath(myfile)],


@dsl.pipeline(name="my testing pipeline", description="my testing pipeline description")
def vec2rec_pipeline(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY):
    env_var1 = V1EnvVar(name='AWS_ACCESS_KEY_ID', value=AWS_ACCESS_KEY_ID)
    env_var2 = V1EnvVar(name='AWS_SECRET_ACCESS_KEY', value=AWS_SECRET_ACCESS_KEY)
    pp_op = preprocess_op(S3_BUCKET_BASE).add_env_variable(env_var1).add_env_variable(env_var2)
    tr_op = train_op(
        pp_op.outputs["resume"], pp_op.outputs["job"], pp_op.outputs["train"]
    ).add_env_variable(env_var1).add_env_variable(env_var2)
    # pp_op.container.set_image_pull_policy("Always")
    # tr_op.container.set_image_pull_policy("Always")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(vec2rec_pipeline, __file__ + ".yaml")
