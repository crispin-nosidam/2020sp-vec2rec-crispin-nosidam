import boto3
from urllib.parse import urlparse
import posixpath


def s3_glob(url):
    """
    Return the latest file name in an S3 bucket folder.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (folder  name).
    """
    up = urlparse(url)
    bucket_name = up.netloc
    print(bucket_name)
    prefix, suffix = posixpath.split(up.path[1:])
    suffix = posixpath.splitext(suffix)[1]
    print(prefix)
    print(suffix)
    s3_client = boto3.client('s3')
    objs = s3_client.list_objects_v2(Bucket=bucket_name)['Contents']
    return ["s3://"+bucket_name+"/"+obj["Key"] for obj in objs if obj["Key"].startswith(prefix) and obj["Key"].endswith(suffix)]
    """
    shortlisted_files = dict()
    for obj in objs:
        key = obj['Key']
        timestamp = obj['LastModified']
        # if key starts with folder name retrieve that key
        if key.startswith(prefix):
            # Adding a new key value pair
            shortlisted_files.update( {key : timestamp} )
    return shortlisted_files
    """

print(s3_glob("s3://csci-e29-kwc271/project/resume/*.pdf"))