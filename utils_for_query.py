# Notes re process: jaxlib needs to be 0.1.67.
# when using a different version of jaxlib, error when running CausalTransformer: RuntimeError: Invalid argument: Argument does not match host shape or layout of computation parameter 0: want s32[]{:T(256)}, got s32[]
import os
import sys
import ujson
from pathlib import Path
from functools import partial

import pickle

from google.cloud import storage

IP_ADDR = os.environ["SSH_CONNECTION"].split()[2]

# Util funcs
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    logging.info(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


""" Extracts the name from the output. 
    This assumes that the name is enclosed in brackets """
def extract_nm_fr_resp(response:str = "") -> str:
    if "]" not in response: return "no closing ]"

    end_idx = response.find("]tpu_related/")
    return response[: end_idx]

def rm_white_space(strg: str) -> str:
    lines_without_trailing = strg.strip().split("\n")
    no_whitespace = "\n".join( [ln.strip() for ln in lines_without_trailing] )

    return no_whitespace


# Logging related
import logging
from notifiers import get_notifier

## Logging config
LOGS_DIR = Path("logs")
if not LOGS_DIR.is_dir(): LOGS_DIR.mkdir()

ERROR_LOG_PATH = Path(LOGS_DIR, f"error_{IP_ADDR}.log")
INFO_LOG_PATH = Path(LOGS_DIR, f"info_{IP_ADDR}.log")

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": INFO_LOG_PATH,
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": ERROR_LOG_PATH,
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "loggers": {
        "root": {
            "handlers": ["console", "info", "error"],
            "level": logging.INFO,
            "propagate": True,
        },
    },
}

## for non-logging Telegram notifications
tg = get_notifier('telegram')
def tg_notify(msg: str): tg.notify(message=msg, token=tg_params["token"], chat_id=tg_params["chat_id"])
