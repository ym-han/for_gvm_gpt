# Notes re process: jaxlib needs to be 0.1.67.
# when using a different version of jaxlib, error when running CausalTransformer: RuntimeError: Invalid argument: Argument does not match host shape or layout of computation parameter 0: want s32[]{:T(256)}, got s32[]

import json
import pathlib
from functools import partial

import pickle

from google.cloud import storage


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

    end_idx = response.find("]")
    return response[: end_idx]

def rm_white_space(strg: str) -> str:
    lines_without_trailing = strg.strip().split("\n")
    no_whitespace = "\n".join( [ln.strip() for ln in lines_without_trailing] )

    return no_whitespace


# Logging related
import logging
from notifiers import get_notifier
from notifiers.logging import NotificationHandler

## for Telegram logging
config_tg_path = pathlib.Path("configs/telegram.json")
if not config_tg_path.is_file():  
    download_blob(bucket, "misc/telegram.json", config_tg_path)
tg_params = json.load(open(config_tg_path))

c_hdlr = logging.StreamHandler()
f_hdlr = logging.FileHandler('debug.log')
tg_hdlr = NotificationHandler('telegram', defaults=tg_params)

c_hdlr.setLevel(logging.DEBUG)
f_hdlr.setLevel(logging.WARNING)
tg_hdlr.setLevel(logging.ERROR)

c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_hdlr.setFormatter(c_format)
tg_hdlr.setFormatter(c_format)
f_hdlr.setFormatter(f_format)

logger = logging.getLogger("loggr")
for hdlr in (c_hdlr, f_hdlr, tg_hdlr): logger.addHandler(hdlr)

tg = get_notifier('telegram')
def tg_notify(msg): tg.notify(message=msg, token=tg_params["token"], chat_id=tg_params["chat_id"])
