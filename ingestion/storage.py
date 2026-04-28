from __future__ import annotations
import os
import uuid
import logging
import mimetypes
from typing import BinaryIO

import boto3
from botocore.config import Config

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class Storage:

    def __init__(self):
        self.backend = settings.storage_backend

        if self.backend=="s3":
            self.s3 = boto3.client(
                "s3",
                region_name=settings.s3_region,
                config=Config(
                    retries={"max_attempts": 3, "mode": "standard"},
                    connect_timeout=5,
                    read_timeout=60,
                ),
            )
        os.makedirs("data/uploads", exist_ok=True)
        os.makedirs("data/parsed", exist_ok=True)

    def save(self, file: BinaryIO, filename: str | None = None) -> str:

        filename = filename or getattr(file, "name", "file")
        safe_name = os.path.basename(filename).replace(" ", "_")
        content_type, _ = mimetypes.guess_type(safe_name)
        content_type = content_type or "application/octet-stream"
        
        if self.backend == "local":
            path = os.path.join("data/uploads", safe_name)

            try:
                file.seek(0)
                with open(path, "wb") as f:
                    for chunk in iter(lambda: file.read(1024 * 1024), b""):
                        f.write(chunk)

                logger.info(f"[LOCAL] File saved → {path}")
                return path

            except Exception as e:
                logger.exception("Local file save failed")
                raise RuntimeError(f"Local save failed: {e}") from e

        elif self.backend == "s3":
            key = f"{settings.s3_prefix}/uploads/{uuid.uuid4()}_{safe_name}"

            try:
                file.seek(0)

                self.s3.upload_fileobj(
                    file,
                    settings.s3_bucket_name,
                    key,
                    ExtraArgs={
                        "ContentType": content_type
                    },
                )

                logger.info(f"[S3] Uploaded → {key}")
                return key

            except Exception as e:
                logger.exception("S3 upload failed")
                raise RuntimeError(f"S3 upload failed: {e}") from e

        else:
            raise ValueError(f"Unsupported storage backend: {self.backend}")
    
    def download_file(self, key: str, local_path: str) -> str:

        if self.backend == "local":
            return key

        elif self.backend == "s3":
            try:
                self.s3.download_file(
                    settings.s3_bucket_name,
                    key,
                    local_path,
                )

                logger.info(f"[S3] Downloaded → {local_path}")
                return local_path

            except Exception as e:
                logger.exception("S3 download failed")
                raise RuntimeError(f"S3 download failed: {e}") from e

        else:
            raise ValueError(f"Unsupported storage backend: {self.backend}")
        
    def upload_json(self, local_path: str, key: str) -> str:
        """
        Upload parsed JSON to S3 (if enabled).
        """
        if self.backend == "local":
            return local_path

        elif self.backend == "s3":
            try:
                with open(local_path, "rb") as f:
                    self.s3.upload_fileobj(
                        f,
                        settings.s3_bucket_name,
                        key,
                        ExtraArgs={
                            "ContentType": "application/json"
                        },
                    )

                logger.info(f"[S3] JSON uploaded → {key}")
                return key

            except Exception as e:
                logger.exception("JSON upload failed")
                raise RuntimeError(f"JSON upload failed: {e}") from e
            
    def delete_file(self, key: str) -> None:
        if self.backend == "s3":
            try:
                self.s3.delete_object(
                    Bucket=settings.s3_bucket_name,
                    Key=key,
                )
                logger.info(f"[S3] Deleted → {key}")
            except Exception:
                logger.warning(f"Failed to delete {key}")

