import os
from abc import ABC, abstractmethod
from typing import Optional
import boto3
from botocore.exceptions import ClientError

from src.config import settings


class BaseStorageProvider(ABC):
    @abstractmethod
    def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload raw bytes to storage and return the storage key/path."""
        pass

    @abstractmethod
    def download(self, key: str) -> bytes:
        """Retrieve raw bytes by key."""
        pass

    @abstractmethod
    def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a URL (presigned or local) to access the file."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete file from storage."""
        pass


class S3StorageProvider(BaseStorageProvider):
    """S3-compatible storage provider (works for MinIO, AWS S3, Cloudflare R2, GCP Storage)."""

    def __init__(self):
        self.bucket = settings.S3_BUCKET_NAME
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT_URL,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION,
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                self.client.create_bucket(Bucket=self.bucket)
            except Exception as e:
                print(f"[Storage] Warning: Could not create bucket {self.bucket}: {e}")

    def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type
        )
        return key

    def download(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def get_url(self, key: str, expires_in: int = 3600) -> str:
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        except Exception:
            # Fallback simple URL
            return f"{settings.S3_ENDPOINT_URL}/{self.bucket}/{key}"

    def delete(self, key: str) -> bool:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False


class LocalStorageProvider(BaseStorageProvider):
    """Local filesystem storage provider for lightweight testing without object storage."""

    def __init__(self, base_dir: str = settings.LOCAL_STORAGE_DIR):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def _resolve_path(self, key: str) -> str:
        # Sanitize key to prevent path traversal
        clean_key = os.path.normpath(key).lstrip(r"\/")
        full_path = os.path.join(self.base_dir, clean_key)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return full_path

    def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        file_path = self._resolve_path(key)
        with open(file_path, "wb") as f:
            f.write(data)
        return key

    def download(self, key: str) -> bytes:
        file_path = self._resolve_path(key)
        with open(file_path, "rb") as f:
            return f.read()

    def get_url(self, key: str, expires_in: int = 3600) -> str:
        return f"/api/documents/files/{key}"

    def delete(self, key: str) -> bool:
        file_path = self._resolve_path(key)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False


class StorageService:
    """Enterprise multi-tenant Storage Service."""

    def __init__(self):
        if settings.STORAGE_BACKEND == "s3":
            self.provider: BaseStorageProvider = S3StorageProvider()
        else:
            self.provider: BaseStorageProvider = LocalStorageProvider()

    def build_object_key(self, tenant_id: str, doc_id: str, filename: str) -> str:
        """Create a multi-tenant isolated object key."""
        clean_filename = os.path.basename(filename).replace(" ", "_")
        return f"tenants/{tenant_id}/docs/{doc_id}/{clean_filename}"

    def upload_document(self, tenant_id: str, doc_id: str, filename: str, data: bytes, content_type: str = "application/pdf") -> str:
        key = self.build_object_key(tenant_id, doc_id, filename)
        return self.provider.upload(key, data, content_type)

    def download_document(self, key: str) -> bytes:
        return self.provider.download(key)

    def get_document_url(self, key: str, expires_in: int = 3600) -> str:
        return self.provider.get_url(key, expires_in)

    def delete_document(self, key: str) -> bool:
        return self.provider.delete(key)


# Global storage service instance
storage_service = StorageService()
