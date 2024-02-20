import os

from app.storage.ConfigStore import ConfigS3Storage, ConfigFileSystemStorage, ConfigStore
from app.storage.OptimizerStore import OptimizerStore, FileSystemOptimizerStore, S3OptimizerStorage


class StorageFactory:
    _config_storage = None
    _optimizer_storage = None

    @staticmethod
    def get_config_storage() -> ConfigStore:
        if StorageFactory._config_storage is None:
            storage_type = os.getenv("STORAGE_TYPE", "filesystem")
            if storage_type == "s3":
                StorageFactory._config_storage = ConfigS3Storage()
            else:
                StorageFactory._config_storage = ConfigFileSystemStorage()
        return StorageFactory._config_storage

    @staticmethod
    def get_optimizer_storage() -> OptimizerStore:
        if StorageFactory._optimizer_storage is None:
            storage_type = os.getenv("STORAGE_TYPE", "filesystem")
            if storage_type == "s3":
                StorageFactory._optimizer_storage = S3OptimizerStorage()
            else:
                StorageFactory._optimizer_storage = FileSystemOptimizerStore()
        return StorageFactory._optimizer_storage
