import os, sys
import hashlib
import logging
from typing import Optional, List
from datetime import datetime
import uuid

from abc import ABC, abstractmethod


class Database(ABC):
    """
    Databases API helper.
    """
    def __init__(self, logs_name: str, logs_dir: Optional[str] = None):
        """
        Initializes the helper and its logging.
        """
        super().__init__()

        self._config_logger(logs_name=logs_name, logs_dir=logs_dir)

        return None


    @abstractmethod
    def connect_database(self, *args, **kwargs):
        """
        Connects to the database and creates a connector object. 
        """
        raise NotImplementedError


    @abstractmethod
    def disconnect_database(self, *args, **kwargs):
        """
        Closes the database linked to the connector ``conn``.
        """
        raise NotImplementedError


    @abstractmethod    
    def query_data(self, *args, **kwargs):
        """
        Retreives matching entries/records/documents from the DB.
        """
        raise NotImplementedError


    @abstractmethod    
    def send_data(self, *args, **kwargs):
        """
        Injects data into the DB by creating new entry/record/document.
        """
        raise NotImplementedError


    @abstractmethod
    def update_data(self, *args, **kwargs):
        """
        Updates the values of existing records matching the query.
        """
        raise NotImplementedError


    @abstractmethod    
    def delete_data(self, *args, **kwargs):
        """
        Deletes matching entries/records/documents from the DB.
        """
        raise NotImplementedError


    def _config_logger(self, 
                       logs_name: str, 
                       logs_dir: Optional[str] = None, 
                       logs_level: str = os.getenv("LOGS_LEVEL", "INFO"),
                       logs_output: list[str] = ["console", "file"]):
        """
        Will configure logging accordingly to the plateform the program is running on. This
        is the default behaviour. See ``custom_config()`` to override the parameters.
        """

        if logs_dir is None:
            logs_dir = os.path.join(os.getcwd(), "logs", str(datetime.now().strftime("%Y-%m-%d")))
        os.makedirs(logs_dir, exist_ok=True)

        self.logger = logging.getLogger(logs_name)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # If a logger already exists, this prevents duplication of the logger handlers
        if self.logger.hasHandlers():
            for handler in self.logger.handlers:
                handler.close()

        # Creates/recreates the handler(s)
        if not self.logger.hasHandlers():

            if "console" in logs_output:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging._nameToLevel[logs_level])
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
                self.logger.info("Logging handler configured for console output.")

            if "file" in logs_output:
                file_handler = logging.FileHandler(os.path.join(logs_dir, f"{datetime.now().strftime('%H-%M-%S')}-app.log"))
                file_handler.setLevel(logging._nameToLevel[logs_level])
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                self.logger.info("Logging handler configured for file output.")

        return None

    def _hash_content(self, content: str, prefixes: List[str], algo: str = "md5") -> str:
        """
        Hashes a document content into a unique id of format <prefixes>-<hashed_content>.
        Useful to automatically overwrite a stored document when a document with the same 
        timestamp and content is written into Elasticsearch or SQL.

        Parameters
        ----------
        content : str
            The text content to hash.
        prefixes : list[str]
            A list of prefixes (such as metadata, timestamps...) to prefix the hashed content with.
        algo : str, optional
            The hashing algorithm to use. Supported: "md5", "sha1", "sha256", "uuid5".
            Default is "md5".

        Returns
        -------
        hashed_id : str
            The unique hashed ID.

        Examples
        --------
        >>> print(_hash_content("Message from Caroline: Merry Christmas!", ["2024/12/25", "103010"], algo="md5"))
        '2024/12/25-103010-4432e1c6d1c4f0db2f157d501ae242a7'
        >>> print(_hash_content("Message from Caroline: Merry Christmas!", ["2024/12/25", "103010"], algo="sha256"))
        '2024/12/25-103010-84f6b29a7fa3e11e5f0b0f5d63c024c97b51a9c5f457d07d41b58738e2e0d7f4'
        >>> print(_hash_content("Message from Caroline: Merry Christmas!", ["2024/12/25", "103010"], algo="uuid5"))
        '2024/12/25-103010-4b28f4a0-6bcf-55cc-95b3-2e3d5a64f155'
        """
        base = "-".join(prefixes) + "-" + content

        if algo == "md5":
            digest = hashlib.md5(base.encode()).hexdigest()
        elif algo == "sha1":
            digest = hashlib.sha1(base.encode()).hexdigest()
        elif algo == "sha256":
            digest = hashlib.sha256(base.encode()).hexdigest()
        elif algo == "uuid5":
            digest = str(uuid.uuid5(uuid.NAMESPACE_DNS, base))
        else:
            raise ValueError(f"Unsupported algo: {algo}")

        return f"{'-'.join(prefixes)}-{digest}"
