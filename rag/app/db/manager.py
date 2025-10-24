# JsonFileManager.py

import portalocker
import json
import os
import logging
from typing import Dict, Any
from datetime import datetime 
import tempfile
import shutil
import uuid
from app.custom_errors import PersistenceError 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def json_serial_handler(obj):
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class JsonFileManager:
    """Handles file I/O and concurrent access control via portalocker, implementing atomic writes."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        
        dir_name = os.path.dirname(self.file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"Created directory: {dir_name}")

        if not os.path.exists(self.file_path):
            try:
                # Initialize with empty JSON object if file doesn't exist
                with open(self.file_path, 'w') as f:
                    f.write('{}')
                logger.info(f"Created new empty DB file at: {self.file_path}")
            except Exception as e:
                logger.error(f"Failed to create new DB file at {self.file_path}: {e}")
                raise PersistenceError(f"Initialization failed: Could not create file. {e}")
        
        logger.info(f"Initialized JsonFileManager for file: {file_path}")

    def load_db(self) -> Dict[str, Any]:
        """Reads the entire DB file using a shared lock."""
        try:
            with portalocker.Lock(self.file_path, 'r', flags=portalocker.LOCK_SH, timeout=5) as f:
                content = f.read()
                if not content.strip(): 
                    return {}
                return json.loads(content)
        except portalocker.LockException as e:
            logger.error(f"Failed to acquire read lock on DB file: {e}")
            raise PersistenceError(f"Concurrency error during read: {e}")
        except json.JSONDecodeError:
            logger.error("JSON file content is corrupted or invalid.")
            return {} 
        except Exception as e:
            logger.error(f"Unexpected error during DB load: {e}")
            raise PersistenceError(f"Unexpected file error: {e}")

    def save_db(self, data: Dict[str, Any]):
        """
        Writes the entire DB structure atomically (Write-Atomic Swap).
        If any failure occurs during writing to the temp file, the original file is preserved.
        """
        temp_file_path = self.file_path + ".tmp." + str(uuid.uuid4())
        
        try:
            # 1. Write to the temporary file with exclusive lock
            with portalocker.Lock(temp_file_path, 'w', flags=portalocker.LOCK_EX, timeout=10) as f:
                json.dump(data, f, indent=2, default=json_serial_handler) 
                
                # Force data to be written to disk before swap
                f.flush()
                os.fsync(f.fileno())
            
            # 2. Atomically rename the temporary file to overwrite the original
            shutil.move(temp_file_path, self.file_path)
            
            logger.info("Successfully performed atomic write to DB file.")
            
        except portalocker.LockException as e:
            logger.error(f"Failed to acquire exclusive write lock on temporary file: {e}")
            raise PersistenceError(f"Concurrency error during write: {e}")
        except Exception as e:
            logger.error(f"Error during atomic DB save/swap: {e}")
            # Clean up the temporary file if the process failed before the swap
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.warning(f"Cleaned up failed temporary file: {temp_file_path}")
            raise PersistenceError(f"Atomic file write failed: {e}")