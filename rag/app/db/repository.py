import os
import logging
from pydantic import BaseModel, ValidationError
from typing import Type, Dict, Any, List, Optional
from app.custom_errors import *
from app.db.manager import JsonFileManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JsonRepository:
    """Manages CRUD for a single collection, enforcing Pydantic schema."""
    def __init__(self, manager: JsonFileManager, collection_name: str, schema: Type[BaseModel]):
        self.manager = manager
        self.collection_name = collection_name
        self.schema = schema
        logger.info(f"Repository initialized for collection: '{collection_name}'")

    def create(self, item_data: Dict[str, Any]) -> BaseModel:
        """Creates a new item, validating and enforcing uniqueness."""
        try:
            new_item = self.schema(**item_data)
        except ValidationError as e:
            logger.warning(f"Validation failed for new item in '{self.collection_name}'.")
            raise DataValidationError(f"Validation failed: {e}")
            
        db = self.manager.load_db()
        collection: List[Dict[str, Any]] = db.get(self.collection_name, [])

        # CRITICAL FIX: Check for 'id' instead of '_id'
        if any(item.get('id') == new_item.id for item in collection):
            logger.warning(f"Integrity violation: Item with id {new_item.id} already exists.")
            raise IntegrityError(f"Item with id {new_item.id} already exists in '{self.collection_name}'.")

        collection.append(new_item.model_dump())
        db[self.collection_name] = collection
        self.manager.save_db(db)
        
        return new_item

    def find_all(self) -> List[BaseModel]:
        """Returns all items in the collection, validating on read."""
        db = self.manager.load_db()
        collection: List[Dict[str, Any]] = db.get(self.collection_name, [])
        
        validated_list: List[BaseModel] = []
        for item_data in collection:
             try:
                validated_list.append(self.schema(**item_data))
             except ValidationError:
                 logger.warning(f"Corrupted data found in '{self.collection_name}', skipping item.")
                 continue
        
        return validated_list

    def find_by_id(self, item_id: str) -> Optional[BaseModel]:
        """Finds a single item by its id."""
        db = self.manager.load_db()
        collection: List[Dict[str, Any]] = db.get(self.collection_name, [])
        # CRITICAL FIX: Search for 'id' instead of '_id'
        item_data = next((item for item in collection if item.get('id') == item_id), None)
        
        if item_data:
            try:
                return self.schema(**item_data)
            except ValidationError as e:
                logger.error(f"Failed to validate item {item_id} on read.")
                raise DataValidationError(f"Schema violation for item {item_id}: {e}")
        return None

    def update(self, item_id: str, updates: Dict[str, Any]) -> BaseModel:
        """Updates an existing item, validating the full updated object."""
        db = self.manager.load_db()
        collection: List[Dict[str, Any]] = db.get(self.collection_name, [])
        
        try:
            # CRITICAL FIX: Search for 'id' instead of '_id'
            index = next(i for i, item in enumerate(collection) if item.get('id') == item_id)
        except StopIteration:
            raise ValueError(f"Item with id {item_id} not found.")

        current_data = collection[index]
        updated_data = {**current_data, **updates}
        
        try:
            validated_item = self.schema(**updated_data)
        except ValidationError as e:
            logger.warning(f"Update validation failed for item {item_id}.")
            raise DataValidationError(f"Update validation failed: {e}")
            
        collection[index] = validated_item.model_dump()
        db[self.collection_name] = collection
        self.manager.save_db(db)
        
        return validated_item

    def delete(self, item_id: str) -> bool:
        """Deletes an item by its id."""
        db = self.manager.load_db()
        collection: List[Dict[str, Any]] = db.get(self.collection_name, [])
        
        original_count = len(collection)
        # CRITICAL FIX: Filter by 'id' instead of '_id'
        new_collection = [item for item in collection if item.get('id') != item_id]
        
        if len(new_collection) == original_count:
            return False
            
        db[self.collection_name] = new_collection
        self.manager.save_db(db)
        logger.info(f"Deleted item {item_id} from '{self.collection_name}'.")
        return True
