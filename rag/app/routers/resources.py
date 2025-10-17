# app/routers/resources.py

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile
from fastapi.responses import FileResponse
from typing import Annotated, List
import uuid
import logging
import os
import mimetypes
from app.custom_errors import RepositoryError, IntegrityError
from app.db.repository import JsonRepository
from app.app_context import get_resource_repository
from app.data_models import ResourceModel

router = APIRouter(
    prefix="/resources",
    tags=["Resources"],
)

logger = logging.getLogger(__name__)

ResourceRepoDep = Annotated[JsonRepository, Depends(get_resource_repository)]

STORAGE_DIR = "corpora"
os.makedirs(STORAGE_DIR, exist_ok=True)

@router.post("/upload", response_model=ResourceModel, status_code=status.HTTP_201_CREATED)
async def upload_resource(
    file: UploadFile, 
    resource_repo: ResourceRepoDep
):
    resource_id = str(uuid.uuid4())
    file_name = file.filename
    stored_file_path = os.path.join(STORAGE_DIR, f"{resource_id}_{file_name}")
    
    size_bytes = 0
    
    try:
        # Store the file and measure size
        with open(stored_file_path, "wb") as buffer:
            while chunk := await file.read(8192):
                buffer.write(chunk)
                size_bytes += len(chunk)

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(file_name)
        mime_type = mime_type if mime_type else file.content_type

        # Assemble the ResourceModel data
        resource_dict = {
            'id': resource_id,
            'file_name': file_name,
            'storage_path': stored_file_path,
            'size_bytes': size_bytes,
            'mime_type': mime_type,
        }
        
        # Create database record
        new_resource = resource_repo.create(resource_dict)
        
        logger.info("File uploaded and metadata created.", extra={'resource_id': resource_id, 'file_name': file_name})
        return new_resource
    
    except IntegrityError as e:
        if os.path.exists(stored_file_path):
            os.remove(stored_file_path)
            logger.warning(f"Integrity conflict. Deleted file: {stored_file_path}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        if os.path.exists(stored_file_path):
            os.remove(stored_file_path)
        logger.error(f"Error during upload/metadata creation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred during file processing.")


@router.get("/{resource_id}/download")
async def download_resource(resource_id: str, resource_repo: ResourceRepoDep):
    try:
        resource_data = resource_repo.find_by_id(resource_id)
        if resource_data is None:
            logger.warning("Resource not found for download.", extra={'resource_id': resource_id})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
        
        resource = ResourceModel.model_validate(resource_data)
        file_path = resource.storage_path
        
        if not os.path.exists(file_path):
             logger.error("Resource file missing on server.", extra={'resource_id': resource_id, 'path': file_path})
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Resource file is missing on the server.")

        logger.info("Resource file served.", extra={'resource_id': resource_id})
        return FileResponse(
            path=file_path, 
            filename=resource.file_name, 
            media_type=resource.mime_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during resource download: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.get("/{resource_id}", response_model=ResourceModel)
def read_resource_metadata(resource_id: str, resource_repo: ResourceRepoDep):
    try:
        resource = resource_repo.find_by_id(resource_id)
        if resource is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
        
        logger.info("Resource metadata retrieved.", extra={'resource_id': resource_id})
        return ResourceModel.model_validate(resource)
    except Exception as e:
        logger.error(f"Error retrieving resource metadata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.get("", response_model=List[ResourceModel])
def read_all_resources(resource_repo: ResourceRepoDep):
    try:
        resources = resource_repo.find_all()
        logger.info("All resources metadata retrieved.", extra={'count': len(resources)})
        return [ResourceModel.model_validate(r) for r in resources]
    except Exception as e:
        logger.error(f"Error retrieving all resources metadata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.delete("/{resource_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_resource(resource_id: str, resource_repo: ResourceRepoDep):
    try:
        # A. Retrieve metadata
        resource_data = resource_repo.find_by_id(resource_id)
        if resource_data is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Resource ID '{resource_id}' not found.")
        
        resource = ResourceModel.model_validate(resource_data)
        file_path = resource.storage_path
        
        # B. Delete the file from disk
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("File removed from storage.", extra={'resource_id': resource_id, 'path': file_path})
        else:
            logger.warning("File missing on disk during deletion.", extra={'resource_id': resource_id, 'path': file_path})
            
        # C. Delete the metadata record
        if not resource_repo.delete(resource_id):
             raise RepositoryError("Failed to delete resource metadata.")

        logger.info("Resource and metadata deleted successfully.", extra={'resource_id': resource_id})
        return
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during resource deletion: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")