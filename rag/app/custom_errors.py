class ApplicationError(Exception):
    """Base class for application-specific exceptions."""
    pass

class ModelInitializationError(ApplicationError):
    """Raised when an ML model fails to load or initialize."""
    def __init__(self, model_name, message="Failed to initialize model"):
        self.model_name = model_name
        self.message = f"{message}: {model_name}"
        super().__init__(self.message)


class RepositoryError(Exception):
    """Base exception for all repository-related errors."""
    pass

class DataValidationError(RepositoryError):
    """Raised when data fails Pydantic schema validation."""
    pass

class IntegrityError(RepositoryError):
    """Raised when a unique constraint (like id) is violated."""
    pass

class PersistenceError(RepositoryError):
    """Raised when file I/O or locking fails."""
    pass