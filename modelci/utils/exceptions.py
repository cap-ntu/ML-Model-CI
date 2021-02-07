class ModelCIError(Exception):
    """Super class of all ModelCI exception types."""
    pass


class ModelStructureError(ValueError):
    """
    Exception raised when model structure unable to construct.
    """
    pass
