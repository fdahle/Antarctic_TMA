class SfMError(Exception):
    """
    Base class for all Structure-from-Motion related errors.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"{self.message}"
