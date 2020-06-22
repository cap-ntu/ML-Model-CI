class ServiceException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class DoesNotExistException(ServiceException):
    def __init__(self, message):
        super().__init__(message=message)


class BadRequestValueException(ServiceException):
    def __init__(self, message):
        super().__init__(message=message)
