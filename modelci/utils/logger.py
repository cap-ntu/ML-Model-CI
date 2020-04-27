import logging


class Logger:

    DEFAULT_SEVERITY_LEVELS = {
        "StreamHandler": 'INFO',
    }

    DEFAULT_FORMATTER = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __init__(self, name=None, filename=None, severity_levels=None, formatter=None, welcome=True):
        """

        :param name: optional name of logger
        :param filename: optional filename
        :param severity_levels: optional dictionary that describes severity levels for each handler, for example:
            {
                "StreamHandler": "INFO",
                "FileHandler": "DEBUG",
            }
        :param formatter: str formatter for output
        :param welcome: bool Welcome message to logged when the logger is created. Default to `True`
        """

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        if severity_levels:
            self.severity_levels = severity_levels
        else:
            self.severity_levels = self.DEFAULT_SEVERITY_LEVELS

        if formatter:
            assert isinstance(formatter, str)
            formatter = logging.Formatter(formatter)
        else:
            formatter = logging.Formatter(self.DEFAULT_FORMATTER)

        for handler_name in self.severity_levels:
            if handler_name == "FileHandler":
                if not filename:
                    raise ValueError("filename not provided with FileHandler set")

                handler = getattr(logging, handler_name)(filename)
            else:
                handler = getattr(logging, handler_name)()

            handler.setLevel(getattr(logging, self.severity_levels[handler_name]))
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        if welcome:
            self.logger.info("Logger %s created" % name)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def critical(self, message):
        self.logger.critical(message)
