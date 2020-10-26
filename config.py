""" Configuration class """

from abc import ABC, abstractmethod

class Configuration(ABC):
    """ This abstract base class is the gateway to the LISA set of configuration functions
    """
    def __init__(self, config):
        self.config_type = type(config)
        self.config = config

    @abstractmethod
    def get_parameter(self, parameter_name):
        """ Returns a parameter based on a parameter name """
        pass

    @classmethod
    def type(cls, config):
        """ Docstring..."""
        if isinstance(config, dict):
            return DictionaryClass(config)
        raise ValueError("Invalid configuration format.")

class DictionaryClass(Configuration):
    """ Docstring..."""
#    def __init__(self, config):
#        super().__init__(config)

    def get_parameter(self, parameter_name):
        return self.config[parameter_name]
