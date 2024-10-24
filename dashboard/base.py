import param
import panel as pn
from abc import abstractmethod
from .base_component import BaseComponent

class DataSourceComponent(BaseComponent):
    """Abstract base class for data source components."""
    
    @abstractmethod
    def get_configuration(self):
        """Return the data source configuration."""
        pass
    
    @abstractmethod
    def validate_configuration(self):
        """Validate the current configuration."""
        pass