import panel as pn
import param
from abc import abstractmethod
import os
import io
import pandas as pd
from dotenv import load_dotenv

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


