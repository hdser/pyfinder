import param
from abc import ABCMeta, abstractmethod

class ParameterizedABCMeta(param.Parameterized.__class__, ABCMeta):
    """Combined metaclass for Parameterized and ABC classes."""
    pass

class BaseComponent(param.Parameterized, metaclass=ParameterizedABCMeta):
    """Base class for all dashboard components."""
    def __init__(self, **params):
        super().__init__(**params)
        self._create_components()
        self._setup_callbacks()
    
    def _create_components(self):
        """Create UI components. Override in subclasses."""
        pass
    
    def _setup_callbacks(self):
        """Set up event callbacks. Override in subclasses."""
        pass
    
    def view(self):
        """Return the component's view. Override in subclasses."""
        pass