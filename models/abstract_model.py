import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod

class Abstract_Model(nn.Module):
    
    def __init__(self, args):
        super(Abstract_Model, self).__init__()
        self.args = args
        if self.init_weights:
            self._initialize_weights()
    
    @abstractmethod
    def forward(self, x, batch=None):
        pass

    @abstractmethod
    def _initialize_weights(self):
        pass

    @property
    @abstractmethod
    def init_weights(self):
        return False 