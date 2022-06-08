from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch

@dataclass
class Loss(ABC):
    value: torch.Tensor = None

    @abstractmethod
    def __call__(self, x):
        pass

class MSELoss(Loss):
    def __call__(self, y, y_pred):
        self.value =  (y_pred - y).pow(2).mean()
        return self.value
    
class CrossEntropyLoss(Loss):
    def __call__(self, y, y_pred):
        self.value =  -torch.mean(y * torch.log(y_pred))
        return self.value

class BinaryCrossEntropyLoss(Loss):
    def __call__(self, y, y_pred):
        self.value =  -torch.mean((y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)))
        return self.value