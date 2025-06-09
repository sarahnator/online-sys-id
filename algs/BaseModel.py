import torch
from torch.utils.data import IterableDataset

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def predict_number_params(*args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method must be implemented by subclasses.")
    
    def train(self, IterableDataset):
        """  
        for grad steps:
        1. Get data from IterableDataset
        2. Forward pass through the model
        3. Compute loss using loss_fn
        4. Backward pass to compute gradients
        5. Update model parameters using an optimizer
        6. zero gradients

        Parameters
        ----------
        IterableDataset : torch.utils.data.IterableDataset
            An iterable dataset that provides data for training the model.
        Returns
        -------
        None
        """

    def loss_fn(self, *args, **kwargs):
        raise NotImplementedError("Loss function must be implemented by subclasses.")

    def datastream_predict(self, *args, **kwargs):
        raise NotImplementedError("Data stream prediction method must be implemented by subclasses.")