# import torch
# from torch.utils.data import IterableDataset

# class BaseModel(torch.nn.Module):
#     def __init__(self):
#         super(BaseModel, self).__init__()

#     @staticmethod
#     def predict_number_params(*args, **kwargs):
#         raise NotImplementedError()

#     def forward(self, *args, **kwargs):
#         raise NotImplementedError("Forward method must be implemented by subclasses.")
    
#     def train(self, IterableDataset):
#         """  
#         for grad steps:
#         1. Get data from IterableDataset
#         2. Forward pass through the model
#         3. Compute loss using loss_function
#         4. Backward pass to compute gradients
#         5. Update model parameters using an optimizer
#         6. zero gradients

#         Parameters
#         ----------
#         IterableDataset : torch.utils.data.IterableDataset
#             An iterable dataset that provides data for training the model.
#         Returns
#         -------
#         None
#         """

#     def loss_function(self, *args, **kwargs):
#         raise NotImplementedError("Loss function must be implemented by subclasses.")

#     def datastream_predict(self, *args, **kwargs):
#         raise NotImplementedError("Data stream prediction method must be implemented by subclasses.")

import torch
from abc import ABC, abstractmethod
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import Optimizer
from typing import Any, Callable, Optional
import tqdm
from util.training import train_step, test_eval

class BaseModel(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def predict_number_params(cls, *args, **kwargs) -> int:
        """
        Estimate how many parameters this architecture will have, given some hyperparameters.
        """
        raise NotImplementedError()

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @abstractmethod
    def forward(self, *inputs: Any) -> torch.Tensor:
        """Core forward pass."""
        ...

    def fit(self,
            dataset: IterableDataset,
            optimizer: Optimizer,
            loss_function: Optional[Callable]=None,
            epochs: int = 1,
            batch_size: int = 32,
            device: str = "cpu") -> None:
        """
        A basic training loop.
        """
        dataloader = DataLoader(dataset, batch_size=batch_size)
        dataloader_iter = iter(dataloader)

        self.train()
        device = torch.device(device)
        with tqdm.trange(epochs) as tqdm_bar:
            for epoch in tqdm_bar:
                batch = next(dataloader_iter)
                train_loss = train_step(
                    model=self,
                    batch=batch,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    device=device
                )
                test_loss = test_eval(
                    model=self,
                    dataset=dataset,
                    loss_function=loss_function,
                    device=device
                )
                tqdm_bar.set_postfix_str(f"loss: {train_loss:.2e}, test_loss: {test_loss:.2e}")

    def datastream_predict(self, *args, **kwargs):
        raise NotImplementedError("Data stream prediction method must be implemented by subclasses.")

    # def predict(self,
    #             dataset: IterableDataset,
    #             batch_size: int = 32,
    #             device: str = "cpu",
    #             collate_fn: Optional[Callable]=None) -> torch.Tensor:
    #     """
    #     Runs the model in eval mode over an iterable dataset
    #     and concatenates all outputs.
    #     """
    #     self.eval()
    #     loader = DataLoader(dataset,
    #                         batch_size=batch_size,
    #                         collate_fn=collate_fn)
    #     all_preds = []
    #     with torch.no_grad():
    #         for batch in loader:
    #             inputs = batch.to(device)
    #             all_preds.append(self(inputs))
    #     return torch.cat(all_preds, dim=0)

