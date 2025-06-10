from datasets.van_der_pol import VanDerPolDataset

def get_dataset(name: str, **kwargs):
    """
    Returns a dataset instance based on the provided name.
    
    Args:
        name (str): The name of the dataset to retrieve.
        **kwargs: Additional keyword arguments to pass to the dataset constructor.
    
    Returns:
        An instance of the specified dataset.
    
    Raises:
        ValueError: If the dataset name is not recognized.
    """
    if name == "VanDerPol":
        return VanDerPolDataset(**kwargs)
    else:
        raise ValueError(f"Dataset '{name}' is not recognized.")