import numpy as np
from typing import Tuple


class DatasetPreprocessor:
    """Classe per normalizzare e splittare dataset."""
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
    
    def _prepare(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Flatten, normalizza e ritorna atleast_2d/1d."""
        X = X.reshape(X.shape[0], -1).astype(np.float32)
        if self.normalize:
            X = X / 255.0
        return np.atleast_2d(X), np.atleast_1d(y)
    
    def prepare_train_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splitta in train e test.
        
        Returns:
            (X_train, y_train, X_test, y_test)
        """
        X, y = self._prepare(X, y)
        split = int(len(X) * train_ratio)
        
        return X[:split], y[:split], X[split:], y[split:]
    
    def prepare_train_val_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splitta in train, validation e test.
        
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        X, y = self._prepare(X, y)
        train_end = int(len(X) * train_ratio)
        val_end = int(len(X) * (train_ratio + val_ratio))
        
        return (
            X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:]
        )
    
    def prepare_from_splits(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        val_ratio: float = None
    ) -> Tuple:
        """
        Prepara dati già splittati (es. MNIST).
        
        Args:
            val_ratio: Se specificato, splitta training in train/val.
            
        Returns:
            Se val_ratio è None: (X_train, y_train, X_test, y_test)
            Altrimenti: (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        X_train, y_train = self._prepare(X_train, y_train)
        X_test, y_test = self._prepare(X_test, y_test)
        
        if val_ratio is not None:
            split = int(len(X_train) * (1 - val_ratio))
            return (
                X_train[:split], y_train[:split],
                X_train[split:], y_train[split:],
                X_test, y_test
            )
        
        return X_train, y_train, X_test, y_test
