import copy
import os
from typing import Callable
import numpy as np
import torch
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURAZIONE
# =============================================================================


def set_seed(seed: int = 16) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

class Config:
    """Configurazione centralizzata per il training."""
    
    # Dataset
    DATA_ROOT = "data"
    TRAIN_SPLIT_RATIO = 0.7  # Percentuale dati per training
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10
    PATIENCE = 5  # Early stopping patience
    
    # Architettura
    INPUT_SIZE = 28 * 28
    HIDDEN_1 = 128
    HIDDEN_2 = 64
    OUTPUT_SIZE = 10

    # Persistence
    MODEL_SAVE_PATH = "Model/pytorch/weights/fcnn_model.pth"


# =============================================================================
# DEVICE SETUP
# =============================================================================

def setup_device() -> torch.device:
    """
    Configura e restituisce il device per il training.
    
    Utilizza CUDA/ROCm se disponibile e USE_GPU=TRUE nel .env.
    
    Returns:
        torch.device: Device selezionato (cuda o cpu).
    """
    print(f"torch: {torch.__version__}")
    print(f"torch.version.hip: {torch.version.hip}")
    print(f"cuda.is_available (ROCm usa questa API): {torch.cuda.is_available()}")
    
    load_dotenv()
    use_gpu = os.getenv("USE_GPU", "FALSE").upper() == "TRUE"
    
    if torch.cuda.is_available() and use_gpu:
        print(f"device count: {torch.cuda.device_count()}")
        print(f"device 0: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    print(f"Selected device: {device}")
    return device


# =============================================================================
# DATASET
# =============================================================================

def load_datasets(data_root: str = Config.DATA_ROOT):
    """
    Carica i dataset MNIST per training e test.
    
    Args:
        data_root: Directory radice per i dati.
        
    Returns:
        tuple: (training_data, test_data)
    """
    training_data = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=ToTensor()
    )
    
    test_data = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=ToTensor()
    )
    
    return training_data, test_data


def create_train_val_split(dataset, train_ratio: float = Config.TRAIN_SPLIT_RATIO):
    """
    Divide il dataset in training e validation set.
    
    Args:
        dataset: Dataset completo.
        train_ratio: Percentuale di dati per il training.
        
    Returns:
        tuple: (train_subset, val_subset)
    """
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    
    train_ds = Subset(dataset, range(0, n_train))
    val_ds = Subset(dataset, range(n_train, n_total))
    
    return train_ds, val_ds


# =============================================================================
# MODELLO
# =============================================================================

class FCNN(nn.Module):
    """
    Fully Connected Neural Network.
    
    Args:
        layers: Sequenza di layer nn.Sequential.
    """
    
    def __init__(self, layers: nn.Sequential):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del modello."""
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


def create_model(
    input_size: int = Config.INPUT_SIZE,
    hidden_layers: list[int] = [512, 256, 128, 64],
    output_size: int = Config.OUTPUT_SIZE,
    activations: list[type[nn.Module]] | type[nn.Module] = nn.ReLU
) -> FCNN:
    """
    Crea un'istanza del modello FCNN in modo modulare.
    
    Args:
        input_size: Dimensione input.
        hidden_layers: Lista con il numero di neuroni per ogni hidden layer.
        output_size: Numero classi output.
        activations: Una singola classe di attivazione (applicata a tutti i layer) 
                     o una lista di classi (una per ogni layer nascosto).
        
    Returns:
        FCNN: Modello creato.
    """
    model_layers = []
    last_size = input_size
    
    # Se activations è una singola classe, usala per tutti i layer
    if isinstance(activations, type):
        activations = [activations] * len(hidden_layers)  #[activations] * 4 produce una lista di 4 elementi: [activations, activations, activations, activations]
    
    if len(activations) != len(hidden_layers):
        raise ValueError("Il numero di funzioni di attivazione deve coincidere con il numero di hidden layers.")

    for size, activation_class in zip(hidden_layers, activations):
        model_layers.append(nn.Linear(last_size, size))
        model_layers.append(activation_class())
        last_size = size
    
    model_layers.append(nn.Linear(last_size, output_size))
    
    # '*' è Unpacking Operator (es: nn.Sequential(*model_layers) to nn.Sequential(layer1, layer2, layer3))
    return FCNN(nn.Sequential(*model_layers))


def save_model(model: nn.Module, path: str = Config.MODEL_SAVE_PATH) -> None:
    """Salva i pesi del modello."""
    torch.save(model.state_dict(), path)
    print(f"Modello salvato in: {path}")


def load_model(model: nn.Module, path: str = Config.MODEL_SAVE_PATH) -> None:
    """Carica i pesi del modello."""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Modello caricato da: {path}")
    else:
        print(f"Errore: Il file {path} non esiste.")


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_loop(
    training_data,
    model: nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = Config.BATCH_SIZE,
    verbose: bool = True
) -> float:
    """
    Esegue un'epoca di training e restituisce la loss media dell'epoca.
    
    Args:
        training_data: Dataset di training.
        model: Modello da trainare.
        loss_fn: Funzione di loss.
        optimizer: Ottimizzatore.
        device: Device per il training.
        batch_size: Dimensione batch.
        verbose: Se True, stampa info sul primo batch.
    """
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
    model.train()

    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()

    epoch_loss = []

    for batch_idx, (X, Y) in enumerate(tqdm(train_dataloader, desc="Training")):
        X, Y = X.to(device), Y.to(device)
        
        logits = model(X)
        
        if verbose and batch_idx == 0:
            tqdm.write(f"Batch on: {X.device}, Logits on: {logits.device}")

        loss = loss_fn(logits, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss.append(loss.item())
    
    train_loss = float(np.mean(epoch_loss)) if epoch_loss else 0.0
    print(f"Train loss: {train_loss:.6f}")

    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()
    return train_loss


def test_loop(
    test_data,
    model: nn.Module,
    loss_fn: Callable,
    device: torch.device,
    batch_size: int = Config.BATCH_SIZE
) -> tuple[float, float]:
    """
    Valuta il modello sul test set.
    
    Args:
        test_data: Dataset di test.
        model: Modello da valutare.
        loss_fn: Funzione di loss.
        device: Device per l'inferenza.
        batch_size: Dimensione batch.
        
    Returns:
        tuple: (accuracy, avg_loss)
    """
    test_dataloader = DataLoader(test_data, batch_size, shuffle=False)
    model.eval()
    
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0.0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    
    print(f"Test Error:\n Accuracy: {100 * accuracy:>0.1f}%, Avg loss: {test_loss:>8f}\n")
    
    return accuracy, test_loss


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    device: torch.device
) -> float:
    """
    Calcola la loss media sul dataloader.
    
    Args:
        model: Modello da valutare.
        dataloader: DataLoader per la valutazione.
        loss_fn: Funzione di loss.
        device: Device per l'inferenza.
        
    Returns:
        float: Loss media.
    """
    model.eval()
    val_losses = []

    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        loss = loss_fn(logits, Y)
        val_losses.append(loss.cpu().item())
    
    return float(np.mean(val_losses))


# =============================================================================
# TRAINING CON EARLY STOPPING
# =============================================================================

def train(
    model: nn.Module,
    train_ds,
    val_ds,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = Config.EPOCHS,
    patience: int = Config.PATIENCE,
    batch_size: int = Config.BATCH_SIZE,
    early_stopping: bool = False
) -> dict:
    """
    Training completo con raccolta di history (train/val loss) e opzionale early stopping.
    
    Args:
        model: Modello da trainare.
        train_ds: Dataset di training.
        val_ds: Dataset di validazione.
        loss_fn: Funzione di loss.
        optimizer: Ottimizzatore.
        device: Device per il training.
        epochs: Numero massimo di epoche.
        patience: Epoche senza miglioramento prima dello stop.
        batch_size: Dimensione batch.
        
    Returns:
        dict: Dizionario con best_state e training_history.
    """
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_validation_loss = float("inf")
    best_state = None
    current_patience = patience
    history = {"train_epochs": [], "train_losses": [], "val_losses": []}
    
    for epoch in range(epochs):
        print(f"\n\nEpoch {epoch + 1}/{epochs}\n" + "-" * 40)
        
        # Training
        train_loss = train_loop(train_ds, model, loss_fn, optimizer, device, batch_size)
        # usare numerazione 0-based per le epoche
        history["train_epochs"].append(epoch)
        history["train_losses"].append(train_loss)
        
        # Validation (se disponibile)
        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader, loss_fn, device)
            print(f"Validation loss: {val_loss:.6f}")
            history["val_losses"].append(val_loss)

            # Early stopping check (solo se abilitato)
            if early_stopping:
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    current_patience = patience
                    best_state = copy.deepcopy(model.state_dict())
                    print(f"✓ New best model saved (loss: {val_loss:.6f})")
                else:
                    current_patience -= 1
                    print(f"No improvement. Patience: {current_patience}/{patience}")
                
                if current_patience == 0:
                    print("\n⚠ Early stopping triggered!")
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break
        else:
            # No validation dataset -> append nan to keep list lengths consistent
            history["val_losses"].append(float("nan"))
    
    if early_stopping:
        return {"best_state": best_state, "history": history}
    return {"best_state": epoch, "history": history}
 
def plot_history(history: dict, filename: str = "training_history.png") -> None:
    """Plotta e salva il grafico di train e validation loss per epoca."""
    epochs = history.get("train_epochs", [])
    train_losses = history.get("train_losses", [])
    val_losses = history.get("val_losses", [])

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    
    # Aggiunge linea verticale per la miglior validation loss
    if val_losses:
        best_val_idx = val_losses.index(min(val_losses))
        best_epoch = epochs[best_val_idx]
        plt.axvline(x=best_epoch, color='k', linestyle='--', label=f'Best Epoch ({best_epoch})')

    # assicurarsi che il tick 0 sia mostrato (se ci sono epoche)
    if epochs:
        plt.xticks(epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train & Validation Loss per Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    print(f"Saved training plot to {filename}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close()

    plt.close()