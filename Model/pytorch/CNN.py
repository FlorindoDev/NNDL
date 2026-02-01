import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from dotenv import load_dotenv
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import numpy as np
import random


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


class CNN(nn.Module):
   def __init__(self, fun_activation,pooling,linear_layer: nn.Linear,*args,**kwargs):
        """
        Building blocks of convolutional neural network.

        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(CNN, self).__init__()

        #self.layers = args
        self.blocks = nn.ModuleList()

        #args: ((conv1,conv2),(conv3),...)
        for block_in_args in args: #block_in_args è una tupla

            # Se block_in_args non è iterabile (es. è un singolo layer come (conv3) che python valuta come conv3),
            # lo avvolgiamo in una lista.
            if isinstance(block_in_args, (tuple, list)):
                self.blocks.append(nn.ModuleList(block_in_args))
            else:
                self.blocks.append(nn.ModuleList([block_in_args]))

    
        self.fun_activation=fun_activation
        self.pool = pooling
        self.fc1 = linear_layer
        

   def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: Input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """

        #Per ogni blocco di Conv applico la Conv e la funzione di ativazione e infine faccio il pooling 
        for block in self.blocks:
            for layer in block:
                #(conv1,conv2) 
                x = layer(x)
                x = self.fun_activation(x)
                
            x = self.pool(x)
        
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)            # Apply fully connected layer             
           
        return x    
   


# Configuration
class Config:
    BATCH_SIZE = 32
    EPOCHS = 50
    PATIENCE = 5
    TRAIN_SPLIT_RATIO = 0.7
    DATA_ROOT = "data"
    LEARNING_RATE = 0.001



def create_train_val_split(dataset, train_ratio: float = Config.TRAIN_SPLIT_RATIO):
    """Divide il dataset in training e validation set."""
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    train_ds = torch.utils.data.Subset(dataset, range(0, n_train))
    val_ds = torch.utils.data.Subset(dataset, range(n_train, n_total))
    return train_ds, val_ds


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


def train_loop(train_ds, model: nn.Module, loss_fn, optimizer, device, batch_size: int = Config.BATCH_SIZE, verbose: bool = True):
    train_dataloader = DataLoader(train_ds, batch_size, shuffle=True)
    model.train()

    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()

    avg_loss = 0.0
    num_batches = len(train_dataloader)

    for batch_idx, (X, Y) in enumerate(tqdm(train_dataloader, desc="Training")):
        X, Y = X.to(device), Y.to(device)

        logits = model(X)

        if verbose and batch_idx == 0:
            tqdm.write(f"Batch on: {X.device}, Logits on: {logits.device}")

        loss = loss_fn(logits, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        avg_loss += loss.item()

    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()
        
    return avg_loss / num_batches


@torch.no_grad()
def evaluate_loss(model: nn.Module, dataloader: DataLoader, loss_fn, device: torch.device):
    model.eval()
    val_losses = []

    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        loss = loss_fn(logits, Y)
        val_losses.append(loss.cpu().item())

    return float(torch.tensor(val_losses).float().mean().item())


def train(model: nn.Module, train_ds, val_ds, loss_fn, optimizer, device: torch.device, epochs: int = Config.EPOCHS, patience: int = Config.PATIENCE, batch_size: int = Config.BATCH_SIZE, early_stopping: bool = False):
    """Training completo con early stopping (opzionale)."""
    if early_stopping:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
        best_validation_loss = float("inf")
        best_state = None
        current_patience = patience
        history = {"train_epochs": [], "val_losses": [], "train_losses": []}

    for epoch in range(epochs):
        print(f"\n\nEpoch {epoch + 1}/{epochs}\n" + "-" * 40)

        epoch_train_loss = train_loop(train_ds, model, loss_fn, optimizer, device, batch_size)
        print(f"Training loss: {epoch_train_loss:.6f}")

        if early_stopping:
            val_loss = evaluate_loss(model, val_loader, loss_fn, device)
            print(f"Validation loss: {val_loss:.6f}")

            history["train_epochs"].append(epoch + 1)
            history["val_losses"].append(val_loss)
            history["train_losses"].append(epoch_train_loss)

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
                model.load_state_dict(best_state)
                break

    if early_stopping:
        return {"best_state": best_state, "history": history}

    return {"best_state": epoch}


def test_loop(
    test_data,
    model: nn.Module,
    loss_fn,
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
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True)
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


def main():
    """Entry point principale."""
    
    device=setup_device()
    # Carica dati
    training_data, test_data = load_datasets()
    train_ds, val_ds = create_train_val_split(training_data)
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_ds)}")
    print(f"  Validation: {len(val_ds)}")
    print(f"  Test: {len(test_data)}")
    
    set_seed()

    fun_activation = F.relu
    pooling = nn.MaxPool2d(kernel_size=2,stride=2)

    conv_layer_1 = (nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=1),nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding=1))

    conv_layer_2 = (nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, padding=1), nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, padding=1))

    # Crea modello
    model = CNN(
        fun_activation,
        pooling,
        nn.LazyLinear(10),
        conv_layer_1,
        conv_layer_2,
    ).to(device)
    
    # Setup training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training
    result = train(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        early_stopping=True
    )
    
    # Test finale
    print("\n" + "=" * 50)
    print("FINAL TEST EVALUATION")
    print("=" * 50)
    test_loop(test_data, model, loss_fn, device)
    if "history" in result:
        plot_history(result["history"])


if __name__ == "__main__":
    main()