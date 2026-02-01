import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from dotenv import load_dotenv
import torch.nn.functional as F
import os
from tqdm import tqdm


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
        self.layers = nn.ModuleList(args)
        self.fun_activation=fun_activation
        self.pool = pooling
        self.fc1 = linear_layer
        
        # # 1st convolutional layer
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        # # Max pooling layer
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # # 2nd convolutional layer
        # self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        # # Fully connected layer
        # self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

   def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: Input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """

        for layer in self.layers:
            x = self.fun_activation(layer(x))
            x = self.pool(x)
        
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)            # Apply fully connected layer             
           
        return x    
   
    #    x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
    #    x = self.pool(x)           # Apply max pooling
    #    x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
    #    x = self.pool(x)           # Apply max pooling
    #    x = x.reshape(x.shape[0], -1)  # Flatten the tensor
    #    x = self.fc1(x)            # Apply fully connected layer


# Configuration
class Config:
    BATCH_SIZE = 32
    EPOCHS = 10
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

    for batch_idx, (X, Y) in enumerate(tqdm(train_dataloader, desc="Training")):
        X, Y = X.to(device), Y.to(device)

        logits = model(X)

        if verbose and batch_idx == 0:
            print(f"Batch on: {X.device}, Logits on: {logits.device}")

        loss = loss_fn(logits, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()


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
        history = {"train_epochs": [], "val_losses": []}

    for epoch in range(epochs):
        print(f"\n\nEpoch {epoch + 1}/{epochs}\n" + "-" * 40)

        train_loop(train_ds, model, loss_fn, optimizer, device, batch_size)

        if early_stopping:
            val_loss = evaluate_loss(model, val_loader, loss_fn, device)
            print(f"Validation loss: {val_loss:.6f}")

            history["train_epochs"].append(epoch + 1)
            history["val_losses"].append(val_loss)

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                current_patience = patience
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
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
    

    fun_activation = F.relu
    pooling = nn.MaxPool2d(kernel_size=2,stride=2)
  
    # Crea modello
    model = CNN(
        fun_activation,
        pooling,
        nn.Linear(16*7*7, 10),
        nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
    ).to(device)
    
    # Setup training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training
    result = train(
        model=model,
        train_ds=training_data,
        val_ds=val_ds,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        early_stopping=False
    )
    
    # Test finale
    print("\n" + "=" * 50)
    print("FINAL TEST EVALUATION")
    print("=" * 50)
    test_loop(test_data, model, loss_fn, device)


if __name__ == "__main__":
    main()