# MNIST Neural Network Implementation & Benchmarking

---
## [English Version]

This repository contains various implementations of Neural Networks (FCNN and CNN) for handwriting digit classification using the MNIST dataset. The project includes a "manual" implementation based exclusively on NumPy and modern implementations using PyTorch.

### Requirements and Setup
The project is optimized for **Python 3.13.3**.

#### Dependency Installation
1. Create a virtual environment:
   ```powershell
   python -m venv .NNDL
   ```
2. Activate the environment:
   - **Windows (PowerShell):** `.\.NNDL\Scripts\Activate.ps1`
   - **Linux/macOS:** `source .NNDL/bin/activate`
3. Install necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Project Architecture
The project is structured modularly to separate model logic from common utilities:

- **Entry Points (Root):**
  - `ManualNN_main.py`: Training and testing of the manual NumPy network.
  - `FCNN_pytorch_main.py`: Training and testing of the PyTorch FCNN.
  - `CNN_pytorch_main.py`: Training and testing of the PyTorch CNN.

- **Models (Model/):**
  - `manual/NeuralNetwork.py`: NumPy implementation of the neural network.
  - `pytorch/FCNN.py`: PyTorch Fully Connected model definition.
  - `pytorch/CNN.py`: PyTorch Convolutional model definition.

### Usage Tutorials
All entry point scripts use internal flags (`TRAIN_AND_SAVE`, `LOAD_AND_TEST`) to manage execution flow.

#### 1. Manual Neural Network (NumPy)
The manual implementation allows you to see "under the hood" how forward and backpropagation work.
- **Run:** `python ManualNN_main.py`
- **Weights:** Saved in `Model/manual/weight/fcnn_manual_weights.npz`.

#### 2. PyTorch FCNN (Fully Connected)
A robust and high-performance version of a Multi-Layer Perceptron.
- **Run:** `python FCNN_pytorch_main.py`
- **Flexibility:** Dynamically pass layer sizes (e.g., `[512, 256, 128]`) and set activation functions.
- **Weights:** Saved in `Model/pytorch/weights/fcnn_model.pth`.

#### 3. PyTorch CNN (Convolutional)
The most advanced model, ideal for image recognition.
- **Run:** `python CNN_pytorch_main.py`
- **Structure:** Supports complex convolutional blocks passed as tuples.
- **Weights:** Saved in `Model/pytorch/weights/cnn_model.pth`.

### Benchmarking Results
Extensive testing has been performed on different architectures and hyperparameters. Detailed results can be found in the following files:
- [NNDL-CNN.csv](NNDL-CNN.csv): Benchmarks for Convolutional architectures.
- [NNDL-FCNN.csv](NNDL-FCNN.csv): Benchmarks for Fully Connected architectures.

#### Highlight Results:

| Model Type | Architecture | Activation | Filters | Optimizer | Epoch (Stop) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN** | Conv(1,32)-Conv(32,64)-Conv(64,128) | ReLU | 5x5 | Adam | 11 | **99.3%** |
| **CNN** | Conv(1,32)-Conv(32,64)-Conv(64,128)-Conv(128,256) | LeakyReLU | 5x5 | Adam | 8 | **99.3%** |
| **FCNN** | [784, 512, 256, 128, 64, 10] | LeakyReLU | - | Adam | 14 | **98.2%** |
| **FCNN** | [784, 256, 128, 64, 10] | LeakyReLU | - | Adam | 13 | **97.9%** |

> **Note on fixed parameters:** All tests were conducted using a Learning Rate of 0.001, CrossEntropy Loss, Seed 16, and Early Stopping with Patience 5. Data batching: Adam (32), RProp (Full-Batch).

---

## [Versione Italiana]

Questo repository contiene diverse implementazioni di Reti Neurali (FCNN e CNN) per la classificazione delle cifre scritte a mano del dataset MNIST. Il progetto include un'implementazione "manuale" basata esclusivamente su NumPy e implementazioni moderne che utilizzano PyTorch.

### Requisiti e Setup
Il progetto è ottimizzato per **Python 3.13.3**.

#### Installazione Dipendenze
1. Crea un ambiente virtuale:
   ```powershell
   python -m venv .NNDL
   ```
2. Attiva l'ambiente:
   - **Windows (PowerShell):** `.\.NNDL\Scripts\Activate.ps1`
   - **Linux/macOS:** `source .NNDL/bin/activate`
3. Installa le librerie necessarie:
   ```bash
   pip install -r requirements.txt
   ```

### Architettura del Progetto
Il progetto è strutturato in modo modulare:

- **Entry Points (Root):** `ManualNN_main.py`, `FCNN_pytorch_main.py`, `CNN_pytorch_main.py`.
- **Modelli (Model/):** Implementazioni manuali (NumPy) e PyTorch.
- **Utilities (common/):** Funzioni di attivazione, loss, regole di aggiornamento e pre-processing.

### Risultati dei Test
Sono stati condotti numerosi test per confrontare le performance delle varie architetture. I dettagli sono disponibili nei file:
- `NNDL-CNN.csv`: Test sulle architetture convoluzionali.
- `NNDL-FCNN.csv`: Test sulle architetture fully connected.

#### Risultati Interessanti:

| Tipo Modello | Architettura | Attivazione | Filtri | Ottimizzatore | Epoca (Stop) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN** | Conv(1,32)-Conv(32,64)-Conv(64,128) | ReLU | 5x5 | Adam | 11 | **99.3%** |
| **CNN** | Conv(1,32)-Conv(32,64)-Conv(64,128)-Conv(128,256) | LeakyReLU | 5x5 | Adam | 8 | **99.3%** |
| **FCNN** | [784, 512, 256, 128, 64, 10] | LeakyReLU | - | Adam | 14 | **98.2%** |
| **FCNN** | [784, 256, 128, 64, 10] | LeakyReLU | - | Adam | 13 | **97.9%** |

> **Nota sui parametri fissi:** Tutti i test sono stati eseguiti con Learning Rate 0.001, Loss function CrossEntropy, Seed 16, ed Early Stopping con Patience 5. Gestione Batch: Adam (32), RProp (Full-Batch).

---

## Persistenza dei Modelli
Tutte le reti supportano il salvataggio e il caricamento dei pesi (checkpointing):
- **Manuale:** Usa `save_weights()` e `load_weights()` basati su NumPy.
- **PyTorch:** Usa `save_model()` e `load_model()` basati su `torch.save/load`.

## Configurazione Avanzata
Puoi gestire alcune impostazioni tramite il file `.env`:
- `LOGGER=TRUE`: Abilita i log dettagliati (per la versione manuale).
- `USE_GPU=TRUE`: Tenta di utilizzare CUDA/ROCm per i modelli PyTorch.
