# Implementation and Benchmarking of FCNN and CNN Architectures on MNIST

## NNDL - Simple Neural Network (MNIST)

Implementazione didattica di una rete neurale fully-connected in NumPy,
con training su MNIST. Il codice include attivazioni, loss, inizializzazioni
pesi e regole di aggiornamento (SGD,RProp e Adam).

### Requisiti
- Python 3.10+ consigliato
- Dipendenze: `requirements.txt`

### Installazione
#### Windows (PowerShell)
```powershell
python -m venv .NNDL
.\.NNDL\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Linux/macOS
```bash
python3 -m venv .NNDL
source .NNDL/bin/activate
pip install -r requirements.txt
```

### Esecuzione
```bash
python main.py
```

`main.py` scarica MNIST via `tensorflow.keras.datasets`, normalizza le immagini,
allena una rete fully-connected e mostra accuracy + grafico della loss.

### Struttura progetto
- `main.py` - entrypoint di training/valutazione su MNIST
- `Model/NeuralNetwork.py` - implementazione della rete (forward, backward, train, evaluate)
- `common/activation.py` - ReLU, LeakyReLU, Softmax + derivate
- `common/loss.py` - CrossEntropy
- `common/weight_Init.py` - He, Glorot
- `common/update_rule.py` - StandardUpdateRule (SGD) e RProp
- `common/logger.py` - logger con attivazione via variabile d'ambiente

### Configurazione
Il logger si abilita impostando la variabile `LOGGER=TRUE` nel file `.env`.
Esempio (gia presente):
```
LOGGER=TRUE
```

### Note
- La rete assume output `num_classes` e usa Softmax + CrossEntropy.
- Il training usa batch_size e epochs configurabili in `main.py`.

## Implementazioni PyTorch
Oltre all'implementazione manuale (NumPy), il progetto include modelli implementati con PyTorch.

### Modelli Disponibili
- **FCNN (Fully Connected Neural Network)**: `Model/pytorch/FCNN.py`
  - Implementazione MLP classica su MNIST.
- **CNN (Convolutional Neural Network)**: `Model/pytorch/CNN.py`
  - Rete convoluzionale per feature extraction e classificazione su MNIST.

### Esecuzione Modelli PyTorch
Per allenare i modelli PyTorch, eseguire direttamente gli script:

#### CNN
```bash
python Model/pytorch/CNN.py
```

#### FCNN
```bash
python Model/pytorch/FCNN.py
```

### Configurazione GPU
I modelli PyTorch supportano l'accelerazione GPU. Per abilitarla, assicurarsi di avere CUDA/ROCm configurati e impostare nel `.env` (opzionale se lo script rileva automaticamente):
```
USE_GPU=TRUE
```
