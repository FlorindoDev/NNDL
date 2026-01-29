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
