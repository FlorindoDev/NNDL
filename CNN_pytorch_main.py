import torch
from torch import nn
import torch.nn.functional as F
from Model.pytorch.CNN import (
    setup_device, set_seed, load_datasets, create_train_val_split, 
    CNN, Config, train, save_model, load_model, test_loop, plot_history
)

def main():
    """Entry point principale per CNN PyTorch."""
    # Flag di esecuzione
    TRAIN_AND_SAVE = False  # Se True, allena il modello e salva i pesi
    LOAD_AND_TEST = True    # Se True, carica i pesi esistenti e testa il modello

    device = setup_device()
    set_seed(16)
    
    # Carica dati
    training_data, test_data = load_datasets()
    train_ds, val_ds = create_train_val_split(training_data)
    
    # Crea modello
    fun_activation = F.relu
    pooling = nn.MaxPool2d(kernel_size=2, stride=2)
    # Esempio di architettura complessa (tuple di layer)
    conv_layer_1 = (
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=1),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=1)
    )
    conv_layer_2 = (nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1))

    model = CNN(
        fun_activation,
        pooling,
        nn.LazyLinear(10),
        conv_layer_1,
        conv_layer_2
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()

    if TRAIN_AND_SAVE:
        print(f"\n--- TRAINING MODE (CNN) ---")
        print(f"Dataset sizes: Training: {len(train_ds)}, Validation: {len(val_ds)}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        result = train(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            early_stopping=True
        )
        
        save_model(model, Config.MODEL_SAVE_PATH)
    

        if "history" in result:
            plot_history(result["history"])

    if LOAD_AND_TEST:
        print(f"\n--- LOAD & TEST MODE (CNN) ---")
        load_model(model, Config.MODEL_SAVE_PATH)
        
        print("\nFINAL TEST EVALUATION")
        print("=" * 50)
        test_loop(test_data, model, loss_fn, device)

if __name__ == "__main__":
    main()
