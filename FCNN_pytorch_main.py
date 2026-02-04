import torch
from torch import nn
from Model.pytorch.FCNN import (
    setup_device, set_seed, load_datasets, create_train_val_split, 
    create_model, Config, train, save_model, load_model, test_loop, plot_history
)

def main():
    """Entry point principale per FCNN PyTorch."""
    # Flag di esecuzione
    TRAIN_AND_SAVE = False   # Se True, allena il modello e salva i pesi
    LOAD_AND_TEST = True     # Se True, carica i pesi esistenti e testa il modello

    # Setup
    device = setup_device()
    set_seed(16)
    
    # Carica dati
    training_data, test_data = load_datasets()
    train_ds, val_ds = create_train_val_split(training_data)
    
    # Crea modello
    model = create_model(
        input_size=Config.INPUT_SIZE,
        hidden_layers=[256, 128, 64],
        output_size=Config.OUTPUT_SIZE,
        activations=nn.LeakyReLU
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()

    if TRAIN_AND_SAVE:
        print(f"\n--- TRAINING MODE (FCNN) ---")
        print(f"Dataset sizes: Training: {len(train_ds)}, Validation: {len(val_ds)}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        result = train(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            batch_size=Config.BATCH_SIZE,
            early_stopping=True
        )
        
        save_model(model, Config.MODEL_SAVE_PATH)

        if "history" in result:
            plot_history(result["history"])

    if LOAD_AND_TEST:
        print(f"\n--- LOAD & TEST MODE (FCNN) ---")
        load_model(model, Config.MODEL_SAVE_PATH)
        
        print("\nFINAL TEST EVALUATION")
        print("=" * 50)
        test_loop(test_data, model, loss_fn, device)

if __name__ == "__main__":
    main()
