import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.model import MiniVLA


# ====================================================
# Configuration
# ====================================================

BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_WORKERS = 4

BEST_MODEL_PATH = "results/best_minivla.pt"
TRAINING_CURVES_PATH = "results/training_curves.png"


# ====================================================
# Reproducibility
# ====================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ====================================================
# Device
# ====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Performance
torch.set_float32_matmul_precision("high")


# ====================================================
# Training Function
# ====================================================

def train_model(
    train_dataset,
    val_dataset,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    num_workers=NUM_WORKERS,
    best_model_path=BEST_MODEL_PATH,
    training_curves_path=TRAINING_CURVES_PATH
):

    # ================================================
    # Dataset sizes
    # ================================================

    print("\nDataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")


    # ================================================
    # DataLoaders
    # ================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )


    # Model
    model = MiniVLA().to(device)


    # ================================================
    # Trainable Parameters
    # ================================================

    trainable_params = [
        param
        for param in model.parameters()
        if param.requires_grad
    ]

    total_params = sum(
        param.numel()
        for param in model.parameters()
    )

    trainable_params_count = sum(
        param.numel()
        for param in trainable_params
    )

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params_count:,}")
    print(f"Trainable ratio:      {100 * trainable_params_count / total_params:.2f}%")


    # Loss
    criterion = nn.MSELoss()


    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)


    # ================================================
    # Mixed Precision
    # ================================================

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


    # ================================================
    # History
    # ================================================

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")


    # Training Loop
    for epoch in range(num_epochs):

        # --------------------------------------------
        # Train
        # --------------------------------------------

        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for batch in train_progress:

            images = batch["image"].to(device, non_blocking=True)
            states = batch["state"].to(device, non_blocking=True)
            actions = batch["action"].to(device, non_blocking=True)
            instructions = batch["instruction"]

            optimizer.zero_grad(set_to_none=True)


            # ----------------------------------------
            # Forward + Loss
            # ----------------------------------------

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):

                predictions = model(images, instructions, states)
                loss = criterion(predictions, actions)


            # ----------------------------------------
            # Backpropagation
            # ----------------------------------------

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            # Accumulate loss
            train_loss += (loss.item() * images.size(0))


        train_loss /= len(train_dataset)


        # --------------------------------------------
        # Validation
        # --------------------------------------------

        model.eval()
        val_loss = 0.0

        with torch.no_grad():

            for batch in val_loader:

                images = batch["image"].to(device, non_blocking=True)
                states = batch["state"].to(device, non_blocking=True)
                actions = batch["action"].to(device, non_blocking=True)
                instructions = batch["instruction"]


                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):

                    predictions = model(images, instructions, states)
                    loss = criterion(predictions, actions)

                val_loss += (loss.item() * images.size(0))


        val_loss /= len(val_dataset)


        # --------------------------------------------
        # Store losses
        # --------------------------------------------

        train_losses.append(train_loss)
        val_losses.append(val_loss)


        # --------------------------------------------
        # Save best model
        # --------------------------------------------

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            best_marker = " ← BEST"

        else:
            best_marker = ""


        # Epoch result
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}{best_marker}")


    # ================================================
    # Save Training Curves
    # ================================================

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MiniVLA Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(training_curves_path, dpi=300, bbox_inches="tight")
    plt.close()


    # ================================================
    # Final Message
    # ================================================

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training curves saved to: {training_curves_path}")
