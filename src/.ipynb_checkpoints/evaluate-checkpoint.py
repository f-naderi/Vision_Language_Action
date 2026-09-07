import torch
from torch.utils.data import DataLoader 

from src.model import MiniVLA


# ===========================
# Configuration
# ===========================

CHECKPOINT_PATH = "results/best_minivla.pt"
BATCH_SIZE = 64
NUM_WORKERS = 4


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# Evaluation Function
# ===========================

def evaluate_model(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, checkpoint_path=CHECKPOINT_PATH):
    

    # ===========================
    # Test DataLoader
    # ===========================

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0
    )

    print("\nTest dataset:")
    print(f"Test: {len(test_dataset)}")


    # ===========================
    # Load model
    # ===========================

    model = MiniVLA().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)


    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)


    model.eval()

    print(f"\nLoaded checkpoint: {checkpoint_path}")


    # ===========================
    # Evaluation
    # ===========================

    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_values = 0


    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(device, non_blocking=True)
            states = batch["state"].to(device, non_blocking=True)
            actions = batch["action"].to(device, non_blocking=True)
            instructions = batch["instruction"]

            # Model prediction
            predictions = model(images, instructions, states)

            # Errors
            squared_error = (predictions - actions) ** 2
            absolute_error = torch.abs(predictions - actions)
            total_squared_error += (squared_error.sum().item())
            total_absolute_error += (absolute_error.sum().item())
            total_values += actions.numel()


    # ===========================
    # Final metrics
    # ===========================

    test_mse = (total_squared_error / total_values)
    test_mae = (total_absolute_error / total_values)


    # ===========================
    # Results
    # ===========================

    print("\n" + "-" * 50)
    print("FINAL TEST RESULTS")
    print("-" * 50)

    print(f"Test samples: {len(test_dataset)}")
    print(f"Test MSE:     {test_mse:.6f}")
    print(f"Test MAE:     {test_mae:.6f}")




