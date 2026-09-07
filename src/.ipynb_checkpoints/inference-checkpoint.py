import numpy as np
import torch

from src.model import MiniVLA


CHECKPOINT_PATH = "results/best_minivla.pt"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gripper State
def gripper_state(value):

    if value < 0:
        return "CLOSED"

    return "OPEN"


# Print Human-Readable Action
def print_action(action, title):

    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)
    print(f"X movement       : {action[0]:+.4f}")
    print(f"Y movement       : {action[1]:+.4f}")
    print(f"Z movement       : {action[2]:+.4f}")
    print(f"Rotation X       : {action[3]:+.4f}")
    print(f"Rotation Y       : {action[4]:+.4f}")
    print(f"Rotation Z       : {action[5]:+.4f}")
    print(f"Gripper          : {action[6]:+.4f} → {gripper_state(action[6])}")

# ==========================================================
# Inference Function
# ==========================================================

def run_inference(test_dataset, sample_index=0, checkpoint_path=CHECKPOINT_PATH):

    # Load Model
    model = MiniVLA().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)


    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")


    # Validate Sample Index
    if not 0 <= sample_index < len(test_dataset):
        raise IndexError(f"sample_index must be between 0 and {len(test_dataset) - 1}")

    # ======================================================
    # Get Test Sample
    # ======================================================

    sample = test_dataset[sample_index]
    image_tensor = sample["image"]
    state = (sample["state"].unsqueeze(0).to(device))
    instruction = sample["instruction"]
    ground_truth = (sample["action"].to(device))
    image = (image_tensor.unsqueeze(0).to(device))

    # ======================================================
    # Inference
    # ======================================================

    with torch.no_grad():
        prediction = model(image, [instruction], state)

    prediction = prediction.squeeze(0)

    # ======================================================
    # Compute Metrics
    # ======================================================

    difference = prediction - ground_truth
    mse = torch.mean(difference ** 2).item()
    mae = torch.mean(torch.abs(difference)).item()

    # ======================================================
    # Convert Image to NumPy
    # ======================================================

    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy())
    image_np = np.clip(image_np, 0, 1)

    # ======================================================
    # Convert Actions to NumPy
    # ======================================================

    ground_truth_np = (ground_truth.cpu().numpy())
    prediction_np = (prediction.cpu().numpy())

    # ======================================================
    # Inference Result
    # ======================================================

    print("\n" + "-" * 50)
    print("INFERENCE RESULT")
    print("-" * 50)
    print(f"Sample index: {sample_index}")
    print("\nInstruction:")
    print(instruction)


    # ======================================================
    # Raw Ground Truth
    # ======================================================

    print("\nGround Truth Action:")
    print(np.array2string(ground_truth_np, precision=4, suppress_small=True))


    # ======================================================
    # Raw Prediction
    # ======================================================

    print("\nPredicted Action:")
    print(np.array2string(prediction_np, precision=4, suppress_small=True))


    # ======================================================
    # Metrics
    # ======================================================

    print(f"\nAction MSE: {mse:.6f}")
    print(f"Action MAE: {mae:.6f}")

    # Human-Readable Ground Truth
    print_action(ground_truth_np, "GROUND TRUTH — HUMAN READABLE")

    # Human-Readable Prediction
    print_action(prediction_np, "PREDICTION — HUMAN READABLE")


    # ======================================================
    # Action Comparison
    # ======================================================

    gt_gripper = gripper_state(ground_truth_np[6])
    pred_gripper = gripper_state(prediction_np[6])

    print("\n" + "-" * 50)
    print("ACTION COMPARISON")
    print("-" * 50)

    if gt_gripper == pred_gripper:
        print(f" Gripper state agrees: {gt_gripper}")
    else:
        print(f" Gripper state differs: Ground Truth = {gt_gripper}, Prediction = {pred_gripper}")

    # Return Image
    return image_np

