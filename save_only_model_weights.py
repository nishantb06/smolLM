import torch
from smollm_training import SmolLM, SmolLMConfig, SmolLMLightning


def save_only_model_weights(checkpoint_path: str, output_path: str):
    """
    Load a Lightning checkpoint and save only the model weights.

    Args:
        checkpoint_path (str): Path to the Lightning checkpoint file
        output_path (str): Path where to save the model weights
    """
    # Load the Lightning checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Create a new state dict with only the model weights
    model_state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        # Remove the 'model.' prefix from the keys
        if key.startswith("model."):
            new_key = key[6:]  # Remove 'model.' prefix
            model_state_dict[new_key] = value

    # Save only the model weights
    print(f"Saving model weights to {output_path}")
    torch.save(model_state_dict, output_path)
    print("Done!")


if __name__ == "__main__":
    checkpoint_path = "best-checkpoint.ckpt"
    output_path = "model_weights.pt"
    save_only_model_weights(checkpoint_path, output_path)
