"""Gradio demo for retinal blood vessel segmentation.

Loads a trained U-Net checkpoint and segments blood vessels from an uploaded
retinal fundus image.

Usage:
    python app.py --weights models/unet.pth

If no GPU is available the model runs on CPU. The app preprocesses the input
exactly like the training pipeline: convert to grayscale, resize to 512x512,
scale to [0, 1]. The output is the binary vessel mask (sigmoid > 0.5).
"""

import argparse

import cv2
import gradio as gr
import numpy as np
import torch

from src.model import build_unet

H = W = 512


def load_model(weights_path, device):
    model = build_unet().to(device)
    state = torch.load(weights_path, map_location=device)
    # Support both raw state_dict and {"model_state_dict": ...} checkpoints.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(image):
    """RGB/gray uint8 array -> (1, 1, H, W) float tensor in [0, 1]."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (W, H))
    image = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)[None, None, ...]
    return tensor


def segment(image, model, device):
    if image is None:
        return None
    with torch.no_grad():
        x = preprocess(image).to(device)
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8) * 255
    return mask


def build_interface(model, device):
    return gr.Interface(
        fn=lambda img: segment(img, model, device),
        inputs=gr.Image(type="numpy", label="Retinal fundus image"),
        outputs=gr.Image(type="numpy", label="Segmented vessels"),
        title="Retinal Blood Vessel Segmentation (U-Net)",
        description="Upload a retinal fundus image to extract the blood vessel mask.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="models/unet.pth",
        help="Path to the trained U-Net checkpoint (.pth).",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)

    demo = build_interface(model, device)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
