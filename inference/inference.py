import argparse
import h5py
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt


def load_mri_and_mask(h5_path):
    """Load MRI slice and ground-truth mask."""
    with h5py.File(h5_path, "r") as f:
        image = f["x"][:]
        mask = f["y"][:]

    return image, mask


def preprocess(image):
    """Convert MRI to ONNX input format."""
    return image.astype(np.float32)[None, None, :, :]


def predict(session, image):
    """Run ONNX inference."""
    input_name = session.get_inputs()[0].name

    output = session.run(
        None,
        {input_name: preprocess(image)}
    )[0]

    probability = 1.0 / (1.0 + np.exp(-output))

    prediction = (
        probability > 0.5
    ).astype(np.uint8)[0, 0]

    return prediction


def calculate_dice(prediction, ground_truth):
    """Calculate binary Dice score."""

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    dice = (
        2.0 * intersection
        / (
            prediction.sum()
            + ground_truth.sum()
            + 1e-8
        )
    )

    return dice


def save_result(
    image,
    ground_truth,
    prediction,
    dice,
    model_type,
    output_path
):
    """Save MRI, ground truth and prediction comparison."""

    ground_truth_binary = (ground_truth > 0).astype(np.uint8)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("MRI")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_binary, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="gray")
    plt.title(f"{model_type} Prediction - Dice {dice * 100:.2f}%")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def main():

    parser = argparse.ArgumentParser(
        description="Brain Tumor Segmentation using ONNX U-Net"
    )

    parser.add_argument(
        "--model",
        default="models/unet_int8.onnx"
    )

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        default="prediction.png"
    )

    args = parser.parse_args()

    model_type = (
        "INT8"
        if "int8" in args.model.lower()
        else "FP32"
    )

    print(f"Loading {model_type} ONNX model...")
    print(f"Model path: {args.model}")

    session = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"]
    )

    print("Model loaded successfully.")

    image, ground_truth = load_mri_and_mask(
        args.input
    )

    print("MRI shape:", image.shape)
    print("MRI dtype:", image.dtype)

    prediction = predict(
        session,
        image
    )

    ground_truth_binary = (
        ground_truth > 0
    ).astype(np.uint8)

    dice = calculate_dice(
        prediction,
        ground_truth_binary
    )

    print("Prediction completed.")
    print("Predicted tumor pixels:", int(prediction.sum()))
    print(
        "Ground-truth tumor pixels:",
        int(ground_truth_binary.sum())
    )
    print(f"Dice score: {dice * 100:.2f}%")

    save_result(
        image,
        ground_truth,
        prediction,
        dice,
        model_type,
        args.output
    )

    print("Result saved:", args.output)


if __name__ == "__main__":
    main()