import os
import glob
import h5py
import numpy as np
import onnxruntime as ort


DATASET_DIR = r"C:\Users\SUBHAM\Desktop\IEEE_Dataset\extracted\BRATS_001"

FP32_MODEL = "models/unet_fp32.onnx"
INT8_MODEL = "models/unet_int8.onnx"


def load_sample(path):
    with h5py.File(path, "r") as f:
        image = f["x"][:]
        mask = f["y"][:]

    image = image.astype(np.float32)[None, None, :, :]
    mask = (mask > 0).astype(bool)

    return image, mask


def predict(session, image):
    input_name = session.get_inputs()[0].name

    output = session.run(
        None,
        {input_name: image}
    )[0]

    probability = 1.0 / (1.0 + np.exp(-output))

    prediction = probability > 0.5

    return prediction[0, 0]


def dice_score(prediction, ground_truth):
    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    denominator = (
        prediction.sum()
        + ground_truth.sum()
    )

    return (
        2.0 * intersection
        / (denominator + 1e-8)
    )


def evaluate(model_path, files):

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    scores = []

    for path in files:

        image, ground_truth = load_sample(path)

        prediction = predict(
            session,
            image
        )

        dice = dice_score(
            prediction,
            ground_truth
        )

        scores.append(dice)

    return np.array(scores)


def print_results(name, scores):

    print(f"\n{name}")
    print("-" * 40)

    print(
        f"Mean Dice:   {scores.mean() * 100:.2f}%"
    )

    print(
        f"Median Dice: {np.median(scores) * 100:.2f}%"
    )

    print(
        f"Minimum:     {scores.min() * 100:.2f}%"
    )

    print(
        f"Maximum:     {scores.max() * 100:.2f}%"
    )


def main():

    files = sorted(
        glob.glob(
            os.path.join(
                DATASET_DIR,
                "BRATS_001_*.h5"
            )
        )
    )

    print("Found", len(files), "H5 files.")

    if not files:
        raise FileNotFoundError(
            "No H5 files found."
        )

    print("\nFiles being evaluated:")

    for path in files:
        print(
            " ",
            os.path.basename(path)
        )

    print("\nRunning FP32 validation...")

    fp32_scores = evaluate(
        FP32_MODEL,
        files
    )

    print("\nRunning INT8 validation...")

    int8_scores = evaluate(
        INT8_MODEL,
        files
    )

    print_results(
        "FP32 RESULTS",
        fp32_scores
    )

    print_results(
        "INT8 RESULTS",
        int8_scores
    )

    differences = (
        fp32_scores - int8_scores
    )

    print("\nQUANTIZATION COMPARISON")
    print("-" * 40)

    print(
        f"Mean Dice difference: "
        f"{differences.mean() * 100:.2f} percentage points"
    )

    print(
        f"Mean absolute difference: "
        f"{np.abs(differences).mean() * 100:.2f} percentage points"
    )

    within_one = (
        np.abs(differences) <= 0.01
    ).sum()

    print(
        f"INT8 within 1 percentage point: "
        f"{within_one}/{len(files)} slices"
    )

    print("\nPER-SLICE RESULTS")
    print("-" * 40)

    for path, fp32, int8 in zip(
        files,
        fp32_scores,
        int8_scores
    ):

        print(
            f"{os.path.basename(path):20s} "
            f"FP32: {fp32 * 100:6.2f}%   "
            f"INT8: {int8 * 100:6.2f}%   "
            f"Δ: {(fp32 - int8) * 100:+6.2f} pp"
        )


if __name__ == "__main__":
    main()