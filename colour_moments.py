import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATASET_PATH = "dataset"
RESULTS_PATH = "results/colour_moments"

os.makedirs(
    RESULTS_PATH,
    exist_ok=True
)


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(folder):

    image_paths = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):

                image_paths.append(
                    os.path.join(
                        root,
                        file
                    )
                )

    return sorted(image_paths)


# ============================================================
# RIPENESS
# ============================================================

def determine_ripeness(
    h_mean,
    s_mean,
    v_mean
):

    if h_mean < 15 and s_mean > 100:
        return "Ripe"

    elif h_mean < 30 and s_mean > 70:
        return "Overripe"

    else:
        return "Unripe"


# ============================================================
# MOMENTS
# ============================================================

def calculate_channel_moments(
    values
):

    values = values.astype(
        np.float64
    )

    if len(values) == 0:

        return (
            0.0,
            0.0,
            0.0
        )

    mean = np.mean(
        values
    )

    std = np.std(
        values
    )

    if std == 0:

        skewness = 0.0

    else:

        skewness = np.mean(
            (
                (values - mean) /
                std
            ) ** 3
        )

    return (
        mean,
        std,
        skewness
    )


# ============================================================
# COLOUR MOMENTS
# ============================================================

def calculate_colour_moments(
    image,
    mask
):

    pixels = image[
        mask > 0
    ]

    if len(pixels) == 0:

        return {

            "R_mean": 0.0,
            "R_std": 0.0,
            "R_skewness": 0.0,

            "G_mean": 0.0,
            "G_std": 0.0,
            "G_skewness": 0.0,

            "B_mean": 0.0,
            "B_std": 0.0,
            "B_skewness": 0.0,

            "H_mean": 0.0,
            "H_std": 0.0,
            "H_skewness": 0.0,

            "S_mean": 0.0,
            "S_std": 0.0,
            "S_skewness": 0.0,

            "V_mean": 0.0,
            "V_std": 0.0,
            "V_skewness": 0.0
        }

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    B = pixels[:, 0]
    G = pixels[:, 1]
    R = pixels[:, 2]

    R_mean, R_std, R_skew = (
        calculate_channel_moments(R)
    )

    G_mean, G_std, G_skew = (
        calculate_channel_moments(G)
    )

    B_mean, B_std, B_skew = (
        calculate_channel_moments(B)
    )


    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hsv_pixels = hsv[
        mask > 0
    ]

    H = hsv_pixels[:, 0]
    S = hsv_pixels[:, 1]
    V = hsv_pixels[:, 2]

    H_mean, H_std, H_skew = (
        calculate_channel_moments(H)
    )

    S_mean, S_std, S_skew = (
        calculate_channel_moments(S)
    )

    V_mean, V_std, V_skew = (
        calculate_channel_moments(V)
    )


    return {

        "R_mean": R_mean,
        "R_std": R_std,
        "R_skewness": R_skew,

        "G_mean": G_mean,
        "G_std": G_std,
        "G_skewness": G_skew,

        "B_mean": B_mean,
        "B_std": B_std,
        "B_skewness": B_skew,

        "H_mean": H_mean,
        "H_std": H_std,
        "H_skewness": H_skew,

        "S_mean": S_mean,
        "S_std": S_std,
        "S_skewness": S_skew,

        "V_mean": V_mean,
        "V_std": V_std,
        "V_skewness": V_skew
    }


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    image_path
):

    image = cv2.imread(
        image_path
    )

    if image is None:
        return None

    # Standalone mask
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower = np.array(
        [0, 40, 30]
    )

    upper = np.array(
        [179, 255, 255]
    )

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    features = calculate_colour_moments(
        image,
        mask
    )

    ripeness = determine_ripeness(
        features["H_mean"],
        features["S_mean"],
        features["V_mean"]
    )

    features["ripeness"] = ripeness

    return {
        "image": image,
        "mask": mask,
        "features": features
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n======================================")
    print("COLOUR MOMENTS FEATURE EXTRACTION")
    print("======================================")

    image_paths = find_images(
        DATASET_PATH
    )

    print(
        f"Images found: {len(image_paths)}"
    )

    if len(image_paths) == 0:

        print(
            "No images found."
        )

        return

    results = []

    for image_path in image_paths:

        result = process_image(
            image_path
        )

        if result is None:
            continue

        row = {
            "image":
                os.path.basename(
                    image_path
                )
        }

        row.update(
            result["features"]
        )

        results.append(
            row
        )

    df = pd.DataFrame(
        results
    )

    csv_path = os.path.join(
        RESULTS_PATH,
        "colour_moments_results.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print("\n======================================")
    print("COLOUR MOMENTS RESULTS")
    print("======================================")

    print(
        f"Images processed: {len(df)}"
    )

    print(
        f"\nAverage R Mean: "
        f"{df['R_mean'].mean():.2f}"
    )

    print(
        f"Average G Mean: "
        f"{df['G_mean'].mean():.2f}"
    )

    print(
        f"Average B Mean: "
        f"{df['B_mean'].mean():.2f}"
    )

    print(
        f"\nAverage H Mean: "
        f"{df['H_mean'].mean():.2f}"
    )

    print(
        f"Average S Mean: "
        f"{df['S_mean'].mean():.2f}"
    )

    print(
        f"Average V Mean: "
        f"{df['V_mean'].mean():.2f}"
    )

    print("\nRipeness:")

    print(
        df["ripeness"].value_counts()
    )

    print(
        f"\nCSV saved to: {csv_path}"
    )

    # Display first image

    result = process_image(
        image_paths[0]
    )

    if result is not None:

        features = result[
            "features"
        ]

        plt.figure(
            figsize=(12, 7)
        )

        plt.subplot(1, 2, 1)

        plt.imshow(
            cv2.cvtColor(
                result["image"],
                cv2.COLOR_BGR2RGB
            )
        )

        plt.title(
            "Input Image"
        )

        plt.axis("off")

        plt.subplot(1, 2, 2)

        plt.axis("off")

        text = (
            "COLOUR MOMENTS\n\n"

            "RGB\n"
            f"R Mean: "
            f"{features['R_mean']:.2f}\n"
            f"R Std: "
            f"{features['R_std']:.2f}\n"
            f"R Skew: "
            f"{features['R_skewness']:.2f}\n\n"

            f"G Mean: "
            f"{features['G_mean']:.2f}\n"
            f"G Std: "
            f"{features['G_std']:.2f}\n"
            f"G Skew: "
            f"{features['G_skewness']:.2f}\n\n"

            f"B Mean: "
            f"{features['B_mean']:.2f}\n"
            f"B Std: "
            f"{features['B_std']:.2f}\n"
            f"B Skew: "
            f"{features['B_skewness']:.2f}\n\n"

            "HSV\n"
            f"H Mean: "
            f"{features['H_mean']:.2f}\n"
            f"S Mean: "
            f"{features['S_mean']:.2f}\n"
            f"V Mean: "
            f"{features['V_mean']:.2f}\n\n"

            f"Ripeness: "
            f"{features['ripeness']}"
        )

        plt.text(
            0.05,
            0.95,
            text,
            verticalalignment="top",
            fontsize=10
        )

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()