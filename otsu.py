import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATASET_PATH = "dataset"
RESULTS_PATH = "results/otsu"

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
# OTSU THRESHOLDING
# ============================================================

def otsu_threshold(
    image,
    hsv_mask=None
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    if hsv_mask is not None:

        gray = cv2.bitwise_and(
            gray,
            gray,
            mask=hsv_mask
        )

    threshold, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return threshold, mask


# ============================================================
# HSV FEATURES FOR RIPENESS
# ============================================================

def get_ripeness_features(
    image,
    mask
):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    pixels = hsv[
        mask > 0
    ]

    if len(pixels) == 0:

        return (
            0.0,
            0.0,
            0.0
        )

    H = pixels[:, 0].astype(float)
    S = pixels[:, 1].astype(float)
    V = pixels[:, 2].astype(float)

    return (
        np.mean(H),
        np.mean(S),
        np.mean(V)
    )


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

    threshold, mask = otsu_threshold(
        image
    )

    h_mean, s_mean, v_mean = (
        get_ripeness_features(
            image,
            mask
        )
    )

    ripeness = determine_ripeness(
        h_mean,
        s_mean,
        v_mean
    )

    fruit_pixels = np.count_nonzero(
        mask
    )

    fruit_area = (
        fruit_pixels /
        mask.size
    ) * 100

    return {
        "image": image,
        "mask": mask,
        "threshold": threshold,
        "H_mean": h_mean,
        "S_mean": s_mean,
        "V_mean": v_mean,
        "fruit_area_percentage":
            fruit_area,
        "ripeness":
            ripeness
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n======================================")
    print("OTSU THRESHOLDING")
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

        results.append({
            "image":
                os.path.basename(image_path),

            "otsu_threshold":
                result["threshold"],

            "H_mean":
                result["H_mean"],

            "S_mean":
                result["S_mean"],

            "V_mean":
                result["V_mean"],

            "fruit_area_percentage":
                result["fruit_area_percentage"],

            "ripeness":
                result["ripeness"]
        })

    df = pd.DataFrame(
        results
    )

    csv_path = os.path.join(
        RESULTS_PATH,
        "otsu_results.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print("\n======================================")
    print("OTSU RESULTS")
    print("======================================")

    print(
        f"Images processed: {len(df)}"
    )

    print(
        f"Average Otsu Threshold: "
        f"{df['otsu_threshold'].mean():.2f}"
    )

    print(
        f"Average Fruit Area: "
        f"{df['fruit_area_percentage'].mean():.2f}%"
    )

    print("\nRipeness:")

    print(
        df["ripeness"].value_counts()
    )

    print(
        f"\nCSV saved to: {csv_path}"
    )

    # Display first result

    result = process_image(
        image_paths[0]
    )

    if result is not None:

        image_rgb = cv2.cvtColor(
            result["image"],
            cv2.COLOR_BGR2RGB
        )

        plt.figure(
            figsize=(12, 6)
        )

        plt.subplot(1, 3, 1)

        plt.imshow(
            image_rgb
        )

        plt.title(
            "Original Image"
        )

        plt.axis("off")

        plt.subplot(1, 3, 2)

        plt.imshow(
            result["mask"],
            cmap="gray"
        )

        plt.title(
            "Otsu Threshold"
        )

        plt.axis("off")

        plt.subplot(1, 3, 3)

        plt.axis("off")

        text = (
            "OTSU RESULTS\n\n"
            f"Threshold: "
            f"{result['threshold']:.2f}\n\n"
            f"H Mean: "
            f"{result['H_mean']:.2f}\n"
            f"S Mean: "
            f"{result['S_mean']:.2f}\n"
            f"V Mean: "
            f"{result['V_mean']:.2f}\n\n"
            f"Ripeness: "
            f"{result['ripeness']}"
        )

        plt.text(
            0.05,
            0.95,
            text,
            verticalalignment="top",
            fontsize=12
        )

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()