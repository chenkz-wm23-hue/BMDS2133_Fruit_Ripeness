import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATASET_PATH = "dataset"
RESULTS_PATH = "results/watershed"

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
# WATERSHED
# ============================================================

def watershed_segmentation(
    image,
    binary_mask
):

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    opening = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    sure_background = cv2.dilate(
        opening,
        kernel,
        iterations=3
    )

    distance = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    if distance.max() == 0:

        return (
            np.zeros_like(binary_mask),
            np.zeros_like(binary_mask),
            0
        )

    _, sure_foreground = cv2.threshold(
        distance,
        0.35 * distance.max(),
        255,
        cv2.THRESH_BINARY
    )

    sure_foreground = np.uint8(
        sure_foreground
    )

    unknown = cv2.subtract(
        sure_background,
        sure_foreground
    )

    num_labels, markers = cv2.connectedComponents(
        sure_foreground
    )

    markers = markers + 1

    markers[
        unknown == 255
    ] = 0

    watershed_image = image.copy()

    markers = cv2.watershed(
        watershed_image,
        markers
    )

    fruit_mask = np.zeros(
        binary_mask.shape,
        dtype=np.uint8
    )

    fruit_mask[
        markers > 1
    ] = 255

    object_count = len(
        np.unique(markers[markers > 1])
    )

    return (
        markers,
        fruit_mask,
        object_count
    )


# ============================================================
# RIPENESS FEATURES
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

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    _, initial_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    markers, mask, object_count = (
        watershed_segmentation(
            image,
            initial_mask
        )
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

    fruit_area = (
        np.count_nonzero(mask) /
        mask.size
    ) * 100

    return {
        "image": image,
        "mask": mask,
        "markers": markers,
        "object_count":
            object_count,
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
    print("WATERSHED SEGMENTATION")
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

            "object_count":
                result["object_count"],

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
        "watershed_results.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print("\n======================================")
    print("WATERSHED RESULTS")
    print("======================================")

    print(
        f"Images processed: {len(df)}"
    )

    print(
        f"Average detected objects: "
        f"{df['object_count'].mean():.2f}"
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

    result = process_image(
        image_paths[0]
    )

    if result is not None:

        boundary = result["image"].copy()

        boundary[
            result["markers"] == -1
        ] = [0, 255, 0]

        boundary_rgb = cv2.cvtColor(
            boundary,
            cv2.COLOR_BGR2RGB
        )

        plt.figure(
            figsize=(12, 6)
        )

        plt.subplot(1, 3, 1)

        plt.imshow(
            cv2.cvtColor(
                result["image"],
                cv2.COLOR_BGR2RGB
            )
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
            "Watershed Mask"
        )

        plt.axis("off")

        plt.subplot(1, 3, 3)

        plt.imshow(
            boundary_rgb
        )

        plt.title(
            "Watershed Boundaries"
        )

        plt.axis("off")

        plt.tight_layout()
        plt.show()

        print(
            f"\nRipeness: "
            f"{result['ripeness']}"
        )


if __name__ == "__main__":
    main()