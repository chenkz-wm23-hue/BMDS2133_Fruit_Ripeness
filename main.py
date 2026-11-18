import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hsv import (
    hsv_segmentation,
    calculate_hsv_mean,
    determine_ripeness
)

from otsu import (
    otsu_threshold
)

from watershed import (
    watershed_segmentation
)

from colour_moments import (
    calculate_colour_moments
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "dataset"

RESULTS_PATH = "results/integrated"

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
# SURFACE QUALITY
# ============================================================

def calculate_surface_quality(
    image,
    fruit_mask
):

    fruit_pixels = np.count_nonzero(
        fruit_mask
    )

    total_pixels = fruit_mask.size

    if total_pixels == 0:

        return 0.0

    return (
        fruit_pixels /
        total_pixels
    ) * 100


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path
):

    print(
        "\n======================================"
    )

    print(
        f"Processing: "
        f"{os.path.basename(image_path)}"
    )

    print(
        "======================================"
    )


    # ========================================================
    # STEP 1
    # INPUT RGB IMAGE
    # ========================================================

    image = cv2.imread(
        image_path
    )

    if image is None:

        return None


    # ========================================================
    # STEP 2
    # RGB → HSV
    # ========================================================

    hsv_image, hsv_mask = (
        hsv_segmentation(
            image
        )
    )


    # ========================================================
    # STEP 3
    # HSV THRESHOLDING
    #
    # Included inside hsv_segmentation()
    # ========================================================


    # ========================================================
    # STEP 4
    # MORPHOLOGICAL OPENING
    #
    # Included inside hsv_segmentation()
    # ========================================================


    # ========================================================
    # STEP 5
    # MORPHOLOGICAL CLOSING
    #
    # Included inside hsv_segmentation()
    # ========================================================


    # ========================================================
    # STEP 6
    # FINAL HSV MASK
    # ========================================================


    # ========================================================
    # STEP 7
    # HSV SEGMENTED IMAGE
    # ========================================================

    hsv_segmented = cv2.bitwise_and(
        image,
        image,
        mask=hsv_mask
    )


    # ========================================================
    # STEP 8
    # OTSU THRESHOLDING
    # ========================================================

    otsu_value, otsu_mask = (
        otsu_threshold(
            image,
            hsv_mask
        )
    )


    # ========================================================
    # STEP 9
    # WATERSHED
    # ========================================================

    markers, watershed_mask, object_count = (
        watershed_segmentation(
            image,
            otsu_mask
        )
    )


    # ========================================================
    # STEP 10
    # COLOUR MOMENTS
    # ========================================================

    colour_features = (
        calculate_colour_moments(
            image,
            watershed_mask
        )
    )


    # ========================================================
    # STEP 11
    # RIPENESS
    # ========================================================

    ripeness = determine_ripeness(
        colour_features["H_mean"],
        colour_features["S_mean"],
        colour_features["V_mean"]
    )


    # ========================================================
    # STEP 12
    # SURFACE QUALITY
    # ========================================================

    surface_area = (
        calculate_surface_quality(
            image,
            watershed_mask
        )
    )


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "image":
            os.path.basename(
                image_path
            ),

        "otsu_threshold":
            otsu_value,

        "object_count":
            object_count,

        "surface_area_percentage":
            surface_area,

        "ripeness":
            ripeness
    }

    result.update(
        colour_features
    )


    return {

        "result":
            result,

        "image":
            image,

        "hsv":
            hsv_image,

        "hsv_mask":
            hsv_mask,

        "hsv_segmented":
            hsv_segmented,

        "otsu_mask":
            otsu_mask,

        "markers":
            markers,

        "watershed_mask":
            watershed_mask
    }


# ============================================================
# DISPLAY PIPELINE
# ============================================================

def display_pipeline(
    result
):

    image = result["image"]

    hsv = result["hsv"]

    hsv_mask = result["hsv_mask"]

    hsv_segmented = result[
        "hsv_segmented"
    ]

    otsu_mask = result[
        "otsu_mask"
    ]

    markers = result[
        "markers"
    ]

    watershed_mask = result[
        "watershed_mask"
    ]

    features = result[
        "result"
    ]


    # --------------------------------------------------------
    # Convert images
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    hsv_rgb = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB
    )

    hsv_segmented_rgb = cv2.cvtColor(
        hsv_segmented,
        cv2.COLOR_BGR2RGB
    )


    # Watershed boundary
    boundary = image.copy()

    boundary[
        markers == -1
    ] = [0, 255, 0]

    boundary_rgb = cv2.cvtColor(
        boundary,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    plt.figure(
        figsize=(16, 10)
    )


    # 1
    plt.subplot(2, 4, 1)

    plt.imshow(
        image_rgb
    )

    plt.title(
        "Input RGB Image"
    )

    plt.axis("off")


    # 2
    plt.subplot(2, 4, 2)

    plt.imshow(
        hsv_rgb
    )

    plt.title(
        "RGB → HSV"
    )

    plt.axis("off")


    # 3
    plt.subplot(2, 4, 3)

    plt.imshow(
        hsv_mask,
        cmap="gray"
    )

    plt.title(
        "Final HSV Mask"
    )

    plt.axis("off")


    # 4
    plt.subplot(2, 4, 4)

    plt.imshow(
        hsv_segmented_rgb
    )

    plt.title(
        "HSV Segmented Image"
    )

    plt.axis("off")


    # 5
    plt.subplot(2, 4, 5)

    plt.imshow(
        otsu_mask,
        cmap="gray"
    )

    plt.title(
        "Otsu Threshold"
    )

    plt.axis("off")


    # 6
    plt.subplot(2, 4, 6)

    plt.imshow(
        watershed_mask,
        cmap="gray"
    )

    plt.title(
        "Watershed Mask"
    )

    plt.axis("off")


    # 7
    plt.subplot(2, 4, 7)

    plt.imshow(
        boundary_rgb
    )

    plt.title(
        "Watershed Boundaries"
    )

    plt.axis("off")


    # 8
    plt.subplot(2, 4, 8)

    plt.axis("off")


    text = (
        "FINAL RESULTS\n\n"

        "COLOUR MOMENTS\n"

        f"R Mean: "
        f"{features['R_mean']:.2f}\n"

        f"G Mean: "
        f"{features['G_mean']:.2f}\n"

        f"B Mean: "
        f"{features['B_mean']:.2f}\n\n"

        f"H Mean: "
        f"{features['H_mean']:.2f}\n"

        f"S Mean: "
        f"{features['S_mean']:.2f}\n"

        f"V Mean: "
        f"{features['V_mean']:.2f}\n\n"

        f"Objects: "
        f"{features['object_count']}\n"

        f"Surface Area: "
        f"{features['surface_area_percentage']:.2f}%\n\n"

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print(
        "================================================"
    )

    print(
        " BMDS2133 FRUIT RIPENESS ASSESSMENT"
    )

    print(
        " MODE A - INTEGRATED IMAGE PROCESSING"
    )

    print(
        "================================================"
    )


    print(
        "\nPROCESS FLOW:"
    )

    print(
        "RGB Image"
    )

    print(
        "↓"
    )

    print(
        "RGB → HSV"
    )

    print(
        "↓"
    )

    print(
        "HSV Thresholding"
    )

    print(
        "↓"
    )

    print(
        "Morphological Opening"
    )

    print(
        "↓"
    )

    print(
        "Morphological Closing"
    )

    print(
        "↓"
    )

    print(
        "Final HSV Mask"
    )

    print(
        "↓"
    )

    print(
        "HSV Segmented Image"
    )

    print(
        "↓"
    )

    print(
        "Otsu Thresholding"
    )

    print(
        "↓"
    )

    print(
        "Watershed"
    )

    print(
        "↓"
    )

    print(
        "Colour Moments"
    )

    print(
        "↓"
    )

    print(
        "Ripeness + Surface Quality"
    )


    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_paths = find_images(
        DATASET_PATH
    )

    print(
        f"\nImages found: "
        f"{len(image_paths)}"
    )


    if len(image_paths) == 0:

        print(
            "\nERROR:"
        )

        print(
            "No images found inside dataset/"
        )

        return


    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    results = []


    for image_path in image_paths:

        processed = process_image(
            image_path
        )

        if processed is None:
            continue

        results.append(
            processed["result"]
        )


    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        RESULTS_PATH,
        "integrated_results.csv"
    )


    df.to_csv(
        csv_path,
        index=False
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print(
        "================================================"
    )

    print(
        " INTEGRATED RESULTS"
    )

    print(
        "================================================"
    )


    print(
        f"Images processed: "
        f"{len(df)}"
    )


    if len(df) > 0:

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

        print(
            f"\nAverage Surface Area: "
            f"{df['surface_area_percentage'].mean():.2f}%"
        )

        print(
            "\nRipeness:"
        )

        print(
            df["ripeness"].value_counts()
        )


    print(
        f"\nResults saved to:"
    )

    print(
        csv_path
    )


    # --------------------------------------------------------
    # Display first image
    # --------------------------------------------------------

    first = process_image(
        image_paths[0]
    )

    if first is not None:

        display_pipeline(
            first
        )


if __name__ == "__main__":
    main()