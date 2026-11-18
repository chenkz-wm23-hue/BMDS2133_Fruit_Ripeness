# ============================================================
# BMDS2133 FRUIT RIPENESS ASSESSMENT
# HSV COLOUR SEGMENTATION
#
# Individual Algorithm:
# RGB Image
#     ↓
# RGB → HSV
#     ↓
# HSV Thresholding
#     ↓
# Morphological Opening
#     ↓
# Morphological Closing
#     ↓
# Final HSV Mask
#     ↓
# HSV Segmented Image
#     ↓
# Mean H, S, V
#     ↓
# Ripeness Assessment
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 2. SETTINGS
# ============================================================

DATASET_PATH = "dataset"

RESULTS_PATH = "results/hsv"

os.makedirs(
    RESULTS_PATH,
    exist_ok=True
)


# ============================================================
# 3. FIND IMAGES
# ============================================================

def find_images(folder):

    image_paths = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                )
            ):

                image_paths.append(
                    os.path.join(
                        root,
                        file
                    )
                )

    return sorted(image_paths)


# ============================================================
# 4. RIPENESS ASSESSMENT
# ============================================================

def determine_ripeness(
    h_mean,
    s_mean,
    v_mean
):

    """
    Rule-based ripeness assessment.

    IMPORTANT:
    These thresholds are initial experimental
    thresholds and should be adjusted based
    on the dataset and ground-truth results.
    """

    if (
        h_mean < 15
        and
        s_mean > 100
    ):

        return "Ripe"

    elif (
        h_mean < 30
        and
        s_mean > 70
    ):

        return "Overripe"

    else:

        return "Unripe"


# ============================================================
# 5. HSV COLOUR SEGMENTATION
# ============================================================

def hsv_segmentation(image):

    """
    Perform:

    BGR → HSV
    HSV thresholding
    Morphological opening
    Morphological closing
    """

    # --------------------------------------------------------
    # Convert image to HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    # --------------------------------------------------------
    # HSV threshold values
    # --------------------------------------------------------

    lower_hsv = np.array(
        [
            0,
            40,
            30
        ],
        dtype=np.uint8
    )

    upper_hsv = np.array(
        [
            179,
            255,
            255
        ],
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # HSV THRESHOLDING
    # --------------------------------------------------------

    mask = cv2.inRange(
        hsv,
        lower_hsv,
        upper_hsv
    )


    # --------------------------------------------------------
    # Morphological kernel
    # --------------------------------------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8
    )


    # --------------------------------------------------------
    # MORPHOLOGICAL OPENING
    # Removes small noise
    # --------------------------------------------------------

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )


    # --------------------------------------------------------
    # MORPHOLOGICAL CLOSING
    # Fills small holes
    # --------------------------------------------------------

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )


    return hsv, mask


# ============================================================
# 6. CALCULATE HSV MEAN
# ============================================================

def calculate_hsv_mean(
    hsv,
    mask
):

    """
    Calculate the mean Hue, Saturation and Value
    from the segmented fruit region.
    """

    # Select only pixels inside the mask
    pixels = hsv[
        mask > 0
    ]


    # No segmented pixels
    if len(pixels) == 0:

        return (
            0.0,
            0.0,
            0.0
        )


    # Separate HSV channels
    H = pixels[:, 0]
    S = pixels[:, 1]
    V = pixels[:, 2]


    # Calculate means
    h_mean = np.mean(H)

    s_mean = np.mean(S)

    v_mean = np.mean(V)


    return (
        h_mean,
        s_mean,
        v_mean
    )


# ============================================================
# 7. PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path
):

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )


    if image is None:

        print(
            f"Cannot read image: {image_path}"
        )

        return None


    # --------------------------------------------------------
    # HSV SEGMENTATION
    # --------------------------------------------------------

    hsv, mask = hsv_segmentation(
        image
    )


    # --------------------------------------------------------
    # CALCULATE HSV MEANS
    # --------------------------------------------------------

    h_mean, s_mean, v_mean = (
        calculate_hsv_mean(
            hsv,
            mask
        )
    )


    # --------------------------------------------------------
    # RIPENESS
    # --------------------------------------------------------

    ripeness = determine_ripeness(
        h_mean,
        s_mean,
        v_mean
    )


    # --------------------------------------------------------
    # HSV SEGMENTED IMAGE
    # --------------------------------------------------------

    segmented_image = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )


    # --------------------------------------------------------
    # SEGMENTATION AREA
    # --------------------------------------------------------

    total_pixels = (
        mask.shape[0]
        *
        mask.shape[1]
    )

    fruit_pixels = np.count_nonzero(
        mask
    )


    if total_pixels > 0:

        fruit_area_percentage = (
            fruit_pixels
            /
            total_pixels
        ) * 100

    else:

        fruit_area_percentage = 0


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {

        "image":
            os.path.basename(
                image_path
            ),

        "H_mean":
            h_mean,

        "S_mean":
            s_mean,

        "V_mean":
            v_mean,

        "fruit_area_percentage":
            fruit_area_percentage,

        "ripeness":
            ripeness
    }


    return (
        result,
        image,
        hsv,
        mask,
        segmented_image
    )


# ============================================================
# 8. DISPLAY ONE IMAGE
# ============================================================

def display_image(
    image,
    hsv,
    mask,
    segmented_image,
    result
):

    # --------------------------------------------------------
    # Convert images for display
    # --------------------------------------------------------

    original_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # HSV visualisation
    hsv_display = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB
    )


    segmented_rgb = cv2.cvtColor(
        segmented_image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(16, 8)
    )


    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    plt.subplot(
        2,
        3,
        1
    )

    plt.imshow(
        original_rgb
    )

    plt.title(
        "Original Image"
    )

    plt.axis(
        "off"
    )


    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    plt.subplot(
        2,
        3,
        2
    )

    plt.imshow(
        hsv_display
    )

    plt.title(
        "HSV Image"
    )

    plt.axis(
        "off"
    )


    # --------------------------------------------------------
    # HSV Mask
    # --------------------------------------------------------

    plt.subplot(
        2,
        3,
        3
    )

    plt.imshow(
        mask,
        cmap="gray"
    )

    plt.title(
        "HSV Segmentation"
    )

    plt.axis(
        "off"
    )


    # --------------------------------------------------------
    # Segmented Fruit
    # --------------------------------------------------------

    plt.subplot(
        2,
        3,
        4
    )

    plt.imshow(
        segmented_rgb
    )

    plt.title(
        "HSV Segmented Image"
    )

    plt.axis(
        "off"
    )


    # --------------------------------------------------------
    # HSV values
    # --------------------------------------------------------

    plt.subplot(
        2,
        3,
        5
    )

    plt.axis(
        "off"
    )


    hsv_text = (
        "HSV RESULTS\n\n"

        f"Mean H: "
        f"{result['H_mean']:.2f}\n"

        f"Mean S: "
        f"{result['S_mean']:.2f}\n"

        f"Mean V: "
        f"{result['V_mean']:.2f}\n\n"

        f"Fruit Area: "
        f"{result['fruit_area_percentage']:.2f}%\n\n"

        f"Ripeness: "
        f"{result['ripeness']}"
    )


    plt.text(
        0.05,
        0.95,
        hsv_text,
        verticalalignment="top",
        fontsize=12
    )


    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    plt.subplot(
        2,
        3,
        6
    )

    plt.axis(
        "off"
    )


    pipeline_text = (
        "HSV PIPELINE\n\n"

        "Input RGB Image\n"
        "↓\n"
        "RGB → HSV\n"
        "↓\n"
        "HSV Thresholding\n"
        "↓\n"
        "Morphological Opening\n"
        "↓\n"
        "Morphological Closing\n"
        "↓\n"
        "Final HSV Mask\n"
        "↓\n"
        "Ripeness Assessment"
    )


    plt.text(
        0.05,
        0.95,
        pipeline_text,
        verticalalignment="top",
        fontsize=11
    )


    plt.tight_layout()

    plt.show()


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    print(
        "\n=============================================="
    )

    print(
        " HSV COLOUR SEGMENTATION"
    )

    print(
        " FRUIT RIPENESS ASSESSMENT"
    )

    print(
        "=============================================="
    )


    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_paths = find_images(
        DATASET_PATH
    )


    print(
        f"\nImages found: {len(image_paths)}"
    )


    if len(image_paths) == 0:

        print(
            "\nERROR:"
        )

        print(
            "No images found in the dataset folder."
        )

        return


    # --------------------------------------------------------
    # Process all images
    # --------------------------------------------------------

    results = []


    for index, image_path in enumerate(
        image_paths
    ):

        processed = process_image(
            image_path
        )


        if processed is None:

            continue


        result = processed[0]

        results.append(
            result
        )


        print(
            f"Processed "
            f"{index + 1}/"
            f"{len(image_paths)}: "
            f"{result['image']}"
        )


    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_csv = os.path.join(
        RESULTS_PATH,
        "hsv_results.csv"
    )


    df.to_csv(
        output_csv,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n=============================================="
    )

    print(
        "HSV EXPERIMENT COMPLETED"
    )

    print(
        "=============================================="
    )


    print(
        f"Images processed: {len(df)}"
    )


    if len(df) > 0:

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
            f"\nAverage Fruit Area: "
            f"{df['fruit_area_percentage'].mean():.2f}%"
        )


        print(
            "\nRipeness Results:"
        )

        print(
            df["ripeness"].value_counts()
        )


    print(
        f"\nCSV saved to:"
    )

    print(
        output_csv
    )


    # ========================================================
    # DISPLAY FIRST IMAGE
    # ========================================================

    first_result = process_image(
        image_paths[0]
    )


    if first_result is not None:

        (
            result,
            image,
            hsv,
            mask,
            segmented_image
        ) = first_result


        print(
            "\n=============================================="
        )

        print(
            "FIRST IMAGE RESULT"
        )

        print(
            "=============================================="
        )


        print(
            f"Mean Hue: "
            f"{result['H_mean']:.2f}"
        )

        print(
            f"Mean Saturation: "
            f"{result['S_mean']:.2f}"
        )

        print(
            f"Mean Value: "
            f"{result['V_mean']:.2f}"
        )

        print(
            f"Fruit Area: "
            f"{result['fruit_area_percentage']:.2f}%"
        )

        print(
            f"\nRipeness: "
            f"{result['ripeness']}"
        )


        display_image(
            image,
            hsv,
            mask,
            segmented_image,
            result
        )


# ============================================================
# RUN HSV FILE DIRECTLY
# ============================================================

if __name__ == "__main__":

    main()