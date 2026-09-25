import numpy as np


def extract_spatial_features(
    words,
    image_width,
    image_height,
    density_radius=0.12
):
    """
    Extract spatial and spatial-density features from
    OCR-detected words.

    Features include:
        1. Normalized X position
        2. Normalized Y position
        3. Normalized width
        4. Normalized height
        5. Aspect ratio
        6. OCR confidence
        7. Nearest-neighbor distance
        8. Average neighbor distance
        9. Local neighbor count
        10. Local spatial density
        11. Normalized text area
        12. Relative vertical position

    These features are used by Local Outlier Factor (LOF)
    to identify unusual text positions, sizes, spacing,
    and local spatial arrangements.
    """

    # --------------------------------------------------------
    # Handle empty OCR result
    # --------------------------------------------------------

    if not words:
        return np.empty((0, 12), dtype=np.float32)


    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    image_width = max(float(image_width), 1.0)
    image_height = max(float(image_height), 1.0)


    # --------------------------------------------------------
    # Extract normalized text-center coordinates
    # --------------------------------------------------------

    centers = []

    for word in words:

        x = float(word["x"])
        y = float(word["y"])
        width = float(word["width"])
        height = float(word["height"])

        center_x = x + (width / 2.0)
        center_y = y + (height / 2.0)

        normalized_x = center_x / image_width
        normalized_y = center_y / image_height

        centers.append([
            normalized_x,
            normalized_y
        ])


    centers = np.array(
        centers,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Calculate pairwise spatial distances
    # --------------------------------------------------------

    difference = (
        centers[:, np.newaxis, :]
        - centers[np.newaxis, :, :]
    )

    distances = np.sqrt(
        np.sum(
            difference ** 2,
            axis=2
        )
    )


    # Ignore distance from a word to itself
    np.fill_diagonal(
        distances,
        np.inf
    )


    # --------------------------------------------------------
    # Spatial-density parameters
    # --------------------------------------------------------

    radius = float(density_radius)

    # Area of circular neighborhood
    neighborhood_area = (
        np.pi * radius * radius
    )


    # --------------------------------------------------------
    # Create feature vectors
    # --------------------------------------------------------

    features = []


    for index, word in enumerate(words):

        x = float(word["x"])
        y = float(word["y"])
        width = float(word["width"])
        height = float(word["height"])

        confidence = float(
            word["confidence"]
        )


        # ----------------------------------------------------
        # Basic spatial features
        # ----------------------------------------------------

        center_x = x + (width / 2.0)
        center_y = y + (height / 2.0)

        normalized_x = (
            center_x / image_width
        )

        normalized_y = (
            center_y / image_height
        )

        normalized_width = (
            width / image_width
        )

        normalized_height = (
            height / image_height
        )

        aspect_ratio = (
            width / max(height, 1.0)
        )


        # ----------------------------------------------------
        # OCR confidence
        # ----------------------------------------------------

        normalized_confidence = (
            confidence / 100.0
        )


        # ----------------------------------------------------
        # Distances to other OCR words
        # ----------------------------------------------------

        word_distances = distances[index]

        nearest_distance = float(
            np.min(word_distances)
        )


        finite_distances = (
            word_distances[
                np.isfinite(word_distances)
            ]
        )


        if len(finite_distances) > 0:

            average_neighbor_distance = float(
                np.mean(finite_distances)
            )

        else:

            average_neighbor_distance = 0.0


        # ----------------------------------------------------
        # Local neighborhood
        # ----------------------------------------------------

        nearby_mask = (
            word_distances <= radius
        )


        neighbor_count = int(
            np.sum(nearby_mask)
        )


        # ----------------------------------------------------
        # Local spatial density
        #
        # Density = number of nearby OCR words /
        #           neighborhood area
        # ----------------------------------------------------

        local_density = (
            neighbor_count
            / max(neighborhood_area, 1e-8)
        )


        # Normalize density so its scale is manageable
        normalized_density = (
            local_density
            / 100.0
        )


        # ----------------------------------------------------
        # Text area
        # ----------------------------------------------------

        normalized_text_area = (
            normalized_width
            * normalized_height
        )


        # ----------------------------------------------------
        # Relative vertical position
        # ----------------------------------------------------

        relative_vertical_position = (
            normalized_y
        )


        # ----------------------------------------------------
        # Final feature vector
        # ----------------------------------------------------

        features.append([

            # Basic spatial features
            normalized_x,
            normalized_y,
            normalized_width,
            normalized_height,
            aspect_ratio,

            # OCR quality
            normalized_confidence,

            # Spatial relationship features
            nearest_distance,
            average_neighbor_distance,

            # Spatial density features
            neighbor_count,
            normalized_density,

            # Text geometry
            normalized_text_area,

            # Vertical alignment
            relative_vertical_position
        ])


    return np.array(
        features,
        dtype=np.float32
    )