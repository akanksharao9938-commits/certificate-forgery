import numpy as np

from sklearn.neighbors import LocalOutlierFactor


def detect_lof_anomalies(
    features,
    n_neighbors=10
):
    """
    Detect local outliers using Local Outlier Factor.

    LOF identifies observations whose local density is
    substantially different from that of their neighbors.
    """

    # --------------------------------------------------------
    # Handle empty feature matrix
    # --------------------------------------------------------

    if features is None or len(features) == 0:

        return {

            "labels": [],

            "scores": [],

            "message":
                "No features available for LOF analysis."
        }


    # --------------------------------------------------------
    # Not enough observations
    # --------------------------------------------------------

    if len(features) < 3:

        return {

            "labels":
                [1] * len(features),

            "scores":
                [1.0] * len(features),

            "message":
                "Not enough OCR words for LOF analysis."
        }


    # --------------------------------------------------------
    # Make sure feature values are numeric
    # --------------------------------------------------------

    features = np.asarray(
        features,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    # --------------------------------------------------------
    # Select valid number of neighbors
    # --------------------------------------------------------

    actual_neighbors = min(
        n_neighbors,
        len(features) - 1
    )


    # --------------------------------------------------------
    # LOF model
    # --------------------------------------------------------

    model = LocalOutlierFactor(

        n_neighbors=actual_neighbors,

        contamination="auto"
    )


    # --------------------------------------------------------
    # Fit LOF
    # --------------------------------------------------------

    labels = model.fit_predict(
        features
    )


    # --------------------------------------------------------
    # LOF scores
    #
    # sklearn returns negative_outlier_factor_.
    # Multiplying by -1 makes larger values indicate
    # stronger outlier behavior.
    # --------------------------------------------------------

    scores = (
        -model.negative_outlier_factor_
    )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "labels":
            labels.tolist(),

        "scores":
            scores.tolist(),

        "message":
            "LOF spatial-density analysis completed."
    }