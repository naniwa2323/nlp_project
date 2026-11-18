
"""Member 3: AI-generated text boundary detection.

Uses Member 2's classifier and the existing segmentation algorithm.
"""

from segmenter import segment_article as _segment_article


def segment_article(
    article: str,
    model,
    threshold: float = 0.5,
    min_len: int = 100,
) -> list[dict]:
    """
    Divide an article into Human (0) and AI (1) regions.

    Returns:
        [
            {
                "start": int,
                "end": int,
                "label": 0 or 1,
                "score": float,
            },
            ...
        ]

    Regions cover the entire article without gaps.
    """
    if not isinstance(article, str):
        raise TypeError("article must be a string")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    if min_len < 0:
        raise ValueError("min_len must be non-negative")

    return _segment_article(
        article,
        model,
        threshold=threshold,
        min_len=min_len,
    )
