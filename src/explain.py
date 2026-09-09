from __future__ import annotations

from datetime import datetime
import html
import re
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


DISPLAY_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if",
    "then", "than", "so", "to", "of", "in", "on",
    "at", "for", "from", "with", "without", "by",
    "as", "is", "are", "was", "were", "be", "been",
    "being", "am", "i", "me", "my", "mine", "we",
    "our", "ours", "you", "your", "yours", "he",
    "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "theirs", "this", "that",
    "these", "those", "do", "does", "did", "doing",
    "have", "has", "had", "having", "can", "could",
    "will", "would", "should", "may", "might", "must",
    "not", "no", "very", "just", "really", "more",
    "most", "some", "any", "all", "about", "into",
    "out", "up", "down", "over", "under", "again",
    "also", "there", "here", "when", "where", "why",
    "how", "what", "which", "who", "whom", "because",
    "while",
}


def explain_text(
    predictor,
    text: str,
    dt: datetime,
    num_features: int = 12,
    num_samples: int = 800,
):
    """
    Generate a LIME explanation for the text while
    keeping posting time fixed.
    """

    from lime.lime_text import LimeTextExplainer

    class_names = predictor.class_names

    explainer = LimeTextExplainer(
        class_names=class_names,
        random_state=42,
    )

    def classifier_fn(texts):
        return predictor.predict_proba(
            texts,
            dt=dt,
            batch_size=32,
        )

    probs = classifier_fn([text])[0]

    predicted_label = int(
        probs.argmax()
    )

    lime_exp = explainer.explain_instance(
        text,
        classifier_fn,
        labels=[predicted_label],
        num_features=num_features,
        num_samples=num_samples,
    )

    weights = lime_exp.as_list(
        label=predicted_label
    )

    temporal = predictor.temporal_effect(
        text,
        dt,
    )

    return {
        "predicted_label": predicted_label,
        "probabilities": probs,
        "weights": weights,
        "temporal": temporal,
        "num_samples": int(num_samples),
    }


def _normalized_term(
    term: str,
) -> str:

    return re.sub(
        r"[^a-z0-9']+",
        " ",
        term.lower(),
    ).strip()


def meaningful_weights(
    weights: Iterable[tuple[str, float]],
    max_items: int = 6,
    min_abs_weight: float = 0.003,
) -> list[tuple[str, float]]:
    """
    Remove common low-information words from the
    human-readable summary.

    This does NOT modify LIME itself.
    """

    kept = []

    for term, weight in sorted(
        weights,
        key=lambda x: abs(x[1]),
        reverse=True,
    ):

        normalized = _normalized_term(term)

        if not normalized:
            continue

        if normalized in DISPLAY_STOPWORDS:
            continue

        if abs(float(weight)) < min_abs_weight:
            continue

        kept.append(
            (term, float(weight))
        )

        if len(kept) >= max_items:
            break

    return kept


def split_supporting_opposing(
    weights: Iterable[tuple[str, float]],
    max_each: int = 5,
):

    meaningful = meaningful_weights(
        weights,
        max_items=max_each * 3,
    )

    support = sorted(
        [
            (term, weight)
            for term, weight in meaningful
            if weight > 0
        ],
        key=lambda x: x[1],
        reverse=True,
    )[:max_each]

    oppose = sorted(
        [
            (term, weight)
            for term, weight in meaningful
            if weight < 0
        ],
        key=lambda x: x[1],
    )[:max_each]

    return support, oppose


def build_plain_language_summary(
    class_name: str,
    confidence: float,
    weights: Iterable[tuple[str, float]],
) -> str:

    support, oppose = split_supporting_opposing(
        weights,
        max_each=4,
    )

    def terms(items):
        return ", ".join(
            f'“{term}”'
            for term, _ in items
        )

    parts = [
        (
            f"The model classified this post as "
            f"**{class_name}** with "
            f"**{confidence * 100:.1f}% confidence**."
        ),
        (
            "LIME estimated which words or phrases "
            "locally influenced this specific prediction "
            "while the posting time was kept fixed."
        ),
    ]

    if support:
        parts.append(
            "The strongest informative text features "
            "supporting the displayed class were "
            f"{terms(support)}."
        )
    else:
        parts.append(
            "No strong informative non-stopword feature "
            "clearly supported the displayed class."
        )

    if oppose:
        parts.append(
            "Features pushing the model away from the "
            f"displayed class included {terms(oppose)}."
        )
    else:
        parts.append(
            "No strong informative feature clearly "
            "pushed against the displayed class."
        )

    parts.append(
        "These contributions explain the model's local "
        "behaviour, not a person's psychological state "
        "or a clinical diagnosis."
    )

    return " ".join(parts)


def highlight_text_html(
    text: str,
    weights: Iterable[tuple[str, float]],
) -> str:

    by_term = {}

    for term, weight in weights:

        normalized = _normalized_term(term)

        if not normalized:
            continue

        current = by_term.get(normalized)

        if (
            current is None
            or abs(float(weight))
            > abs(current[1])
        ):
            by_term[normalized] = (
                term,
                float(weight),
            )

    terms = sorted(
        (
            value[0]
            for value in by_term.values()
        ),
        key=len,
        reverse=True,
    )

    if not terms:
        return (
            f"<div>{html.escape(text)}</div>"
        )

    pattern = re.compile(
        r"(?i)(?<!\w)("
        + "|".join(
            re.escape(term)
            for term in terms
        )
        + r")(?!\w)"
    )

    output = []
    cursor = 0

    for match in pattern.finditer(text):

        output.append(
            html.escape(
                text[cursor:match.start()]
            )
        )

        matched = match.group(0)

        normalized = _normalized_term(
            matched
        )

        _, weight = by_term.get(
            normalized,
            (matched, 0.0),
        )

        if weight >= 0:

            style = (
                "background:#1f6f43;"
                "color:#ffffff;"
                "padding:2px 4px;"
                "border-radius:4px;"
                "font-weight:600;"
            )

            title = (
                "Supports displayed class "
                f"(LIME {weight:+.4f})"
            )

        else:

            style = (
                "background:#8b2f3c;"
                "color:#ffffff;"
                "padding:2px 4px;"
                "border-radius:4px;"
                "font-weight:600;"
            )

            title = (
                "Pushes against displayed class "
                f"(LIME {weight:+.4f})"
            )

        output.append(
            f'<span style="{style}" '
            f'title="{html.escape(title)}">'
            f'{html.escape(matched)}</span>'
        )

        cursor = match.end()

    output.append(
        html.escape(text[cursor:])
    )

    return (
        '<div style="'
        'line-height:1.9;'
        'font-size:1.02rem;'
        'padding:14px;'
        'border:1px solid #444;'
        'border-radius:8px;'
        'white-space:pre-wrap;">'
        + "".join(output)
        + "</div>"
    )


def lime_dataframe(
    weights: Iterable[tuple[str, float]],
) -> pd.DataFrame:

    rows = [
        {
            "Word / phrase": term,
            "LIME weight": float(weight),
            "Direction": (
                "Supports displayed class"
                if weight > 0
                else "Pushes against displayed class"
            ),
        }
        for term, weight in sorted(
            weights,
            key=lambda x: abs(x[1]),
            reverse=True,
        )
    ]

    return pd.DataFrame(rows)


def contribution_figure(
    weights: Iterable[tuple[str, float]],
    max_items: int = 10,
):
    """
    Create horizontal LIME contribution chart.
    """

    items = meaningful_weights(
        weights,
        max_items=max_items,
    )

    if not items:

        items = sorted(
            weights,
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:max_items]

    items = sorted(
        items,
        key=lambda x: x[1],
    )

    labels = [
        term
        for term, _ in items
    ]

    values = [
        float(weight)
        for _, weight in items
    ]

    colors = [
        "#2e8b57"
        if value >= 0
        else "#b24a5a"
        for value in values
    ]

    height = max(
        3.0,
        0.42 * len(items) + 1.2,
    )

    fig, ax = plt.subplots(
        figsize=(8.5, height)
    )

    ax.barh(
        labels,
        values,
        color=colors,
    )

    ax.axvline(
        0,
        color="grey",
        linewidth=1,
    )

    ax.set_xlabel(
        "LIME weight for displayed class"
    )

    ax.set_ylabel("")

    ax.set_title(
        "Local text contributions"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    return fig


def temporal_effect_text(
    temporal: dict,
    class_name: str,
) -> str:

    delta_pp = (
        float(
            temporal[
                "displayed_class_delta"
            ]
        )
        * 100.0
    )

    abs_pp = abs(delta_pp)

    if abs_pp < 0.5:
        magnitude = "negligible"

    elif abs_pp < 2.0:
        magnitude = "small"

    elif abs_pp < 5.0:
        magnitude = "moderate"

    else:
        magnitude = "substantial"

    if delta_pp > 0:

        direction = (
            "increased the probability of "
            f"**{class_name}**"
        )

    elif delta_pp < 0:

        direction = (
            "decreased the probability of "
            f"**{class_name}**"
        )

    else:

        direction = (
            "did not change the probability of "
            f"**{class_name}**"
        )

    return (
        "With the post text held unchanged, "
        f"the selected posting time "
        f"({temporal['timestamp_utc']}) "
        f"{direction} by "
        f"**{delta_pp:+.2f} percentage points** "
        "compared with the model's average "
        "temporal context. "
        f"This is a **{magnitude} local temporal effect**. "
        "It is a model-sensitivity estimate, "
        "not a causal or clinical claim."
    )