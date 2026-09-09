from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from config import CLASS_NAMES, DATA_DIR, MODEL_DIR, RESULTS_DIR


st.set_page_config(
    page_title="Explainable Depression-Related Content Analysis",
    page_icon="🧠",
    layout="wide",
)

st.title("Explainable Deep Learning for Depression-Related Social Media Analysis")
st.caption("NLP • Temporal Behavioural Analysis • DistilBERT • LIME")
# st.warning(
#     " "
#     ""
# )


@st.cache_resource
def load_predictor():
    from src.inference import DepressionPredictor
    return DepressionPredictor(MODEL_DIR)


def model_is_ready() -> bool:
    required = [
        MODEL_DIR / "model_metadata.json",
        MODEL_DIR / "hybrid_head.pt",
        MODEL_DIR / "temporal_scaler.joblib",
        MODEL_DIR / "transformer",
        MODEL_DIR / "tokenizer",
    ]
    return all(p.exists() for p in required)


def label_name(label: int) -> str:
    return CLASS_NAMES[int(label)]


def load_csv(name: str):
    path = DATA_DIR / name
    return pd.read_csv(path) if path.exists() else None


def temporal_dataset_context(dt: datetime) -> dict | None:
    """Return population-level hour/day context from prepared temporal aggregates."""
    hour = load_csv("temporal_hour.csv")
    dow = load_csv("temporal_day_of_week.csv")

    if hour is None or dow is None:
        return None

    hour_rows = hour[hour["hour"].eq(dt.hour)].copy()
    dow_rows = dow[dow["day_of_week"].eq(dt.weekday())].copy()

    def share(frame, label):
        row = frame[frame["label"].eq(label)]
        if row.empty:
            return None
        return float(row.iloc[0]["within_class_share"])

    return {
        "hour": int(dt.hour),
        "day_name": dt.strftime("%A"),
        "control_hour_share": share(hour_rows, 0),
        "depression_hour_share": share(hour_rows, 1),
        "control_day_share": share(dow_rows, 0),
        "depression_day_share": share(dow_rows, 1),
    }


def show_prediction_page():
    st.header("Prediction and Explainability")
    st.write(
        "Enter a social-media post. The final classifier combines DistilBERT text representations "
        "with posting-time features. LIME explains the text contribution while the posting time is held fixed."
    )

    text = st.text_area(
        "Post text",
        height=220,
        placeholder="Enter the title and/or body of a Reddit-style post...",
    )

    col1, col2 = st.columns(2)
    with col1:
        post_date = st.date_input("Posting date (UTC)")
    with col2:
        post_time = st.time_input("Posting time (UTC)")

    post_dt = datetime.combine(post_date, post_time, tzinfo=timezone.utc)

    if not model_is_ready():
        st.info(
            "The application code is ready, but the trained model artifact is not present yet. "
            "Run prepare_data.py and train_model.py first. The Temporal Analysis page can be used after data preparation."
        )
        return

    predictor = load_predictor()

    if st.button("Analyse post", type="primary", use_container_width=True):
        if len(text.strip()) < 5:
            st.error("Please enter a longer piece of text.")
        else:
            result = predictor.predict_one(text, dt=post_dt)
            st.session_state["last_text"] = text
            st.session_state["last_dt"] = post_dt
            st.session_state["last_result"] = result

    result = st.session_state.get("last_result")

    if result and st.session_state.get("last_text") == text:
        c1, c2 = st.columns(2)
        c1.metric("Model output", result["label"])
        c2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

        probs = pd.DataFrame(
            {
                "Class": list(result["probabilities"].keys()),
                "Probability": list(result["probabilities"].values()),
            }
        ).set_index("Class")
        st.bar_chart(probs)

        if st.button("Generate full LIME explanation", type="secondary"):
            with st.spinner(
                "Generating local explanation and temporal sensitivity analysis..."
            ):
                from src.explain import (
                    build_plain_language_summary,
                    contribution_figure,
                    explain_text,
                    highlight_text_html,
                    lime_dataframe,
                    split_supporting_opposing,
                    temporal_effect_text,
                )

                explanation = explain_text(
                    predictor,
                    text,
                    post_dt,
                    num_features=14,
                    num_samples=1000,
                )

            pred_label = explanation["predicted_label"]
            class_name = CLASS_NAMES[pred_label]
            confidence = float(explanation["probabilities"][pred_label])
            weights = explanation["weights"]
            temporal = explanation["temporal"]

            st.divider()
            st.subheader(f"Full explanation for: {class_name}")

            st.markdown("### 1. Plain-language model explanation")
            st.info(
                build_plain_language_summary(
                    class_name=class_name,
                    confidence=confidence,
                    weights=weights,
                )
            )

            st.markdown("### 2. Highlighted text evidence")
            st.markdown(
                "<span style='background:#1f6f43;color:white;padding:2px 6px;border-radius:4px;'>"
                "Supports displayed class</span>&nbsp;&nbsp;"
                "<span style='background:#8b2f3c;color:white;padding:2px 6px;border-radius:4px;'>"
                "Pushes against displayed class</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                highlight_text_html(text, weights),
                unsafe_allow_html=True,
            )

            st.markdown("### 3. Strongest local text contributions")
            fig = contribution_figure(weights, max_items=10)
            st.pyplot(fig, clear_figure=True, use_container_width=False)

            support, oppose = split_supporting_opposing(weights, max_each=5)
            left, right = st.columns(2)

            with left:
                st.markdown("**Supports displayed class**")
                if support:
                    for term, weight in support:
                        st.write(f"• {term}: {weight:+.4f}")
                else:
                    st.write(
                        "No strong informative supporting term in this local explanation."
                    )

            with right:
                st.markdown("**Pushes against displayed class**")
                if oppose:
                    for term, weight in oppose:
                        st.write(f"• {term}: {weight:+.4f}")
                else:
                    st.write(
                        "No strong informative opposing term in this local explanation."
                    )

            st.markdown("### 4. Posting-time contribution")
            delta_pp = float(temporal["displayed_class_delta"]) * 100.0

            t1, t2, t3 = st.columns(3)
            t1.metric(
                "Selected-time probability",
                f"{temporal['actual_probability'] * 100:.1f}%",
            )
            t2.metric(
                "Average-time probability",
                f"{temporal['average_temporal_probability'] * 100:.1f}%",
            )
            t3.metric("Local time effect", f"{delta_pp:+.2f} pp")

            st.write(temporal_effect_text(temporal, class_name))

            context = temporal_dataset_context(post_dt)
            if context is not None:
                st.markdown("**Dataset context for the selected time**")

                control_hour = context["control_hour_share"]
                dep_hour = context["depression_hour_share"]
                control_day = context["control_day_share"]
                dep_day = context["depression_day_share"]

                if None not in (
                    control_hour,
                    dep_hour,
                    control_day,
                    dep_day,
                ):
                    st.write(
                        f"At {context['hour']:02d}:00 UTC, "
                        f"{dep_hour * 100:.2f}% of depression-associated-source posts "
                        f"and {control_hour * 100:.2f}% of control-source posts in the prepared "
                        "temporal aggregates occurred in that hour. "
                        f"On {context['day_name']}, the corresponding within-class shares were "
                        f"{dep_day * 100:.2f}% and {control_day * 100:.2f}%."
                    )
                    st.caption(
                        "These are normalized population-level dataset patterns, not individual "
                        "behavioural histories and not prevalence estimates."
                    )

            with st.expander("5. Detailed LIME table and technical interpretation"):
                exp_df = lime_dataframe(weights)
                st.dataframe(
                    exp_df.style.format({"LIME weight": "{:+.4f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown(
                    f"**How to read this:** positive LIME weights support **{class_name}** for this "
                    "specific post; negative weights push against that displayed class. The table is "
                    "ranked by absolute local contribution. "
                    f"LIME generated this approximation using {explanation['num_samples']:,} perturbed "
                    "text samples while holding the selected posting time fixed."
                )
                st.warning(
                    "LIME explains the model's local prediction behaviour; it does not establish that "
                    "any highlighted word is a symptom, cause, diagnosis, or clinically meaningful "
                    "indicator on its own."
                )


def show_temporal_page():
    st.header("Temporal Behavioural Analysis")
    st.write(
        "These plots summarize population-level posting patterns. Because the Kaggle dataset does not contain "
        "author identifiers, the project does not claim longitudinal tracking of individual users."
    )

    hour = load_csv("temporal_hour.csv")
    dow = load_csv("temporal_day_of_week.csv")
    month = load_csv("temporal_month.csv")
    engagement = load_csv("temporal_engagement.csv")

    if hour is None:
        st.info("Run `python prepare_data.py` to generate the temporal aggregates.")
        return

    hour["Class"] = hour["label"].map({0: CLASS_NAMES[0], 1: CLASS_NAMES[1]})
    hour_plot = hour.pivot(
        index="hour",
        columns="Class",
        values="within_class_share",
    ).fillna(0)

    st.subheader("Posting time by hour (within-class share)")
    st.line_chart(hour_plot)

    if dow is not None:
        names = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }
        dow["day"] = dow["day_of_week"].map(names)
        dow["Class"] = dow["label"].map({0: CLASS_NAMES[0], 1: CLASS_NAMES[1]})
        order = list(names.values())
        pivot = dow.pivot(
            index="day",
            columns="Class",
            values="within_class_share",
        ).reindex(order).fillna(0)

        st.subheader("Posting activity by day of week")
        st.bar_chart(pivot)

    if month is not None:
        month["Class"] = month["label"].map({0: CLASS_NAMES[0], 1: CLASS_NAMES[1]})
        pivot = month.pivot(
            index="year_month",
            columns="Class",
            values="within_class_share",
        ).fillna(0)

        st.subheader("Posting distribution over calendar time")
        st.line_chart(pivot)
        st.caption(
            "The chart is normalized within each class. Differences can reflect collection/source effects as well as behaviour, "
            "so it is not a population prevalence estimate."
        )

    if engagement is not None:
        st.subheader("Engagement and text-length trends")
        metric = st.selectbox(
            "Metric",
            ["mean_upvotes", "mean_comments", "mean_text_length"],
        )
        engagement["Class"] = engagement["label"].map({0: CLASS_NAMES[0], 1: CLASS_NAMES[1]})
        p = engagement.pivot(
            index="year_month",
            columns="Class",
            values=metric,
        ).fillna(0)
        st.line_chart(p)
        st.caption(
            "Upvotes and comments are used for descriptive behavioural analysis only; they are not classifier inputs because "
            "they are not known at the moment a post is first published."
        )


def show_evaluation_page():
    st.header("Model Evaluation")
    hybrid_path = RESULTS_DIR / "hybrid_metrics.json"
    baseline_path = RESULTS_DIR / "baseline_metrics.json"

    if not hybrid_path.exists() and not baseline_path.exists():
        st.info("Training/evaluation results will appear here after the pipeline has been run.")
        return

    rows = []
    for name, path in [
        ("TF-IDF + Logistic Regression baseline", baseline_path),
        ("Final Hybrid DistilBERT", hybrid_path),
    ]:
        if path.exists():
            m = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "Model": name,
                    "Accuracy": m.get("accuracy"),
                    "Precision": m.get("precision"),
                    "Recall": m.get("recall"),
                    "F1": m.get("f1"),
                    "ROC-AUC": m.get("roc_auc"),
                }
            )

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    cm = RESULTS_DIR / "hybrid_confusion_matrix.png"
    if cm.exists():
        st.subheader("Final model confusion matrix")
        st.image(str(cm), width=620)

    history = RESULTS_DIR / "training_history.csv"
    if history.exists():
        h = pd.read_csv(history).set_index("epoch")
        st.subheader("Training history")

        if {"train_f1", "val_f1"}.issubset(h.columns):
            st.line_chart(h[["train_f1", "val_f1"]])

        if {"train_loss", "val_loss"}.issubset(h.columns):
            st.line_chart(h[["train_loss", "val_loss"]])


def show_project_page():
    st.header("Project Scope, Data and Responsible Use")

    manifest_path = DATA_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        st.subheader("Prepared dataset")
        st.json(manifest)

    st.subheader("What the software does")
    st.markdown(
        """
- Cleans and validates the Kaggle Reddit depression dataset.
- Fine-tunes one final hybrid DistilBERT classifier using text and posting-time features.
- Compares the final deep-learning model against a conventional TF-IDF + Logistic Regression baseline.
- Produces temporal behavioural analysis from timestamps, upvotes, comments and text length.
- Generates detailed LIME explanations for individual model predictions.
- Explains the local contribution of posting-time features to each prediction.
- Presents prediction, explainability, temporal analysis and evaluation in one Streamlit application.
        """
    )

    st.subheader("Interpretation limits")
    st.markdown(
        """
- `label = 1` represents posts collected from depression-associated source communities; it is **not** a clinical diagnosis.
- The dataset has no stable author identifier, so temporal findings are **population-level**, not longitudinal individual monitoring.
- Source/community names are removed when they appear explicitly in model text to reduce direct label leakage.
- Upvotes and comment counts are descriptive only and are excluded from live prediction to avoid using information that appears after publication.
- LIME explanations describe the model's local prediction behaviour; they do not establish clinical symptoms or causes.
- Posting-time contribution is a model-sensitivity estimate, not a causal or clinical finding.
- The software is an academic research artefact and must not be used for clinical or emergency decision-making.
        """
    )


page = st.sidebar.radio(
    "Navigation",
    ["Prediction & LIME", "Temporal Analysis", "Evaluation", "Project & Ethics"],
)

if page == "Prediction & LIME":
    show_prediction_page()
elif page == "Temporal Analysis":
    show_temporal_page()
elif page == "Evaluation":
    show_evaluation_page()
else:
    show_project_page()