import streamlit as st
import pandas as pd
from utils import (
    analyze_text,
    classify_text,
    evaluate_responses,
    init_annotation_state,
    add_annotation_row,
    reset_annotations,
)

st.set_page_config(
    page_title="AI Trainer Evaluation App",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Trainer Evaluation App")
st.caption("A portfolio project for AI Trainer, Data Annotation, and LLM Evaluation roles.")

menu = st.sidebar.radio(
    "Choose Module",
    [
        "Home",
        "Text Classification",
        "Manual Annotation",
        "Response Evaluation",
        "Batch Processing"
    ]
)

# ---------------- HOME ----------------
if menu == "Home":
    st.subheader("Project Overview")
    st.write("""
This web app demonstrates practical workflows used in AI Trainer and Data Annotation jobs.

### Included Features
- Text classification
- Manual dataset labeling
- AI response comparison and ranking
- CSV upload and batch processing
- Downloadable results

### Skills Demonstrated
- Python
- Streamlit
- Data labeling
- Response evaluation
- LLM output analysis
- CSV handling
- Annotation workflow design
    """)

# ---------------- TEXT CLASSIFICATION ----------------
elif menu == "Text Classification":
    st.subheader("🏷️ Text Classification")

    text = st.text_area("Enter text", height=180)

    task = st.selectbox(
        "Select classification task",
        ["Sentiment", "Toxicity", "Spam"]
    )

    if st.button("Classify"):
        if text.strip():
            result = classify_text(text, task)
            st.success("Prediction completed")
            st.write(f"**Label:** {result['label']}")
            st.write(f"**Confidence:** {result['confidence']}%")
            st.write(f"**Reason:** {result['reason']}")
        else:
            st.warning("Please enter text first.")

# ---------------- MANUAL ANNOTATION ----------------
elif menu == "Manual Annotation":
    st.subheader("📝 Manual Annotation Tool")

    init_annotation_state()

    text_to_label = st.text_area("Enter text to annotate", height=140)

    label = st.selectbox(
        "Choose label",
        ["Positive", "Negative", "Neutral", "Toxic", "Safe", "Spam", "Custom"]
    )

    annotator = st.text_input("Annotator Name", value="Ishwari")

    notes = st.text_area("Optional Notes", height=100)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Add Annotation"):
            if text_to_label.strip():
                add_annotation_row(text_to_label, label, annotator, notes)
                st.success("Annotation added successfully.")
            else:
                st.warning("Enter text before saving annotation.")

    with col2:
        if st.button("Reset All Annotations"):
            reset_annotations()
            st.info("All annotations cleared.")

    with col3:
        if len(st.session_state.annotations) > 0:
            annotation_df = pd.DataFrame(st.session_state.annotations)
            csv = annotation_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="annotations.csv",
                mime="text/csv"
            )

    st.markdown("### Current Annotations")
    if len(st.session_state.annotations) > 0:
        st.dataframe(pd.DataFrame(st.session_state.annotations), use_container_width=True)
    else:
        st.info("No annotations added yet.")

# ---------------- RESPONSE EVALUATION ----------------
elif menu == "Response Evaluation":
    st.subheader("⚖️ AI Response Evaluation")

    prompt = st.text_area("Prompt", height=100)
    response_a = st.text_area("Response A", height=180)
    response_b = st.text_area("Response B", height=180)

    if st.button("Evaluate Responses"):
        if prompt.strip() and response_a.strip() and response_b.strip():
            result = evaluate_responses(prompt, response_a, response_b)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Response A")
                st.write(f"**Relevance:** {result['A']['relevance']}/5")
                st.write(f"**Clarity:** {result['A']['clarity']}/5")
                st.write(f"**Safety:** {result['A']['safety']}/5")
                st.write(f"**Factuality:** {result['A']['factuality']}/5")
                st.write(f"**Total Score:** {result['A']['total']}/20")

            with col2:
                st.markdown("### Response B")
                st.write(f"**Relevance:** {result['B']['relevance']}/5")
                st.write(f"**Clarity:** {result['B']['clarity']}/5")
                st.write(f"**Safety:** {result['B']['safety']}/5")
                st.write(f"**Factuality:** {result['B']['factuality']}/5")
                st.write(f"**Total Score:** {result['B']['total']}/20")

            st.success(f"Winner: {result['winner']}")
            st.info(result["summary"])
        else:
            st.warning("Please complete prompt, Response A, and Response B.")

# ---------------- BATCH PROCESSING ----------------
elif menu == "Batch Processing":
    st.subheader("📂 Batch Processing")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(df.head(), use_container_width=True)

        text_col = st.selectbox("Select text column", df.columns)
        batch_task = st.selectbox("Choose task", ["Sentiment", "Toxicity", "Spam"])

        if st.button("Run Batch Classification"):
            output_rows = []

            for text in df[text_col].astype(str):
                result = classify_text(text, batch_task)
                output_rows.append({
                    "text": text,
                    "predicted_label": result["label"],
                    "confidence": result["confidence"],
                    "reason": result["reason"]
                })

            output_df = pd.DataFrame(output_rows)
            final_df = pd.concat([df.reset_index(drop=True), output_df.drop(columns=["text"])], axis=1)

            st.write("### Processed Results")
            st.dataframe(final_df, use_container_width=True)

            csv = final_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Processed CSV",
                data=csv,
                file_name="batch_results.csv",
                mime="text/csv"
            )