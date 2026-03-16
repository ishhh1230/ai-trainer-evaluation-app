import re
from datetime import datetime
import streamlit as st


def analyze_text(text: str) -> dict:
    words = text.split()
    cleaned_words = [w.strip(".,!?;:").lower() for w in words]
    unique_keywords = [w for w in cleaned_words if len(w) > 4]

    return {
        "word_count": len(words),
        "char_count": len(text),
        "keywords": list(dict.fromkeys(unique_keywords))[:8]
    }


def classify_text(text: str, task: str = "Sentiment") -> dict:
    text_lower = text.lower()

    positive_words = ["good", "great", "excellent", "amazing", "love", "happy", "best", "useful"]
    negative_words = ["bad", "worst", "hate", "poor", "terrible", "awful", "waste"]
    toxic_words = ["idiot", "stupid", "shut up", "hate you", "useless"]
    spam_words = ["free money", "click here", "win now", "limited offer", "claim prize"]

    if task == "Sentiment":
        pos_score = sum(word in text_lower for word in positive_words)
        neg_score = sum(word in text_lower for word in negative_words)

        if pos_score > neg_score:
            return {
                "label": "Positive",
                "confidence": min(95, 70 + pos_score * 8),
                "reason": "More positive sentiment indicators were found."
            }
        elif neg_score > pos_score:
            return {
                "label": "Negative",
                "confidence": min(95, 70 + neg_score * 8),
                "reason": "More negative sentiment indicators were found."
            }
        else:
            return {
                "label": "Neutral",
                "confidence": 75,
                "reason": "Strong positive or negative indicators were not detected."
            }

    elif task == "Toxicity":
        toxic_score = sum(word in text_lower for word in toxic_words)
        if toxic_score > 0:
            return {
                "label": "Toxic",
                "confidence": min(96, 80 + toxic_score * 5),
                "reason": "Toxic or abusive language indicators were found."
            }
        return {
            "label": "Non-Toxic",
            "confidence": 88,
            "reason": "No major toxic expressions were detected."
        }

    elif task == "Spam":
        spam_score = sum(word in text_lower for word in spam_words)
        if spam_score > 0:
            return {
                "label": "Spam",
                "confidence": min(95, 80 + spam_score * 5),
                "reason": "Spam-like promotional phrases were identified."
            }
        return {
            "label": "Not Spam",
            "confidence": 86,
            "reason": "The text does not appear to contain common spam patterns."
        }

    return {
        "label": "Unknown",
        "confidence": 50,
        "reason": "Task type not recognized."
    }


def score_relevance(prompt: str, response: str) -> int:
    prompt_words = set(re.findall(r"\b\w+\b", prompt.lower()))
    response_words = set(re.findall(r"\b\w+\b", response.lower()))
    overlap = len(prompt_words.intersection(response_words))

    if overlap >= 8:
        return 5
    elif overlap >= 5:
        return 4
    elif overlap >= 3:
        return 3
    elif overlap >= 1:
        return 2
    return 1


def score_clarity(response: str) -> int:
    word_count = len(response.split())
    if word_count >= 80:
        return 5
    elif word_count >= 50:
        return 4
    elif word_count >= 25:
        return 3
    elif word_count >= 10:
        return 2
    return 1


def score_safety(response: str) -> int:
    unsafe_patterns = ["kill", "harm", "attack", "hate", "abuse"]
    found = sum(p in response.lower() for p in unsafe_patterns)

    if found == 0:
        return 5
    elif found == 1:
        return 3
    return 1


def score_factuality(response: str) -> int:
    # heuristic placeholder
    # real factuality would use external validation or model-based scoring
    if len(response.split()) >= 40:
        return 4
    elif len(response.split()) >= 20:
        return 3
    return 2


def evaluate_responses(prompt: str, response_a: str, response_b: str) -> dict:
    a_scores = {
        "relevance": score_relevance(prompt, response_a),
        "clarity": score_clarity(response_a),
        "safety": score_safety(response_a),
        "factuality": score_factuality(response_a),
    }
    a_scores["total"] = sum(a_scores.values())

    b_scores = {
        "relevance": score_relevance(prompt, response_b),
        "clarity": score_clarity(response_b),
        "safety": score_safety(response_b),
        "factuality": score_factuality(response_b),
    }
    b_scores["total"] = sum(b_scores.values())

    if a_scores["total"] > b_scores["total"]:
        winner = "Response A"
    elif b_scores["total"] > a_scores["total"]:
        winner = "Response B"
    else:
        winner = "Tie"

    summary = (
        f"The system selected {winner} based on combined scores for "
        f"relevance, clarity, safety, and factuality."
    )

    return {
        "A": a_scores,
        "B": b_scores,
        "winner": winner,
        "summary": summary
    }


def init_annotation_state():
    if "annotations" not in st.session_state:
        st.session_state.annotations = []


def add_annotation_row(text: str, label: str, annotator: str, notes: str):
    st.session_state.annotations.append({
        "text": text,
        "label": label,
        "annotator": annotator,
        "notes": notes,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


def reset_annotations():
    st.session_state.annotations = []