# ============================================================
# MODULE 1: Data Loading, Cleaning & Sentiment Analysis
# Powered by: Groq API (FREE) — Hybrid approach (fast!)
# Works with ANY CSV dataset — auto column detection
# Course: MGNM521 – Disruptive Technologies for Business
# ============================================================

import pandas as pd
import re
import json
import time
import warnings
warnings.filterwarnings("ignore")

from groq import Groq
from column_mapper import normalize_dataframe

_client = None

def init_groq(api_key: str):
    global _client
    _client = Groq(api_key=api_key)
    print("[INFO] Groq client initialized.")


def load_data(filepath: str = "Reviews.csv", sample_size: int = 1000) -> tuple:
    """Load from file path. Returns (df, mapping, warnings)."""
    print(f"[INFO] Loading dataset from: {filepath}")
    raw = pd.read_csv(filepath)
    print(f"[INFO] Raw shape: {raw.shape}, Columns: {list(raw.columns)}")
    df, mapping, warns = normalize_dataframe(raw, sample_size)
    print(f"[INFO] Detected mapping: {mapping}")
    return df, mapping, warns


def load_data_from_upload(file_obj, sample_size: int = 1000) -> tuple:
    """Load from Streamlit uploaded file. Returns (df, mapping, warnings)."""
    raw = pd.read_csv(file_obj)
    print(f"[INFO] Uploaded CSV shape: {raw.shape}, Columns: {list(raw.columns)}")
    df, mapping, warns = normalize_dataframe(raw, sample_size)
    print(f"[INFO] Detected mapping: {mapping}")
    return df, mapping, warns


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Cleaning text data...")
    df["clean_text"] = df["Text"].apply(clean_text)
    df["clean_summary"] = df["Summary"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)
    df["input_text"] = df["clean_text"].str[:400]
    print(f"[INFO] Preprocessing complete. {len(df)} usable reviews.")
    return df


def _groq_classify_ambiguous(texts: list, batch_size: int = 50) -> list:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        numbered = "\n".join([f"{j+1}. {t[:200]}" for j, t in enumerate(batch)])
        prompt = f"""For each review output ONLY a JSON array: [{{"label":"POSITIVE","score":0.8}}, ...]
One object per review. Label must be POSITIVE or NEGATIVE only. No other text.

Reviews:
{numbered}"""
        try:
            response = _client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end > start:
                raw = raw[start:end]
            batch_results = json.loads(raw)
            while len(batch_results) < len(batch):
                batch_results.append({"label": "POSITIVE", "score": 0.5})
            results.extend(batch_results[:len(batch)])
        except Exception:
            results.extend([{"label": "POSITIVE", "score": 0.5}] * len(batch))
        time.sleep(0.5)
    return results


def run_sentiment_on_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hybrid: instant for 1,2,4,5★ — Groq only for 3★ ambiguous reviews.
    Works regardless of dataset source.
    """
    print(f"[INFO] Running hybrid sentiment on {len(df)} reviews...")

    def classify_by_star(score):
        if score >= 4:
            return "POSITIVE", 0.92
        elif score <= 2:
            return "NEGATIVE", 0.91
        else:
            return "NEUTRAL", 0.5

    labels, scores = zip(*df["Score"].apply(classify_by_star))
    df["sentiment_label"] = list(labels)
    df["sentiment_score"] = [round(s, 4) for s in scores]

    ambiguous_mask = df["Score"] == 3
    ambiguous_count = ambiguous_mask.sum()
    print(f"[INFO] {ambiguous_count} ambiguous (3★) reviews sending to Groq...")

    if ambiguous_count > 0 and _client is not None:
        ambiguous_texts = df.loc[ambiguous_mask, "input_text"].tolist()
        groq_results = _groq_classify_ambiguous(ambiguous_texts)
        df.loc[ambiguous_mask, "sentiment_label"] = [r["label"] for r in groq_results]
        df.loc[ambiguous_mask, "sentiment_score"] = [round(float(r.get("score", 0.5)), 4) for r in groq_results]

    df["sentiment"] = df["sentiment_label"].map({
        "POSITIVE": "Positive",
        "NEGATIVE": "Negative",
        "NEUTRAL": "Neutral"
    }).fillna("Neutral")

    print("[INFO] Sentiment done.")
    print(df["sentiment"].value_counts())
    return df


def get_sentiment_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    positive = (df["sentiment"] == "Positive").sum()
    negative = (df["sentiment"] == "Negative").sum()
    neutral  = (df["sentiment"] == "Neutral").sum()
    return {
        "total_reviews":    total,
        "positive_count":   int(positive),
        "negative_count":   int(negative),
        "neutral_count":    int(neutral),
        "positive_pct":     round(positive / total * 100, 1),
        "negative_pct":     round(negative / total * 100, 1),
        "neutral_pct":      round(neutral  / total * 100, 1),
        "avg_star_positive": round(df[df["sentiment"] == "Positive"]["Score"].mean(), 2),
        "avg_star_negative": round(df[df["sentiment"] == "Negative"]["Score"].mean(), 2),
    }


def get_top_negative_reviews(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return (
        df[df["sentiment"] == "Negative"]
        .sort_values("sentiment_score", ascending=False)
        .head(n)[["clean_summary", "clean_text", "Score", "sentiment_score"]]
        .reset_index(drop=True)
    )


def get_trend_over_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    trend = (
        df.groupby(["YearMonth", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    return trend
