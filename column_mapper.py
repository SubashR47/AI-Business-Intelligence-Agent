# ============================================================
# COLUMN MAPPER — Auto-detects columns from any CSV dataset
# ============================================================

import pandas as pd

# Common column name variations across different datasets
COLUMN_ALIASES = {
    "text": [
        "text", "review", "review_text", "reviewtext", "body", "content",
        "comment", "description", "feedback", "message", "review_body",
        "reviews", "customer_review", "review_content", "reviewbody"
    ],
    "summary": [
        "summary", "title", "headline", "subject", "review_headline",
        "review_title", "short_review", "reviewheadline", "name"
    ],
    "score": [
        "score", "rating", "stars", "star_rating", "overall", "overall_rating",
        "ratings", "review_score", "rate", "grade", "reviewscore", "starrating",
        "star", "votes", "helpful_votes"
    ],
    "product": [
        "productid", "product_id", "asin", "product", "item_id", "itemid",
        "product_name", "item", "sku", "productname"
    ],
    "date": [
        "time", "date", "timestamp", "review_date", "created_at", "posted_at",
        "reviewtime", "review_time", "datetime", "created", "post_date"
    ],
    "id": [
        "id", "review_id", "reviewid", "uid", "index", "row_id"
    ]
}


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect which columns map to text, score, summary, etc.
    Returns a dict: {"text": "actual_col_name", "score": "actual_col_name", ...}
    """
    cols_lower = {col.lower().replace(" ", "_"): col for col in df.columns}
    mapping = {}

    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in cols_lower:
                mapping[field] = cols_lower[alias]
                break

    return mapping


def normalize_dataframe(df: pd.DataFrame, sample_size: int = 1000) -> tuple:
    """
    Takes ANY CSV dataframe and normalizes it to standard column names.
    Returns: (normalized_df, detected_mapping, warnings)
    """
    warnings = []
    mapping = detect_columns(df)

    # Must have at least a text column
    if "text" not in mapping:
        # Last resort: use the longest string column
        str_cols = df.select_dtypes(include="object").columns.tolist()
        if str_cols:
            longest = max(str_cols, key=lambda c: df[c].dropna().str.len().mean())
            mapping["text"] = longest
            warnings.append(f"No review text column found — using '{longest}' as text.")
        else:
            raise ValueError("Could not find any text column in the uploaded CSV.")

    # Build normalized dataframe
    normalized = pd.DataFrame()
    normalized["Text"] = df[mapping["text"]].fillna("")

    # Summary — fall back to first 100 chars of text
    if "summary" in mapping:
        normalized["Summary"] = df[mapping["summary"]].fillna("")
    else:
        normalized["Summary"] = normalized["Text"].str[:80]
        warnings.append("No summary/title column found — using first 80 chars of text.")

    # Score — fall back to neutral 3
    if "score" in mapping:
        raw_score = pd.to_numeric(df[mapping["score"]], errors="coerce")
        # Normalize to 1-5 scale if needed (e.g., 1-10 scale)
        if raw_score.max() > 5:
            raw_score = (raw_score / raw_score.max() * 5).round()
        elif raw_score.max() <= 1:
            raw_score = (raw_score * 5).round()
        normalized["Score"] = raw_score.fillna(3).clip(1, 5)
    else:
        normalized["Score"] = 3.0
        warnings.append("No rating/score column found — defaulting to 3★ (all reviews treated as ambiguous).")

    # Product ID
    if "product" in mapping:
        normalized["ProductId"] = df[mapping["product"]].fillna("unknown")
    else:
        normalized["ProductId"] = "unknown"

    # Date
    if "date" in mapping:
        try:
            col = df[mapping["date"]]
            if pd.to_numeric(col, errors="coerce").notna().mean() > 0.8:
                # Unix timestamp
                normalized["Date"] = pd.to_datetime(col, unit="s", errors="coerce")
            else:
                normalized["Date"] = pd.to_datetime(col, errors="coerce")
        except Exception:
            normalized["Date"] = pd.Timestamp("2020-01-01")
    else:
        normalized["Date"] = pd.Timestamp("2020-01-01")
        warnings.append("No date column found — using placeholder date.")

    # ID
    if "id" in mapping:
        normalized["Id"] = df[mapping["id"]]
    else:
        normalized["Id"] = range(len(df))

    # Sample
    normalized = normalized.sample(
        n=min(sample_size, len(normalized)), random_state=42
    ).reset_index(drop=True)

    return normalized, mapping, warnings
