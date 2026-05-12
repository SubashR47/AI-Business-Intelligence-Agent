# ============================================================
# MODULE 2: Groq-Powered Chatbot + AI Agent
# Model: llama-3.3-70b-versatile (free, fast, smart)
# Course: MGNM521 – Disruptive Technologies for Business
# ============================================================

import pandas as pd

# ──────────────────────────────────────────────
# SHARED CONTEXT
# ──────────────────────────────────────────────

_context = {
    "df": None,
    "summary": None,
    "top_negatives": None,
    "client": None,
}


def init_agent(client, df: pd.DataFrame, summary: dict, top_negatives: pd.DataFrame):
    """Call this after sentiment analysis."""
    _context["client"] = client
    _context["df"] = df
    _context["summary"] = summary
    _context["top_negatives"] = top_negatives
    print("[INFO] Groq agent context loaded.")


# ──────────────────────────────────────────────
# DATA TOOL FUNCTIONS
# ──────────────────────────────────────────────

def tool_sentiment_summary(_="") -> str:
    if _context["summary"] is None:
        return "No data loaded yet."
    s = _context["summary"]
    return (
        f"Total Reviews Analyzed: {s['total_reviews']}\n"
        f"Positive: {s['positive_count']} ({s['positive_pct']}%)\n"
        f"Negative: {s['negative_count']} ({s['negative_pct']}%)\n"
        f"Avg Star Rating (Positive reviews): {s['avg_star_positive']}\n"
        f"Avg Star Rating (Negative reviews): {s['avg_star_negative']}\n"
    )


def tool_top_complaints(_="") -> str:
    if _context["top_negatives"] is None:
        return "No data loaded yet."
    neg = _context["top_negatives"]
    out = "Top Customer Complaints:\n"
    for i, row in neg.iterrows():
        out += f"\n{i+1}. [{row['Score']}★] {row['clean_summary']}\n"
        out += f"   \"{row['clean_text'][:250]}\"\n"
    return out


def tool_business_recommendation(_="") -> str:
    if _context["summary"] is None:
        return "No data loaded yet."
    s = _context["summary"]
    neg_pct = s["negative_pct"]
    if neg_pct > 40:
        urgency, action = "CRITICAL", "Immediate product quality audit and customer service overhaul required."
    elif neg_pct > 20:
        urgency, action = "MODERATE", "Review top complaint categories and address recurring issues."
    else:
        urgency, action = "LOW", "Maintain current quality. Focus on amplifying positive reviews."
    return (
        f"Business Recommendation (Urgency: {urgency}):\n"
        f"  Negative Sentiment Rate: {neg_pct}%\n"
        f"  Action: {action}\n"
        f"  KPI Target: Reduce negative reviews below 15% within 2 quarters.\n"
        f"  Steps:\n"
        f"    1. Identify top 3 complaint categories\n"
        f"    2. Set up real-time sentiment monitoring\n"
        f"    3. Create rapid response SOP for 1-2 star reviews\n"
        f"    4. Track Net Promoter Score monthly\n"
    )


def tool_search_reviews(query: str) -> str:
    if _context["df"] is None:
        return "No data loaded yet."
    df = _context["df"]
    mask = (
        df["clean_text"].str.contains(query, case=False, na=False) |
        df["clean_summary"].str.contains(query, case=False, na=False)
    )
    filtered = df[mask]
    if filtered.empty:
        return f"No reviews found mentioning '{query}'."
    pos = (filtered["sentiment"] == "Positive").sum()
    neg = (filtered["sentiment"] == "Negative").sum()
    neu = (filtered["sentiment"] == "Neutral").sum()
    avg = filtered["Score"].mean()
    sample_pos = filtered[filtered["sentiment"] == "Positive"]["clean_text"].values
    sample_neg = filtered[filtered["sentiment"] == "Negative"]["clean_text"].values
    sample_neu = filtered[filtered["sentiment"] == "Neutral"]["clean_text"].values
    out = f"Reviews mentioning '{query}': {len(filtered)} | Avg Rating: {avg:.2f}★\n"
    out += f"  😊 Positive: {pos} | 😤 Negative: {neg} | 😐 Neutral: {neu}\n"
    if len(sample_pos) > 0:
        out += f"\n✅ Sample positive: \"{sample_pos[0][:250]}\"\n"
    if len(sample_neg) > 0:
        out += f"\n❌ Sample negative: \"{sample_neg[0][:250]}\"\n"
    if len(sample_neu) > 0:
        out += f"\n😐 Sample neutral: \"{sample_neu[0][:250]}\"\n"
    return out


def _get_best_product(df) -> str:
    if df is None or df.empty:
        return "No data loaded."
    best = (
        df.groupby("ProductId")["Score"]
        .agg(["mean", "count"])
        .query("count >= 5")
        .sort_values("mean", ascending=False)
        .head(5).reset_index()
    )
    if best.empty:
        return "Not enough data to rank products."
    out = "🏆 Top Rated Products (min. 5 reviews):\n"
    for i, row in best.iterrows():
        out += f"  {i+1}. Product ID: {row['ProductId']} — Avg: {row['mean']:.2f}★ ({int(row['count'])} reviews)\n"
    return out


def _get_worst_product(df) -> str:
    if df is None or df.empty:
        return "No data loaded."
    worst = (
        df.groupby("ProductId")["Score"]
        .agg(["mean", "count"])
        .query("count >= 5")
        .sort_values("mean", ascending=True)
        .head(5).reset_index()
    )
    if worst.empty:
        return "Not enough data."
    out = "⚠️ Lowest Rated Products:\n"
    for i, row in worst.iterrows():
        out += f"  {i+1}. Product ID: {row['ProductId']} — Avg: {row['mean']:.2f}★ ({int(row['count'])} reviews)\n"
    return out


def _get_average_rating(df) -> str:
    if df is None or df.empty:
        return "No data loaded."
    avg = df["Score"].mean()
    dist = df["Score"].value_counts().sort_index()
    out = f"⭐ Overall Average Star Rating: {avg:.2f}/5\n\nRating Breakdown:\n"
    for star, count in dist.items():
        bar = "█" * int(count / dist.max() * 20)
        out += f"  {star}★ {bar} ({count})\n"
    return out


def _get_review_count_by_year(df) -> str:
    if df is None or df.empty:
        return "No data loaded."
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    counts = df["Year"].value_counts().sort_index()
    out = "📅 Reviews by Year:\n"
    for year, cnt in counts.items():
        out += f"  {year}: {cnt} reviews\n"
    return out


# ──────────────────────────────────────────────
# CONTEXT BUILDER
# ──────────────────────────────────────────────

def _build_data_context() -> str:
    df = _context["df"]
    s = _context["summary"]
    if s is None:
        return "No dataset loaded yet."
    return f"""=== DATASET CONTEXT ===
Uploaded Review Dataset — {s['total_reviews']} reviews analyzed

SENTIMENT OVERVIEW:
{tool_sentiment_summary()}
TOP COMPLAINTS:
{tool_top_complaints()}
BEST PRODUCTS:
{_get_best_product(df)}
WORST PRODUCTS:
{_get_worst_product(df)}
AVERAGE RATING:
{_get_average_rating(df)}
REVIEWS BY YEAR:
{_get_review_count_by_year(df)}
=== END CONTEXT ==="""


# ──────────────────────────────────────────────
# MAIN GROQ CHATBOT
# ──────────────────────────────────────────────

def groq_chatbot(user_query: str, chat_history: list) -> str:
    """
    Groq-powered chatbot using llama3-70b.
    Fast, free, and understands any natural language question.
    """
    client = _context["client"]
    df = _context["df"]

    if client is None:
        return "Groq client not initialized. Please enter your API key."

    data_context = _build_data_context()

    # Live product/keyword search
    extra_search = ""
    q_lower = user_query.lower()
    stop_words = {"what", "do", "you", "think", "about", "of", "to", "the", "is", "are",
                  "for", "a", "an", "reviews", "on", "tell", "me", "show", "find", "how",
                  "people", "customers", "feel", "say", "said", "opinion", "thoughts",
                  "best", "worst", "rating", "summary", "recommend", "should", "we"}
    words = [w for w in q_lower.split() if w not in stop_words and len(w) > 2]
    if df is not None and words:
        for length in [3, 2, 1]:
            for i in range(len(words) - length + 1):
                phrase = " ".join(words[i:i+length])
                result = tool_search_reviews(phrase)
                if "No reviews found" not in result:
                    extra_search = f"\n\nLIVE SEARCH RESULT for '{phrase}':\n{result}"
                    break
            if extra_search:
                break

    system_prompt = f"""You are an expert AI Business Intelligence Agent for an e-commerce analytics platform.
You have analyzed an uploaded customer review dataset and must answer any question about it intelligently.

{data_context}{extra_search}

INSTRUCTIONS:
- Answer naturally and conversationally like a smart business analyst
- Use the data context above to give specific, accurate answers
- If asked about a specific product/food (e.g., twizzlers, coffee, chips), use the live search result
- Give actionable business insights when relevant
- Be concise but complete — 3-6 sentences unless a detailed breakdown is needed
- If the data doesn't contain specific info, say so honestly
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages += chat_history[-6:]
    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()
