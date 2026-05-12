# ============================================================
# MODULE 3: Streamlit Web App — Universal AI Business Intelligence Agent
# Powered by: Groq API (FREE) — Works with ANY CSV dataset
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq

from sentiment_engine import (
    init_groq,
    load_data,
    load_data_from_upload,
    preprocess,
    run_sentiment_on_df,
    get_sentiment_summary,
    get_top_negative_reviews,
    get_trend_over_time,
)
from chatbot_agent import init_agent, groq_chatbot, tool_business_recommendation


st.set_page_config(
    page_title="AI Business Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border-radius: 12px; padding: 1.2rem;
        border: 1px solid #2e3454; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7c83fd; }
    .metric-label { font-size: 0.85rem; color: #8b8fa8; margin-top: 0.3rem; }
    .chat-bubble-user {
        background: #1e2130; border-left: 3px solid #7c83fd;
        padding: 0.8rem 1rem; border-radius: 0 12px 12px 12px;
        margin: 0.5rem 0; color: #e0e0e0;
    }
    .chat-bubble-bot {
        background: #131720; border-left: 3px solid #4caf8a;
        padding: 0.8rem 1rem; border-radius: 0 12px 12px 12px;
        margin: 0.5rem 0; color: #e0e0e0; white-space: pre-wrap;
    }
    .mapping-badge {
        background: #1e2130; border: 1px solid #2e3454;
        border-radius: 8px; padding: 0.4rem 0.7rem;
        font-size: 0.78rem; color: #8b8fa8; margin: 2px;
        display: inline-block;
    }
    h1, h2, h3 { color: #e8eaf6 !important; }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ──
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 Groq API Key", type="password", placeholder="gsk_...",
        help="Free at console.groq.com — no credit card needed!"
    )
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📁 Upload ANY CSV dataset",
        type=["csv"],
        help="Works with Amazon Reviews, Flipkart, Twitter, Yelp, IMDB, or any review CSV!"
    )

    st.markdown("**Supported datasets:**")
    st.markdown("""
    ✅ Amazon Fine Food Reviews  
    ✅ Flipkart Product Reviews  
    ✅ Yelp Reviews  
    ✅ IMDB Movie Reviews  
    ✅ Twitter Sentiment  
    ✅ Any CSV with text + rating  
    """)

    sample_size = st.slider("Sample Size", min_value=100, max_value=2000, value=500, step=100)

    st.markdown("---")
    st.markdown("**Models:**")
    st.code("Sentiment: llama-3.1-8b-instant\nChatbot:   llama-3.3-70b-versatile", language="text")
    st.caption("MGNM521 · LPU · Apr-Jun 2026")


# ── API KEY GATE ──
if not api_key:
    st.title("🤖 AI Business Intelligence Agent")
    st.info("👈 Enter your **Groq API Key** in the sidebar to get started.")
    st.markdown("""
    **Get a FREE Groq API key:**
    1. Go to **[console.groq.com](https://console.groq.com)**
    2. Sign up (free, no credit card)
    3. Click **API Keys → Create API Key**
    4. Paste it in the sidebar
    
    > ⚡ Works with **any review CSV** — Amazon, Flipkart, Yelp, IMDB, Twitter, and more!
    """)
    st.stop()


# ── DATA LOADING ──
@st.cache_data(show_spinner=False)
def get_processed_data(file_key, sample_size, _api_key, is_upload=False):
    init_groq(_api_key)
    if is_upload:
        df, mapping, warns = load_data_from_upload(file_key, sample_size)
    else:
        df, mapping, warns = load_data(file_key, sample_size)
    df = preprocess(df)
    df = run_sentiment_on_df(df)
    return df, mapping, warns


st.title("🤖 AI Business Intelligence Agent")
st.markdown("**Universal Review Analytics · Groq llama3 · Works with any CSV**")
st.markdown("---")

import io
import glob

# ── AUTO-DETECT CSV FILES IN DIRECTORY ──
local_csvs = sorted(glob.glob("*.csv"))

# ── DATASET SELECTION LOGIC ──
if not uploaded_file and not local_csvs:
    # No file anywhere — show upload instructions
    st.markdown("### 📂 Upload your dataset to get started")
    st.info(
        "👈 Use the **Upload CSV** button in the sidebar to upload your review dataset.\n\n"
        "**Supported:** Amazon Reviews, Flipkart, Yelp, IMDB, Twitter, or any CSV with text + rating columns."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Your CSV should have at least:**")
        st.markdown("- A **text/review** column\n- A **rating/score** column (1–5 stars)")
    with col2:
        st.markdown("**📦 Known working datasets:**")
        st.markdown("- Amazon Fine Food Reviews\n- Flipkart Reviews\n- Yelp Reviews\n- IMDB Reviews")
    st.stop()

# CSVs found locally but nothing uploaded yet — ask user
if not uploaded_file and local_csvs:
    st.markdown("### 📂 Dataset Selection")
    st.success(f"🗂️ Found **{len(local_csvs)}** CSV file(s) in the current directory:")

    options = local_csvs + ["⬆️ Upload a different file instead"]
    choice = st.radio(
        "Which dataset do you want to use?",
        options=options,
        index=0,
    )

    if choice == "⬆️ Upload a different file instead":
        st.info("👈 Use the **Upload CSV** button in the sidebar to upload your file.")
        st.stop()
    else:
        selected_local = choice
        if st.button(f"▶️ Use  {selected_local}", type="primary", use_container_width=True):
            st.session_state["selected_local_csv"] = selected_local
            st.rerun()

    # Wait until user confirms
    if "selected_local_csv" not in st.session_state:
        st.stop()

# ── RESOLVE FINAL DATA SOURCE ──
use_local = not uploaded_file and "selected_local_csv" in st.session_state

with st.spinner("⚡ Analyzing reviews... (usually done in 2-5 seconds!)"):
    try:
        if uploaded_file:
            file_bytes = uploaded_file.read()
            df, mapping, warns = get_processed_data(
                io.BytesIO(file_bytes), sample_size, api_key, is_upload=True
            )
            dataset_name = uploaded_file.name
        else:
            local_path = st.session_state["selected_local_csv"]
            df, mapping, warns = get_processed_data(local_path, sample_size, api_key, is_upload=False)
            dataset_name = local_path

        summary = get_sentiment_summary(df)
        top_neg = get_top_negative_reviews(df, n=5)
        trend = get_trend_over_time(df)

        client = Groq(api_key=api_key)
        init_agent(client, df, summary, top_neg)

        st.success(f"✅ Analyzed **{summary['total_reviews']} reviews** from **{dataset_name}** successfully!")

        if warns:
            with st.expander("⚠️ Column detection notes"):
                for w in warns:
                    st.warning(w)

        with st.expander("🔍 Auto-detected columns"):
            st.markdown("The app automatically mapped your CSV columns:")
            for field, col in mapping.items():
                st.markdown(f'<span class="mapping-badge">**{field}** → `{col}`</span>', unsafe_allow_html=True)

    except ValueError as e:
        st.error(f"⚠️ Could not process this CSV: {e}")
        st.markdown("Make sure your file has at least one **text/review column**.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")
        st.stop()


# ── TABS ──
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💬 AI Chatbot", "📝 Raw Data", "📈 Trends"])


# ── TAB 1: DASHBOARD ──
with tab1:
    st.subheader(f"Business Sentiment Overview — {dataset_name}")

    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        (str(summary['total_reviews']),  "Reviews Analyzed",   "#7c83fd"),
        (f"{summary['positive_pct']}%",  "Positive Sentiment", "#4caf8a"),
        (f"{summary['negative_pct']}%",  "Negative Sentiment", "#ef5350"),
        (f"{summary['neutral_pct']}%",   "Neutral / Mixed",    "#ffb74d"),
        (f"{summary['avg_star_positive']}★", "Avg Stars (Positive)", "#90caf9"),
    ]
    for col, (val, label, color) in zip([col1, col2, col3, col4, col5], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        fig_pie = px.pie(
            values=[summary["positive_count"], summary["negative_count"], summary["neutral_count"]],
            names=["Positive", "Negative", "Neutral"],
            color_discrete_sequence=["#4caf8a", "#ef5350", "#ffb74d"],
            title="Sentiment Distribution", hole=0.45,
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        star_counts = df["Score"].value_counts().sort_index()
        fig_bar = px.bar(
            x=star_counts.index, y=star_counts.values,
            labels={"x": "Star Rating", "y": "Reviews"},
            title="Star Rating Distribution",
            color=star_counts.values,
            color_continuous_scale=["#ef5350", "#ffb74d", "#4caf8a"],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0", showlegend=False, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("🚨 Top Customer Complaints")
    for _, row in top_neg.iterrows():
        with st.expander(f"⭐ {int(row['Score'])} — {row['clean_summary']}"):
            st.write(row["clean_text"][:500])
            st.caption(f"Confidence: {row['sentiment_score']:.2%}")


# ── TAB 2: CHATBOT ──
with tab2:
    st.subheader("💬 Ask the Groq llama3 Business Agent")
    st.markdown("Ask **anything** in natural language about your dataset.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                f"👋 Hi! I've analyzed **{summary['total_reviews']} reviews** from **{dataset_name}**.\n\n"
                "Ask me anything:\n"
                "• *What do people think about [product]?*\n"
                "• *What are the main complaints?*\n"
                "• *Which product has the worst reviews?*\n"
                "• *What should the business do to improve?*"
            )
        }]

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Ask anything about the reviews...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("⚡ Groq is thinking..."):
            history = [m for m in st.session_state.messages if m["role"] != "system"]
            response = groq_chatbot(user_input, history[:-1])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    st.markdown("**Quick Questions:**")
    quick_qs = ["Give me a full summary", "What are customers unhappy about?",
                "What should the business do?", "Best rated product?",
                "What do people say about taste?", "Sentiment vs star ratings?"]
    cols = st.columns(3)
    for i, q in enumerate(quick_qs):
        if cols[i % 3].button(q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            history = [m for m in st.session_state.messages if m["role"] != "system"]
            response = groq_chatbot(q, history[:-1])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()


# ── TAB 3: RAW DATA ──
with tab3:
    st.subheader("📝 Analyzed Reviews")
    sentiment_filter = st.selectbox("Filter by Sentiment:", ["All", "Positive", "Negative", "Neutral"])
    display_df = df if sentiment_filter == "All" else df[df["sentiment"] == sentiment_filter]
    st.dataframe(
        display_df[["clean_summary", "Score", "sentiment", "sentiment_score", "Date"]]
        .head(200)
        .rename(columns={
            "clean_summary": "Review Summary", "Score": "Stars",
            "sentiment": "Sentiment", "sentiment_score": "Confidence", "Date": "Date"
        }),
        use_container_width=True, height=450,
    )
    st.caption(f"Showing top 200 of {len(display_df):,} filtered reviews")


# ── TAB 4: TRENDS ──
with tab4:
    st.subheader("📈 Sentiment Trends Over Time")
    if "Positive" in trend.columns and "Negative" in trend.columns:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend["YearMonth"], y=trend["Positive"], mode="lines+markers",
            name="Positive", line=dict(color="#4caf8a", width=2),
            fill="tozeroy", fillcolor="rgba(76,175,138,0.1)"
        ))
        fig_trend.add_trace(go.Scatter(
            x=trend["YearMonth"], y=trend["Negative"], mode="lines+markers",
            name="Negative", line=dict(color="#ef5350", width=2),
            fill="tozeroy", fillcolor="rgba(239,83,80,0.1)"
        ))
        if "Neutral" in trend.columns:
            fig_trend.add_trace(go.Scatter(
                x=trend["YearMonth"], y=trend["Neutral"], mode="lines+markers",
                name="Neutral", line=dict(color="#ffb74d", width=2),
                fill="tozeroy", fillcolor="rgba(255,183,77,0.1)"
            ))
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0", xaxis_title="Month", yaxis_title="Reviews",
            legend=dict(bgcolor="rgba(0,0,0,0)"), hovermode="x unified",
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Not enough date variation in sample. Try increasing sample size.")

    st.markdown("---")
    st.subheader("🧠 AI Business Insights")
    st.info(tool_business_recommendation())
