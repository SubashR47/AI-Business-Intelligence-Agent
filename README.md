# AI Business Intelligence Agent

Universal AI-powered Business Intelligence Agent using Groq Llama3, Sentiment Analysis, Conversational AI, and Streamlit Dashboard.

---

# Project Overview

The AI Business Intelligence Agent is an end-to-end analytics platform designed to analyze customer review datasets using Artificial Intelligence and Natural Language Processing techniques.

The system combines:

- Hybrid Sentiment Analysis
- Large Language Models (LLMs)
- Conversational AI
- Business Intelligence Dashboards
- Automated Review Analytics

The application supports multiple CSV review datasets including:

- Amazon Reviews
- Flipkart Reviews
- Yelp Reviews
- IMDB Reviews
- Twitter Sentiment Datasets
- Custom Review CSV Files

The system automatically detects dataset columns, performs sentiment analysis, generates business insights, and allows users to interact with the data through an AI-powered chatbot.

---

# Features

## Universal CSV Dataset Support

Automatically detects:

- Review text columns
- Rating columns
- Product columns
- Date columns
- Summary/title columns

Supports almost any review-based CSV dataset.

---

## Hybrid Sentiment Analysis

Uses:

- Rule-based star classification
- Groq Llama3 contextual sentiment analysis

Classification categories:

- Positive
- Negative
- Neutral

---

## AI-Powered Chatbot

Interactive conversational AI agent capable of answering:

- What are customers unhappy about?
- Which products have the worst reviews?
- What business improvements are needed?
- Sentiment trend analysis
- Product-specific review analysis

---

## Business Intelligence Dashboard

Interactive Streamlit dashboard including:

- KPI metrics
- Sentiment distribution
- Rating analytics
- Trend visualization
- Complaint analysis
- Raw review exploration

---

## Trend Analysis

Visualizes:

- Positive sentiment trends
- Negative sentiment trends
- Neutral review trends
- Review activity over time

---

## Business Recommendations

Automatically generates:

- Complaint summaries
- Urgency levels
- KPI targets
- Strategic recommendations

---

# AI Architecture

## System Workflow

```text
User Uploads CSV Dataset
            ↓
Automatic Column Detection
            ↓
Data Cleaning & Preprocessing
            ↓
Hybrid Sentiment Analysis
            ↓
Groq Llama3 Processing
            ↓
Business Intelligence Analysis
            ↓
AI Chatbot Interaction
            ↓
Dashboard Visualization
            ↓
Business Insights & Recommendations

```

# Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Dashboard Framework | Streamlit |
| AI/LLM Provider | Groq |
| Language Models | Llama 3 |
| Data Processing | Pandas |
| Visualization | Plotly |
| NLP | Regex + LLM |
| IDE | Visual Studio Code |

---

# Installation

## Step 1 — Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Business-Intelligence-Agent.git
Step 2 — Open Project Folder
cd AI-Business-Intelligence-Agent
Step 3 — Install Dependencies
pip install -r requirements.txt
Step 4 — Run Streamlit Application
streamlit run app.py

```

# Groq API Setup

This project uses Groq-hosted Llama3 models.

Get FREE API Key

Visit:

https://console.groq.com

Steps:
Sign up
Create API Key
Paste API key inside Streamlit sidebar

No credit card required.

# Supported Datasets

The system supports datasets such as:

Amazon Fine Food Reviews
Flipkart Product Reviews
Yelp Reviews
IMDB Reviews
Twitter Sentiment Datasets
Any CSV with review text + ratings

# Dashboard Modules

1. Business Sentiment Overview

Displays:

<img width="1920" height="1080" alt="Screenshot 2026-05-12 104839" src="https://github.com/user-attachments/assets/45f04d75-73eb-482c-a77c-849842b3efda" />

<img width="1920" height="1080" alt="Screenshot 2026-05-12 104846" src="https://github.com/user-attachments/assets/98f11896-2733-419a-a09a-20349a7d94f0" />

Total reviews analyzed
Positive sentiment %
Negative sentiment %
Neutral sentiment %
Average ratings

2. Sentiment Distribution

Interactive pie chart visualization.

3. Rating Distribution

Bar chart showing customer star ratings.

4. Complaint Analysis

Displays top negative customer complaints.

5. Raw Review Explorer

Filter reviews by:

Positive
Negative
Neutral

6. Trend Analysis

Visualizes sentiment trends over time.

<img width="1916" height="869" alt="image" src="https://github.com/user-attachments/assets/d8f9fd5c-5c09-4bd7-8b66-df7e56f90f52" />

7. AI Chatbot

<img width="1919" height="879" alt="image" src="https://github.com/user-attachments/assets/8768b5f7-c0ce-454b-a3cf-f523dd0d908c" />

Allows conversational querying of dataset insights.

# Example Chatbot Questions
What are customers unhappy about?

Which products have the worst reviews?

What should the business improve?

Give me a sentiment summary.

What do customers think about coffee?

Show sentiment trends over time.
