import streamlit as st
import praw
import pandas as pd
import re
from collections import Counter
from textblob import TextBlob
import openai

# --- CONFIG ---
st.set_page_config(page_title="Reddit Market Research Tool", layout="wide")

st.title("🧠 Reddit Market Research Tool (GummySearch Style)")

# --- Sidebar Inputs ---
st.sidebar.header("Settings")
client_id = st.sidebar.text_input("Reddit Client ID")
client_secret = st.sidebar.text_input("Reddit Client Secret")
user_agent = st.sidebar.text_input("User Agent", value="market_research_app")

subreddits_input = st.sidebar.text_input("Subreddits (comma separated)", "Entrepreneur,marketing,startups")
keywords_input = st.sidebar.text_input("Keywords (comma separated)", "hate,problem,wish,annoying,frustrating")
post_limit = st.sidebar.slider("Posts per subreddit", 10, 200, 50)

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# --- Initialize APIs ---
@st.cache_resource
def init_reddit(client_id, client_secret, user_agent):
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

# --- Helpers ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)
    return text


def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity


def fetch_data(reddit, subreddits, keywords, limit):
    data = []

    for subreddit in subreddits:
        for post in reddit.subreddit(subreddit).hot(limit=limit):
            text = clean_text(post.title + " " + (post.selftext or ""))

            if any(k in text for k in keywords):
                data.append({
                    "source": "post",
                    "subreddit": subreddit,
                    "text": post.title,
                    "score": post.score,
                    "url": post.url
                })

            post.comments.replace_more(limit=0)
            for comment in post.comments.list()[:20]:
                ctext = clean_text(comment.body)

                if any(k in ctext for k in keywords):
                    data.append({
                        "source": "comment",
                        "subreddit": subreddit,
                        "text": comment.body,
                        "score": comment.score,
                        "url": post.url
                    })

    df = pd.DataFrame(data)

    if not df.empty:
        df["sentiment"] = df["text"].apply(analyze_sentiment)

    return df


def generate_ai_summary(df, api_key):
    if df.empty:
        return "No data to summarize."

    openai.api_key = api_key

    sample_text = "\n".join(df["text"].head(50).tolist())

    prompt = f"""
    Analyze the following Reddit user comments and extract:
    1. Top 5 pain points
    2. Key desires
    3. Product opportunities
    4. 5 ad hooks using their language

    Data:
    {sample_text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# --- Run ---
if st.button("Run Research"):
    if not client_id or not client_secret:
        st.error("Please enter Reddit API credentials")
    else:
        reddit = init_reddit(client_id, client_secret, user_agent)

        subreddits = [s.strip() for s in subreddits_input.split(",")]
        keywords = [k.strip() for k in keywords_input.split(",")]

        with st.spinner("Fetching Reddit data..."):
            df = fetch_data(reddit, subreddits, keywords, post_limit)

        st.success(f"Collected {len(df)} insights")

        if not df.empty:
            st.dataframe(df)

            # Word frequency
            words = " ".join(df["text"]).split()
            common_words = Counter(words).most_common(20)

            st.subheader("Top Keywords")
            st.write(common_words)

            if openai_api_key:
                with st.spinner("Generating AI insights..."):
                    summary = generate_ai_summary(df, openai_api_key)

                st.subheader("AI Insights & Hooks")
                st.write(summary)

            # Download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "reddit_insights.csv", "text/csv")
        else:
            st.warning("No matching data found. Try different keywords.")
