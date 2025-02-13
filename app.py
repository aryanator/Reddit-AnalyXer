import streamlit as st
from reddit_fetcher import (
    fetch_posts,
    summarize_posts,
    get_sentiment_distribution,
    generate_word_cloud,
)
import matplotlib.pyplot as plt
import time
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Cache the summarization model to load only once
@st.cache_resource
def load_summarization_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device="cpu")
    return summarizer

# Cache the sentiment analysis model to load only once
@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load the models once
summarizer = load_summarization_model()
sentiment_analyzer = load_sentiment_analyzer()

# Streamlit app
st.title("Reddit Posts Summarizer and Analyzer")
st.markdown("**Use this link for a small and faster model:** [Alternative Model](https://reddit-summarixer.streamlit.app/)")
# User input
subreddit_name = st.text_input("Enter a subreddit name (e.g., MachineLearning):")
limit = st.slider("Number of posts to summarize:", 1, 10, 5)

# Buttons for functionalities
show_summary = st.button("Show Summaries")
show_sentiment = st.button("Show Sentiment Analysis")
show_visualization = st.button("Show Visualizations")

# Fetch posts (only once)
if subreddit_name:
    with st.spinner("Fetching posts..."):
        posts = fetch_posts(subreddit_name, limit=limit)
        time.sleep(1)  # Simulate loading delay

    if not posts:
        st.error("No posts found. Please check the subreddit name and try again.")
        st.stop()

    # Progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)  # Simulate progress
        progress_bar.progress(i + 1)
    st.success("Posts fetched successfully!")

    # Summarize posts and comments
    if show_summary:
        st.write(f"### Summarizing the top {limit} posts from r/{subreddit_name}:")
        with st.spinner("Generating summaries..."):
            summaries = summarize_posts(posts, summarizer)  # Pass the cached summarizer
            for i, summary in enumerate(summaries):
                st.write(f"**Title:** {summary['title']}")
                st.write(f"**Post Summary:** {summary['post_summary']}")
                st.write(f"**Comment Summary:** {summary['comment_summary']}")
                st.write(f"**URL:** [Read more]({summary['url']})")
                st.write("---")

    # Sentiment analysis
    if show_sentiment:
        st.write("### Sentiment Analysis of the Subreddit")
        with st.spinner("Analyzing sentiment..."):
            sentiment_distribution = get_sentiment_distribution(posts, sentiment_analyzer)
            
            # Display sentiment distribution as a bar chart
            st.write("**Sentiment Distribution**")
            labels = list(sentiment_distribution.keys())
            counts = list(sentiment_distribution.values())
            fig, ax = plt.subplots()
            ax.bar(labels, counts, color=["green", "red"])
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Distribution in Posts and Comments")
            st.pyplot(fig)

            # Display overall sentiment
            overall_sentiment = max(sentiment_distribution, key=sentiment_distribution.get)
            st.write(f"**Overall Sentiment:** {overall_sentiment}")

    # Generate and display word cloud
    if show_visualization:
        st.write("### Keywords and Word Cloud")
        with st.spinner("Generating word cloud..."):
            wordcloud = generate_word_cloud(posts)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
