import streamlit as st
from reddit_fetcher import (
    fetch_posts,
    get_sentiment_distribution,
    generate_word_cloud,
)
import matplotlib.pyplot as plt
import time
import os
from together import Together
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer


# Cache the summarization model to load only once

# Cache the sentiment analysis model to load only once
@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load the models once
#summarizer = load_summarization_model()
sentiment_analyzer = load_sentiment_analyzer()

# Function to generate summary using the API
def generate_summary(prompt):
    os.environ["TOGETHER_API_KEY"] = "3045ace567b59cd96ed78310bee29038b11611cfce527e0da8ed9c7ae4da67e1"
    client = Together(api_key=os.environ["TOGETHER_API_KEY"])
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"],
        stream=True
    )
    result = ""
    for token in response:
        if hasattr(token, 'choices'):
            result += token.choices[0].delta.content
    return result

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

    # Summarize posts and comments using API
    if show_summary:
        st.write(f"### Summarizing the top {limit} posts from r/{subreddit_name}:")
        with st.spinner("Generating summaries..."):
            summaries = []
            for post in posts:
                prompt = f"Summarize the following Reddit post and its top comments:\n{post}\n"
                summary = generate_summary(prompt)
                summaries.append({
                    "title": post['title'],
                    "post_summary": summary,
                    "url": post['url']
                })
            for summary in summaries:
                st.write(f"**Title:** {summary['title']}")
                st.write(f"**Post Summary:** {summary['post_summary']}")
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
