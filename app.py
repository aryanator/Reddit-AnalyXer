import streamlit as st
from reddit_fetcher import (
    fetch_posts,
    generate_word_cloud,
)
import matplotlib.pyplot as plt
import time
import os
from together import Together

# Function to generate summary and sentiment using the Together API
def generate_summary_and_sentiment(prompt):
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
    sentiment = None
    for token in response:
        if hasattr(token, 'choices'):
            content = token.choices[0].delta.content
            result += content

            # Check if sentiment is embedded in the response
            if "sentiment" in content.lower():
                sentiment = content

    return result, sentiment

# Streamlit app
st.title("Reddit Posts Summarizer and Analyzer")
st.markdown("**Use this link for a small and faster model:** [Alternative Model](https://reddit-summarixer.streamlit.app/)")

subreddit_name = st.text_input("Enter a subreddit name (e.g., MachineLearning):")
limit = st.slider("Number of posts to summarize:", 1, 10, 5)

show_summary = st.button("Show Summaries")
show_sentiment = st.button("Show Sentiment Analysis")
show_visualization = st.button("Show Visualizations")

if subreddit_name:
    with st.spinner("Fetching posts..."):
        posts = fetch_posts(subreddit_name, limit=limit)
        time.sleep(1)

    if not posts:
        st.error("No posts found. Please check the subreddit name and try again.")
        st.stop()

    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    st.success("Posts fetched successfully!")

    if show_summary:
        st.write(f"### Summarizing the top {limit} posts from r/{subreddit_name}:")
        with st.spinner("Generating summaries..."):
            summaries = []
            for post in posts:
                prompt = f"Summarize the following Reddit post and its top comments:\n{post}\n"
                summary, sentiment = generate_summary_and_sentiment(prompt)
                summaries.append({
                    "title": post['title'],
                    "post_summary": summary,
                    "sentiment": sentiment,
                    "url": post['url']
                })
            for summary in summaries:
                st.write(f"**Title:** {summary['title']}")
                st.write(f"**Post Summary:** {summary['post_summary']}")
                st.write(f"**Sentiment:** {summary['sentiment'] if summary['sentiment'] else 'Not detected'}")
                st.write(f"**URL:** [Read more]({summary['url']})")
                st.write("---")

    if show_sentiment:
        st.write("### Sentiment Analysis of the Subreddit")
        with st.spinner("Analyzing sentiment..."):
            sentiment_distribution = {"positive": 0, "neutral": 0, "negative": 0}
            for post in posts:
                text = f"{post.get('title', '')} {post.get('selftext', '')}".strip()
                if not text:
                    continue
                prompt = f"Analyze the sentiment of the following Reddit post:\n{text}"
                _, sentiment = generate_summary_and_sentiment(prompt)
                
                if sentiment:
                    if "positive" in sentiment.lower():
                        sentiment_distribution["positive"] += 1
                    elif "negative" in sentiment.lower():
                        sentiment_distribution["negative"] += 1
                    else:
                        sentiment_distribution["neutral"] += 1

            st.write("**Sentiment Distribution**")
            labels = list(sentiment_distribution.keys())
            counts = list(sentiment_distribution.values())
            fig, ax = plt.subplots()
            ax.bar(labels, counts, color=["green", "gray", "red"])
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Distribution in Posts")
            st.pyplot(fig)

            overall_sentiment = max(sentiment_distribution, key=sentiment_distribution.get)
            st.write(f"**Overall Sentiment:** {overall_sentiment}")

    if show_visualization:
        st.write("### Keywords and Word Cloud")
        with st.spinner("Generating word cloud..."):
            wordcloud = generate_word_cloud(posts)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
