import praw
import re
from transformers import pipeline
from collections import Counter
from wordcloud import WordCloud

# Replace with your Reddit API credentials
CLIENT_ID = ""
CLIENT_SECRET = ""
USER_AGENT = ""

# Authenticate
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Fetch the latest posts from a subreddit
def fetch_posts(subreddit_name, limit=5):
    """
    Fetch the latest posts from a subreddit.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.new(limit=limit):
        posts.append({
            "title": post.title,
            "body": post.selftext,
            "url": post.url,
            "comments": fetch_comments(post)  # Fetch comments for each post
        })
    return posts

def fetch_comments(post, comment_limit=10):
    """
    Fetch comments for a post.
    """
    post.comments.replace_more(limit=0)  # Avoid loading too many comments
    comments = []
    for comment in post.comments[:comment_limit]:
        comments.append(comment.body)
    return comments

def clean_text(text):
    """
    Clean the text by removing special characters, URLs, and extra spaces.
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove special characters (keep only alphanumeric and basic punctuation)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    # Remove extra spaces
    text = " ".join(text.split())
    return text

def summarize_text(text, summarizer, max_length=50, min_length=25):
    """
    Summarize text using the provided summarizer.
    """
    text = clean_text(text)
    if len(text.split()) < 5:  # Skip if the text is too short
        return text
    if len(text.split()) > 1024:  # Truncate if the text is too long
        text = " ".join(text.split()[:1024])
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return text

def summarize_post(post, summarizer):
    """
    Summarize a post's title and body using the provided summarizer.
    """
    text = f"{post['title']}. {post['body']}"
    return summarize_text(text, summarizer)

def summarize_comments(comments, summarizer):
    """
    Summarize a list of comments using the provided summarizer.
    """
    combined_comments = " ".join(comments)
    return summarize_text(combined_comments, summarizer)

def analyze_sentiment(text, sentiment_analyzer):
    """
    Analyze the sentiment of a text using a pretrained model.
    """
    result = sentiment_analyzer(text)[0]
    return result["label"], result["score"]  # Returns label (POSITIVE/NEGATIVE) and confidence score

def get_sentiment_distribution(posts, sentiment_analyzer):
    """
    Get sentiment distribution for all posts and comments.
    """
    sentiments = []
    for post in posts:
        # Analyze sentiment of the post
        post_label, post_score = analyze_sentiment(post["body"], sentiment_analyzer)
        sentiments.append(post_label)
        
        # Analyze sentiment of the comments
        for comment in post["comments"]:
            comment_label, comment_score = analyze_sentiment(comment, sentiment_analyzer)
            sentiments.append(comment_label)
    
    # Count sentiment labels
    sentiment_distribution = Counter(sentiments)
    return sentiment_distribution

def generate_word_cloud(posts):
    """
    Generate a word cloud from posts and comments.
    """
    combined_text = " ".join([post["body"] for post in posts])
    combined_text += " ".join([" ".join(post["comments"]) for post in posts])
    
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
    return wordcloud

def summarize_posts(posts, summarizer):
    """
    Summarize a list of posts and their comments using the provided summarizer.
    """
    summaries = []
    for post in posts:
        post_summary = summarize_post(post, summarizer)
        comment_summary = summarize_comments(post["comments"], summarizer)
        summaries.append({
            "title": post["title"],
            "post_summary": post_summary,
            "comment_summary": comment_summary,
            "url": post["url"]
        })
    return summaries
