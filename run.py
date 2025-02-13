from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
device = "cpu"
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)

# Save the model and tokenizer
summarizer.model.save_pretrained("saved_model1")
summarizer.tokenizer.save_pretrained("saved_model1")

# Load the saved model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("saved_model1")
tokenizer = AutoTokenizer.from_pretrained("saved_model1")

# Initialize the summarizer pipeline with the loaded model and tokenizer
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)