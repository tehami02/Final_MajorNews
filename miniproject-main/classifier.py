from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_topic(text):
    candidate_labels = ["Politics", "Sports", "Technology", "Business", "Health", "Entertainment", "Science"]
    result = classifier(text, candidate_labels)
    return result['labels'][0]  # Top prediction
