# Add the path to the locally cloned Obsei library
import sys
sys.path.insert(0, "/Users/khoale/Downloads/iNet/obsei/")

# Import necessary libraries
from obsei.analyzer.sentiment_analyzer import TransformersSentimentAnalyzer, TransformersSentimentAnalyzerConfig
from obsei.source.website_crawler_source import TrafilaturaCrawlerSource, TrafilaturaCrawlerConfig
from transformers import pipeline

# Step 1: Configure the web crawler source
"""
This step fetches article content from a specified URL.
We use the TrafilaturaCrawlerSource to extract structured content such as text, avoiding extraneous HTML or scripts.
"""
crawler_source = TrafilaturaCrawlerSource()
crawler_config = TrafilaturaCrawlerConfig(
    urls=["https://vnexpress.net/tong-thong-my-chua-ap-thue-trung-quoc-trong-ngay-dau-nhiem-ky-4841410.html"],  # Target URL
    target_language="vi",  # Vietnamese as the language
    include_comments=False,  # Focus on the main article content
)

def fetch_data(source, config):
    """Fetches the main content from the URL."""
    try:
        print("Fetching the article content...")
        return source.lookup(config)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

data = fetch_data(crawler_source, crawler_config)

# Step 2: Perform sentiment analysis
"""
Sentiment analysis is performed to classify the overall tone of the article. 
The TransformersSentimentAnalyzer applies a multilingual model for polarity classification (positive vs. negative).
"""
analyzer = TransformersSentimentAnalyzer(model_name_or_path="distilbert-base-multilingual-cased")
analyzer_config = TransformersSentimentAnalyzerConfig(
    labels=["positive", "negative"],  # Define the sentiment labels
    multi_class_classification=False,  # Single-label classification for simplicity
)
# Analyze the data from the article with given analyzer and config loaded
def analyze_data(analyzer, data, config):
    """Performs sentiment analysis on the fetched data."""
    try:
        print("Analyzing sentiment...")
        return analyzer.analyze_input(source_response_list=data, analyzer_config=config)
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return []

analyzed_data = analyze_data(analyzer, data, analyzer_config)

# Step 3: Additional Analyses
"""
In this step, we apply additional techniques to gain deeper insights:
1. Key Phrase Extraction: Identifies important concepts and themes in the text.
2. Named Entity Recognition (NER): Extracts structured information such as people, places, and organizations.
3. Summarization: Creates a concise summary of the article.
"""

# Truncate the text matching token allowance when using Facebook's Bart Large CNN summarization model with max 130 and min 30 token of allowance resource. Generally, recommend 1024
def truncate_text(text, max_length=1024):
    """Ensures the text length is compatible with models like summarization."""
    return text[:max_length]

# Key Phrase Extraction
def extract_key_phrases(text):
    """Identifies significant key phrases using a Named Entity Recognition model (bert-base-NER from dslim)."""
    try:
        print("Extracting key phrases...")
        key_phrase_model = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER")
        entities = key_phrase_model(text)
        key_phrases = [entity["word"] for entity in entities if entity["entity"].startswith("B")]
        print(f"Key Phrases: {', '.join(set(key_phrases))}")
        return key_phrases
    except Exception as e:
        print(f"Error extracting key phrases: {e}")
        return []

# Named Entity Recognition
def perform_ner(text):
    """Extracts named entities for structured understanding (using the bert-large-cased-finetuned-conll03-english model from dbmdz) """
    try:
        print("Performing Named Entity Recognition...")
        ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")
        entities = ner_model(text)
        for entity in entities:
            print(f"Entity: {entity['word']}, Type: {entity['entity']}")
        return entities
    except Exception as e:
        print(f"Error performing NER: {e}")
        return []

# Summarization
def summarize_text(text):
    """Generates a concise summary (using Facebook' Bart Large CNN model)."""
    try:
        text = truncate_text(text)  # Ensure compatible text length
        summarization_model = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
        summary = summarization_model(text, max_length=130, min_length=30, do_sample=False)
        print(f"Summary: {summary[0]['summary_text']}")
        return summary[0]["summary_text"]
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return "Summarization unavailable."

# Step 4: Write Analysis Results
output_file = "output_example.txt"
# Write to external file (txt)
def write_analysis(data, output_path):
    """Writes analysis results into a structured output file."""
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            for insight in data:
                file.write("----\n")
                file.write(f"Processed Text:\n{insight.processed_text}\n\n")

                # Sentiment Analysis
                sentiment = insight.segmented_data.get("classifier_data", {})
                file.write("Sentiment Analysis:\n")
                file.write(f"  Positive: {sentiment.get('positive', 0) * 100:.2f}%\n")
                file.write(f"  Negative: {sentiment.get('negative', 0) * 100:.2f}%\n\n")

                # Key Phrase Extraction
                key_phrases = extract_key_phrases(insight.processed_text)
                file.write("Key Phrases:\n")
                file.write(f"{', '.join(key_phrases)}\n\n")

                # Named Entity Recognition
                named_entities = perform_ner(insight.processed_text)
                file.write("Named Entities:\n")
                for entity in named_entities:
                    file.write(f"  {entity['word']} ({entity['entity']})\n")
                file.write("\n")

                # Summarization
                summary = summarize_text(insight.processed_text)
                file.write("Summary:\n")
                file.write(f"{summary}\n")
                file.write("----\n")
        print(f"Results written to {output_path}")
    except Exception as e:
        print(f"Error during analysis writing: {e}")

write_analysis(analyzed_data, output_file)
