# Add the path to the locally cloned Obsei library
import sys
sys.path.insert(0, "/Users/khoale/Downloads/iNet/obsei/")

# Import necessary library usages
from obsei.analyzer.sentiment_analyzer import TransformersSentimentAnalyzer, TransformersSentimentAnalyzerConfig
from obsei.source.website_crawler_source import TrafilaturaCrawlerSource, TrafilaturaCrawlerConfig
from transformers import pipeline

# Step 1: Configure the web crawler source
"""
The web crawler source is responsible for fetching the content of the article from the provided URL.
TrafilaturaCrawlerSource is used to extract content from web pages.
"""
crawler_source = TrafilaturaCrawlerSource()
crawler_config = TrafilaturaCrawlerConfig(
    urls=["https://vnexpress.net/tong-thong-my-chua-ap-thue-trung-quoc-trong-ngay-dau-nhiem-ky-4841410.html"],  # Target URL
    target_language="vi",  # Specify Vietnamese as the target language
    include_comments=False,  # Exclude comments from the article content
)

# Attempt to fetch the text data from provied source and load the configs with web crawler
def fetch_data(source, config):
    """
    Fetches data (content) from the configured web source.
    """
    try:
        print("Fetching the article content...")
        return source.lookup(config)  # Returns a list of extracted data objects
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []
data = fetch_data(crawler_source, crawler_config)

# Step 2: Configure the Transformers sentiment analyzer
# This analyzer uses a multilingual transformer model for performing sentiment analysis.
analyzer = TransformersSentimentAnalyzer(model_name_or_path="distilbert-base-multilingual-cased")
# Configs
analyzer_config = TransformersSentimentAnalyzerConfig(
    labels=["positive", "negative"],  # Specify sentiment labels
    multi_class_classification=False,  # Use single-label classification
)

# Provide sentiment analysis with analyzer and config loaded from step 1 and 2, the processing with the article data
def analyze_data(analyzer, data, config):
    """
    Performs sentiment analysis on the fetched data.
    """
    try:
        print("Analyzing sentiment...")
        return analyzer.analyze_input(source_response_list=data, analyzer_config=config)
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return []
analyzed_data = analyze_data(analyzer, data, analyzer_config)

# Step 3: Show result to the terminal
def output_results(data):
    """
    Outputs the sentiment analysis results in a readable format.
    """
    try:
        print("\nAnalysis Results:")
        for insight in data:
            print("----") # START 
            print(f"Processed Text: {insight.processed_text}")
            print(f"Sentiment Data: {insight.segmented_data.get('classifier_data', {})}")
            print("----") # END
    except Exception as e:
        print(f"Error outputting results: {e}")
output_results(analyzed_data)

# Step 4: Perform additional analysis
"""
We will perform the following:
1. Extract key phrases to identify the main topics discussed in the article.
2. Perform Named Entity Recognition (NER) to extract specific entities (e.g., people, organizations, places).
3. Generate a summary of the article for a high-level overview.
"""
print("\nPerforming Additional Analysis...\n")

# Key Phrase Extraction
def extract_key_phrases(text):
    """
    Extracts key phrases from the text using Hugging Face's token classification pipeline.
    """
    try:
        print("Extracting key phrases...")
        key_phrase_model = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER")
        entities = key_phrase_model(text)
        key_phrases = [entity["word"] for entity in entities if entity["entity"].startswith("B")]
        print(f"Key Phrases: {', '.join(set(key_phrases))}")
    except Exception as e:
        print(f"Error extracting key phrases: {e}")

# Named Entity Recognition (NER)
def perform_ner(text):
    """
    Performs Named Entity Recognition to extract named entities (e.g., people, organizations, places).
    """
    try:
        print("Performing Named Entity Recognition...")
        ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")
        entities = ner_model(text)
        for entity in entities:
            print(f"Entity: {entity['word']}, Type: {entity['entity']}")
    except Exception as e:
        print(f"Error performing NER: {e}")

# Text Summarization
def summarize_text(text):
    """
    Summarizes the article content to provide a high-level overview.
    """
    try:
        print("Generating Summary...")
        summarization_model = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
        summary = summarization_model(text, max_length=130, min_length=30, do_sample=False)
        print(f"Summary: {summary[0]['summary_text']}")
    except Exception as e:
        print(f"Error summarizing text: {e}")

# Perform additional analyses on the processed text
if analyzed_data:
    processed_text = analyzed_data[0].processed_text  # Extract the main text from the analysis
    extract_key_phrases(processed_text)
    perform_ner(processed_text)
    summarize_text(processed_text)

# Step 5: Perform additional analysis and write results to an external file (txt)
output_file = "output_example.txt"
def perform_additional_analysis(data, output_path):
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            for insight in data:
                file.write("----\n")
                file.write(f"Processed Text:\n{insight.processed_text}\n\n")
                file.write(f"Sentiment Data:\n{insight.segmented_data.get('classifier_data', {})}\n\n")

                # Key Phrase Extraction
                file.write("Key Phrases:\n")
                key_phrase_model = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER")
                entities = key_phrase_model(insight.processed_text)
                key_phrases = [entity["word"] for entity in entities if entity["entity"].startswith("B")]
                file.write(f"{', '.join(set(key_phrases))}\n\n")

                # Named Entity Recognition
                file.write("Named Entities:\n")
                ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")
                entities = ner_model(insight.processed_text)
                for entity in entities:
                    file.write(f"Entity: {entity['word']}, Type: {entity['entity']}\n")
                file.write("\n")

                # Text Summarization
                file.write("Summary:\n")
                summarization_model = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
                summary = summarization_model(insight.processed_text, max_length=130, min_length=30, do_sample=False)
                file.write(f"{summary[0]['summary_text']}\n")
                file.write("\n")
        print(f"Results written to {output_path}")
    except Exception as e:
        print(f"Error during additional analysis: {e}")

perform_additional_analysis(analyzed_data, output_file)
