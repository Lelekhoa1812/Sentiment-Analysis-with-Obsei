import sys
import os
import re
# Init a Flask server
from flask import Flask, request, jsonify, send_file, render_template
sys.path.insert(0, os.path.join(os.getcwd(), "obsei"))
# Detecting language
from langdetect import detect
# Import Obsei-specific modules
from obsei.analyzer.sentiment_analyzer import TransformersSentimentAnalyzer, TransformersSentimentAnalyzerConfig
# Source scrapper, crawler and observer modules
from obsei.source.reddit_source import RedditConfig, RedditSource, RedditCredInfo
from obsei.source.website_crawler_source import TrafilaturaCrawlerSource, TrafilaturaCrawlerConfig
# Pipeline module from transformer for NER and summarization utils
from transformers import pipeline
# Summary specific model for summary and Vietnamese language
from transformers import MBartForConditionalGeneration, MBartTokenizer  # mBART for summarization
from transformers import MBart50Tokenizer # mBART50 for multilingual integration

app = Flask(__name__)

# Visit https://www.obsei.com/ to find more on configurations of different sources (e.g., Facebook, Youtube, Reddit)

# Using Web Crawler with TrafilaturaClawler to directly observing data from a website url
# url = "https://vnexpress.net/tong-thong-my-chua-ap-thue-trung-quoc-trong-ngay-dau-nhiem-ky-4841410.html" # Random sample of Vietnamese article URL
# url = "https://www.theguardian.com/world/2025/jan/20/palestinians-search-gaza-missing-return-ruined-homes" # Random sample of English article URL
crawler_source = TrafilaturaCrawlerSource()

#### Step 1: Configure the web crawler source
# a. Fetch raw content for language detection
def fetch_raw_content(url):
    """Fetches raw article content for language detection."""
    try:
        print("Fetching raw article content for language detection...")
        raw_config = TrafilaturaCrawlerConfig(urls=[url], include_comments=False)
        raw_data = crawler_source.lookup(raw_config)
        if raw_data and raw_data[0].processed_text:
            return raw_data[0].processed_text
        else:
            print("No content fetched for language detection.")
            return None
    except Exception as e:
        print(f"Error fetching raw content: {e}")
        return None

# b. Detect language dynamically
def detect_language(text):
    """Detects the language of the given text."""
    try:
        print(f"Detected language to be {detect(text)}")
        return detect(text)
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "unknown"

"""
This step fetches article content from a specified URL.
We use the TrafilaturaCrawlerSource to extract structured content such as text, avoiding extraneous HTML or scripts.
"""
# c. Set Web Crawler configs. Fetch cleaned content based on detected language
def fetch_cleaned_content(url, language):
    """Fetches cleaned article content with language-specific settings."""
    try:
        print(f"Fetching article content with detected language: {language}")
        config = TrafilaturaCrawlerConfig(
            urls=[url],
            target_language=language,
            include_comments=False,
        )
        data = crawler_source.lookup(config)
        return data
    except Exception as e:
        print(f"Error fetching cleaned content: {e}")
        return []
####

#### Step 2: Perform sentiment analysis
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
####

#### Step 3: Article Analyses and Data Processing
"""
In this step, we apply additional techniques to gain deeper insights:
1. Key Phrase Extraction: Identifies important concepts and themes in the text.
2. Named Entity Recognition (NER): Extracts structured information such as people, places, and organizations.
3. Summarization: Creates a concise summary of the article.
"""

# Helper Functions
# Removing unnecessary text
def filter_text(text):
    """Removes unnecessary text like URLs, non-standard characters."""
    patterns_to_remove = [
        r"http\S+",         # URLs
        r"\[.*?\]",         # Text in brackets
        r"[^\w\s.,]",       # Non-standard characters
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text)
    return text.strip()

# Remove unnecessary name entities
def filter_named_entities(entities):
    """Filters out incomplete or fragmented named entities and merges subwords."""
    valid_entities = []
    current_entity = None
    for entity in entities:
        word = entity["word"].strip()
        if word.startswith("##"):  # Subword continuation
            if current_entity:
                # Merge subword into the previous entity
                current_entity["word"] += word[2:]
        else:
            if current_entity:  # Save the completed entity
                valid_entities.append(current_entity)
            current_entity = entity.copy()  # Start a new entity
    if current_entity:  # Append the last entity if present
        valid_entities.append(current_entity)
    return merge_named_entities(valid_entities)
# Merging consecutive entities before determining which that would be split
def merge_named_entities(entities):
    """Merges consecutive entities of the same type into a single entity and removes duplicates."""
    merged_entities = []
    previous_entity = None
    for entity in entities:
        if previous_entity and previous_entity["entity"] == entity["entity"]:
            # Merge consecutive entities of the same type
            previous_entity["word"] += f" {entity['word']}"
        else:
            if previous_entity:
                # Remove duplicates within the entity's word
                previous_entity["word"] = remove_duplicate_words(previous_entity["word"])
                merged_entities.append(previous_entity)
            previous_entity = entity
    if previous_entity:
        # Remove duplicates for the last entity
        previous_entity["word"] = remove_duplicate_words(previous_entity["word"])
        merged_entities.append(previous_entity)
    return merged_entities
# Remove duplication from the merged listing
def remove_duplicate_words(text):
    """Removes duplicate words from a string."""
    words = text.split()
    seen = set()
    unique_words = []
    for word in words:
        if word.lower() not in seen:
            unique_words.append(word)
            seen.add(word.lower())
    return " ".join(unique_words)

# Truncate text to appropriate length matching the summarizer token allowance
def truncate_text(text, max_length=1024): # Allowance of 1024
    """Ensures text length is compatible with models."""
    return text[:max_length]

# Key Phrase Extraction
def extract_key_phrases(text):
    """Identifies significant key phrases using a Named Entity Recognition model."""
    try:
        print("Extracting key phrases...")
        key_phrase_model = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER")
        entities = key_phrase_model(text)
        entities = filter_named_entities(entities)
        key_phrases = {entity["word"] for entity in entities if entity["entity"].startswith("B")}
        print(f"Key Phrases: {', '.join(key_phrases)}")
        return list(key_phrases)
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
        # for entity in entities:
        #     print(f"Entity: {entity['word']}, Type: {entity['entity']}")
        return entities
    except Exception as e:
        print(f"Error performing NER: {e}")
        return []
    
# Look up for the usage of persuasive languages that has been applied in any language (add more as it should)
def get_persuasive_keywords(language):
    """Returns an extensive list of persuasive keywords for the given language."""
    keyword_table = {
        "en": [
            "should", "must", "need to", "urge", "important", "critical", "essential", "imperative", "vital", 
            "mandatory", "necessary", "indispensable", "require", "obligatory", "compulsory", "priority", 
            "fundamental", "crucial", "pivotal", "significant", "urgent", "key", "pressing", "core", 
            "recommend", "advise", "propose", "demand", "insist", "encourage", "plead", "advocate", 
            "support", "endorse", "stress", "highlight", "emphasize", "persuade", "convince", 
            "appeal", "assert", "claim", "argue", "justify", "prove", "validate", "affirm", 
            "guarantee", "assure", "ensure", "certify", "commit", "pledge", "promise"
        ],
        "vi": [
            "nên", "cần phải", "quan trọng", "thiết yếu", "cấp bách", "không thể thiếu", "bắt buộc", 
            "yêu cầu", "bắt buộc phải", "gấp rút", "trọng điểm", "chủ chốt", "ưu tiên", "đề xuất", 
            "đề nghị", "khuyến nghị", "khuyến khích", "khích lệ", "ủng hộ", "cam kết", "nhấn mạnh", 
            "chứng minh", "đảm bảo", "xác nhận", "bảo đảm", "đề cao", "bổ sung", "giải pháp", 
            "tăng cường", "phải làm", "giảm thiểu", "thực hiện ngay", "không được trì hoãn", 
            "lợi ích", "đem lại", "tác động lớn", "hiệu quả", "đáng kể", "chủ trương", "cơ bản", 
            "giải quyết", "phương án", "không thể bỏ qua", "đáng để ý", "tối quan trọng", 
            "cực kỳ cần thiết", "tầm nhìn", "phát triển bền vững", "không được coi nhẹ"
        ],
    }
    return keyword_table.get(language, [])
# Extracting persuasive contexts from keyword dictionary
def extract_persuasive_contexts(text):
    """Identifies sentences with persuasive or emotional language."""
    language = detect_language(text)
    keywords = get_persuasive_keywords(language)
    sentences = text.split(". ")
    persuasive_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    return persuasive_sentences

# Generic summary
def summarize_text(text):
    """Generates a concise summary using Facebook's Bart Large CNN model."""
    try:
        print("Summarizing text...")
        text = truncate_text(text)  # Ensure compatible text length
        input_length = len(text.split())  # Calculate input length
        min_input_length = 50  # Define a minimum length threshold for summarization
        if input_length < min_input_length:
            print("Input text is too short for summarization. Returning input as summary.")
            return text  # Use input as summary for very short text
        max_summary_length = min(1024, max(input_length // 2, 50))  # Adjust dynamic max_length
        min_summary_length = min(max_summary_length // 2, 30)  # Ensure min_length is reasonable
        # Summarization pipeline
        summarization_model = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
        summary = summarization_model(
            text,
            max_length=max_summary_length,
            min_length=min_summary_length,
            do_sample=False,
            num_beams=4,
            length_penalty=1.5
        )
        print(f"Summary: {summary[0]['summary_text']}")
        return summary[0]["summary_text"]
    except IndexError:
        print("IndexError: Likely due to tokenization or text length mismatch. Returning truncated input as fallback.")
        return truncate_text(text, 512)  # Fallback: truncate to first 1024 tokens
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return "Summarization unavailable."

# Vietnamese-specific summary utilities 
# def summarize_text(text):
#     """Summarizes Vietnamese text using mBART50."""
#     try:
#         print("Summarizing text...")
#         model_name = "facebook/mbart-large-50"
#         tokenizer = MBart50Tokenizer.from_pretrained(model_name)
#         model = MBartForConditionalGeneration.from_pretrained(model_name)
#         # Preprocess text for mBART
#         inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
#         summary_ids = model.generate(inputs.input_ids, max_length=500, min_length=50, length_penalty=2.0, num_beams=4)
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         return summary
#     except Exception as e:
#         print(f"Error summarizing Vietnamese text: {e}")
#         return "Summarization unavailable."
####
  
#### Step 4: Write Analysis Results
output_file = "output_example.txt"
# Write to external file (txt)
def write_analysis(data, output_path):
    """Writes analysis results into a structured output file."""
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            for insight in data:
                processed_text = filter_text(insight.processed_text)
                file.write("----\n")
                file.write(f"Processed Text:\n{processed_text}\n\n")
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
                named_entities = perform_ner(processed_text)
                cleaned_entities = filter_named_entities(named_entities)
                file.write("Named Entities:\n")
                for entity in cleaned_entities:
                    file.write(f"  {entity['word']} ({entity['entity']})\n")
                file.write("\n\n")
                # Persuasive Contexts
                persuasive_contexts = extract_persuasive_contexts(processed_text)
                file.write("Persuasive Contexts:\n")
                file.write("\n".join(persuasive_contexts) + "\n")
                file.write("----\n\n")
                # Summarization
                summary = summarize_text(processed_text)
                file.write("Summary:\n")
                file.write(f"{summary}\n")
                file.write("----\n\n")
        print(f"Results written to {output_path}")
        return processed_text, sentiment, key_phrases, cleaned_entities, persuasive_contexts, summary
    except Exception as e:
        print(f"Error during analysis writing: {e}")
####

# Main Execution
output_file = "obsei_analysis.txt"
# Flask Routes
@app.route("/")
def index():
    """Render the main HTML page."""
    return render_template("index.html")
@app.route("/analyze", methods=["POST"])
def main():
    url = request.form.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        raw_text = fetch_raw_content(url)
        # If raw text (before applying configs) can be fetched
        if raw_text:
            detected_language = detect_language(raw_text)
            data = fetch_cleaned_content(url, detected_language)
        else:
            print("Unable to fetch raw content for language detection.")
            data = []  # Ensure `data` is always defined
        # Check if data has valid content before proceeding
        if data:
            analyzed_data = analyze_data(analyzer, data, analyzer_config)
            processed_text, sentiment, key_phrases, cleaned_entities, persuasive_contexts, summary = write_analysis(analyzed_data, output_file)
            # Get sentiment data directly and fit to the json body sending to the frontend for pie-chart visualization
            positive = sentiment.get("positive", 0) * 100
            negative = sentiment.get("negative", 0) * 100
            return jsonify({
                "message": "Analysis complete",
                "positive": positive,
                "negative": negative,
                "key_phrases": key_phrases,
                "named_entities": cleaned_entities,
                "persuasive_contexts": persuasive_contexts,
                "summary_contexts": summary,
                "download_url": "/download"
            })
        else:
            print("No data available for analysis.")
            return jsonify({"error": "No data available for analysis"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Download routing 
@app.route("/download", methods=["GET"])
def download():
    """Download the analysis file."""
    return send_file(output_file, as_attachment=True, download_name="obsei_analysis.txt")

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True, port=5002)