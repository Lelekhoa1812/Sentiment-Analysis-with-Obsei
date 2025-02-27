
import sys
import os
import re
# For tonal mark removal
import unicodedata  
# Request to Google API (for fact checking) 
import requests 
# Upload API environment
from dotenv import load_dotenv
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
# Saving data to MongoDB
import datetime
from pymongo import MongoClient

app = Flask(__name__)

# Global variable to temporarily store the latest analysis result (used to save JSON to MongoDB)
latest_analysis = {} 

# Using Web Crawler with TrafilaturaClawler to directly observing data from a website url
# url = "https://vnexpress.net/tong-thong-my-chua-ap-thue-trung-quoc-trong-ngay-dau-nhiem-ky-4841410.html" # Random sample of Vietnamese (vi) article URL
# url = "https://www.theguardian.com/world/2025/jan/20/palestinians-search-gaza-missing-return-ruined-homes" # Random sample of English (en) article URL
# url = "https://www.marca.com/futbol/real-madrid/2025/02/05/real-madrid-acude-tribunales-obtener-videos-var-jugada-mbappe.html" # Random sample of Spanish (es) article URL
crawler_source = TrafilaturaCrawlerSource()
load_dotenv()  # Load all environment variables from .env file

##########################################
# Initialise MongoDB with environment    #
##########################################

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("❌ MongoDB URI (MONGO_URI) is missing! Please add it to your .env file.")
client = MongoClient(MONGO_URI)
# Setup db name as article URL
import hashlib
def get_db_for_url(article_url):
    """
    Returns a MongoDB database whose name is derived from an MD5 hash of the article URL.
    This guarantees that the database name is short enough (always 32 hex characters, plus any prefix).
    """
    # Compute MD5 hash of the URL
    hash_digest = hashlib.md5(article_url.encode('utf-8')).hexdigest()
    # Optionally, add a prefix (here "db_") so the database name is something like "db_<hash>"
    db_name = f"db_{hash_digest}"
    return client[db_name]


##########################################
# Functions for Text Preprocessing       #
##########################################

def remove_tonal_marks(text):
    """
    Removes a wide range of diacritical marks and tone modifiers from text.
    This function normalizes the text to NFD form (which separates base characters from their
    diacritical marks), removes combining marks in the range U+0300 to U+036F, and also strips
    additional modifier characters such as ˇ, ˘, ˙, ¨, ˆ, ˜, and `.
    Note: This operation is applied regardless of language; if you want to conditionally apply 
    it only to English text, wrap the call in an appropriate language-detection check.
    """
    # Normalize the text to NFD (decomposed form)
    normalized = unicodedata.normalize('NFD', text)
    # Remove all combining diacritical marks (Unicode range: U+0300 to U+036F)
    cleaned = re.sub(r'[\u0300-\u036F]', '', normalized)
    # Remove additional common tone/modifier symbols if needed
    # You can extend the character class below with any additional symbols as required.
    cleaned = re.sub(r"[ˇ˘˙¨ˆ˜`]", "", cleaned)
    # Optionally, normalize back to NFC (composed form)
    cleaned = unicodedata.normalize('NFC', cleaned)
    return cleaned

def truncate_text_for_model(text, max_length=1024):
    """Truncate text to the specified length (based on tokens or characters)"""
    # For simplicity, using character count here. In production, you might use a tokenizer.
    return text[:max_length]

def filter_text(text):
    """Removes unnecessary text like URLs and non-standard characters."""
    patterns_to_remove = [
        r"http\S+",         # URLs
        r"\[.*?\]",         # Text in brackets
        r"[^\w\s.,]",       # Non-standard characters
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text)
    return text.strip()


##########################################
# Raw and Clean Content Fetch Functions  #
##########################################

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

def detect_language(text):
    """Detects the language of the given text."""
    try:
        language = detect(text)
        print(f"Detected language: {language}")
        return language
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "unknown"

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


##########################################
# Sentiment Analysis Enhancements        #
##########################################

# Instantiate the sentiment analyzer (multilingual)
analyzer = TransformersSentimentAnalyzer(model_name_or_path="distilbert-base-multilingual-cased")
analyzer_config = TransformersSentimentAnalyzerConfig(
    labels=["positive", "negative"],
    multi_class_classification=False,
)
 
# Analyze sentiment per sentence
def analyze_data(analyzer, data, config):
    """Performs sentiment analysis on the fetched data."""
    try:
        print("Analyzing sentiment...")
        return analyzer.analyze_input(source_response_list=data, analyzer_config=config)
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return []

# Process big data, split and collect analysis
def analyze_sentence_sentiments(text, lang="en", article_key_phrases=None, extreme_threshold=0.8):
    """
    Splits the text into sentences (using '.' and ';' as delimiters),
    performs sentiment analysis on each sentence individually,
    computes the percentage of positive and negative sentences,
    and collects sentences with extremely high negativity.
    """
    sentences = re.split(r'[.;]\s*', text)
    total_sentences = 0
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    extreme_negative_sentences = []
    sentence_result = [] # To debug
    # Use the sentiment analysis pipeline for individual sentences.
    sentence_sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    for sentence in sentences:
        sentence = sentence.strip() # Split each to analyze
        if not sentence:
            continue
        total_sentences += 1
        result = sentence_sentiment_model(sentence)[0]
        label = result['label'].lower()  # "positive", "negative", or "neutral"
        score = result['score']  # confidence score between 0 and 1  
        # sentence_result.append({
        #     "sentence": sentence,
        #     "label": label,
        #     "score": score
        # }) 
        # print("Sentence List: ", sentence_result)
        # Append negative labels
        if label == "negative":
            negative_count += 1
            if score >= extreme_threshold:
                # Fact-check the extreme negative sentences using Google Fact Check API.
                is_fact = fact_check_sentence(sentence, key_phrases=article_key_phrases)
                # Store the sentence and its negative score post fact-checking
                extreme_negative_sentences.append({
                    "sentence": sentence,
                    "score": score,
                    "is_fact": str(is_fact) # Convert `True`/`False` to string
                })        
        elif label == "positive":
            positive_count += 1
        elif label == "neutral":
            neutral_count += 1
    # Compute overall pos/neg percentage
    positive_percent = (positive_count / total_sentences) * 100 if total_sentences else 0
    neutral_percent = (neutral_count / total_sentences) * 100 if total_sentences else 0
    negative_percent = (negative_count / total_sentences) * 100 if total_sentences else 0
    # Return percentage (4dp) level computed in overall and list of extreme negative sentences
    return {
        "positive": round(positive_percent, 4),
        "negative": round(negative_percent, 4),
        "neutral": round(neutral_percent, 4),
        "extreme_negative_sentences": extreme_negative_sentences
    }


##########################################
# NER and Key Phrase Extraction          #
##########################################

def filter_named_entities(entities):
    """Filters and merges named entities (merges subwords and consecutive same-type tokens)."""
    valid_entities = []
    current_entity = None
    for entity in entities:
        word = entity["word"].strip()
        if word.startswith("##"):
            if current_entity:
                current_entity["word"] += word[2:]
        else:
            if current_entity:
                valid_entities.append(current_entity)
            current_entity = entity.copy()
    if current_entity:
        valid_entities.append(current_entity)
    return merge_named_entities(valid_entities)

def merge_named_entities(entities):
    """Merges consecutive entities of the same type into a single entity and removes duplicates."""
    merged_entities = []
    previous_entity = None
    for entity in entities:
        if previous_entity and previous_entity["entity"] == entity["entity"]:
            previous_entity["word"] += f" {entity['word']}"
        else:
            if previous_entity:
                previous_entity["word"] = remove_duplicate_words(previous_entity["word"])
                merged_entities.append(previous_entity)
            previous_entity = entity
    if previous_entity:
        previous_entity["word"] = remove_duplicate_words(previous_entity["word"])
        merged_entities.append(previous_entity)
    return merged_entities

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

def extract_key_phrases(text):
    """Identifies significant key phrases using a Named Entity Recognition model."""
    try:
        print("Extracting key phrases...")
        key_phrase_model = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER")
        entities = key_phrase_model(text)
        entities = filter_named_entities(entities)
        key_phrases = {entity["word"] for entity in entities if entity["entity"].startswith("B")}
        # print(f"Key Phrases: {', '.join(key_phrases)}")
        return list(key_phrases)
    except Exception as e:
        print(f"Error extracting key phrases: {e}")
        return []

def perform_ner(text):
    """Extracts named entities using a pre-trained NER model."""
    try:
        print("Performing Named Entity Recognition...")
        ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")
        entities = ner_model(text)
        return entities
    except Exception as e:
        print(f"Error performing NER: {e}")
        return []

def get_persuasive_keywords(language):
    """Returns an extensive list of persuasive keywords for the given language."""
    keyword_table = {
        "en": [
            "should", "must", "need to", "urge", "important", "critical", "essential", "imperative", "vital", 
            "mandatory", "necessary", "indispensable", "require", "obligatory", "compulsory", "priority", 
            "fundamental", "crucial", "pivotal", "significant", "urgent", "key", "pressing", "core", 
            "recommend", "advise", "propose", "demand", "insist", "encourage", "plead", "advocate", 
            "support", "endorse", "stress", "highlight", "emphasize", "persuade", "convince"
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

def extract_persuasive_contexts(text):
    """Identifies sentences with persuasive or emotional language."""
    language = detect_language(text)
    keywords = get_persuasive_keywords(language)
    sentences = re.split(r'[.;]\s*', text)
    persuasive_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    return persuasive_sentences


##########################################
# Fact Checking Tool Function            #
##########################################

GOOGLE_FACT_CHECK_API_KEY = os.getenv("GFC_API_KEY")
# Check for OpenAI API key exist
if not GOOGLE_FACT_CHECK_API_KEY:
    raise ValueError("❌ Google Fact Check API key is missing! Add it to .env file or environment variables.")

def extract_key_facts(sentence, key_phrases=None):
    """
    Extracts key factual segments from the sentence.
    First, it uses a regex to capture segments that start with a number
    (with optional commas) and include up to 7 following tokens.
    Then, if a list of key_phrases is provided, it also searches for each key
    phrase in the sentence and extracts up to 7 non-space tokens before and after
    the key phrase.
    Returns a list of extracted key facts (duplicates are removed).
    """
    key_facts = []
    # Extract numeric facts using regex
    numeric_matches = re.findall(r'(\d[\d,]*(?:\s+\S+){0,7})', sentence)
    for match in numeric_matches:
        if len(match.split()) > 1:
            key_facts.append(match.strip())
    # If key_phrases are provided, extract contextual segments for each phrase.
    if key_phrases:
        for phrase in key_phrases:
            # Build a regex pattern to capture up to 7 tokens before and after the key phrase.
            # The pattern breaks down as:
            #   ((?:\S+\s+){0,7})   --> Capture up to 7 tokens before
            #   (re.escape(phrase)) --> The key phrase (escaped for regex safety)
            #   ((?:\s+\S+){0,7})   --> Capture up to 7 tokens after
            pattern = r'((?:\S+\s+){0,7})(' + re.escape(phrase) + r')((?:\s+\S+){0,7})'
            for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
                before = match.group(1).strip()
                mid = match.group(2).strip()
                after = match.group(3).strip()
                # Concatenate the tokens (if available) with the key phrase in the middle.
                fact_context = (before + " " if before else "") + mid + (" " + after if after else "")
                key_facts.append(fact_context.strip())
    # Remove duplicate entries
    key_facts = list(set(key_facts))
    return key_facts

def fact_check_sentence(sentence, key_phrases=None, api_key=GOOGLE_FACT_CHECK_API_KEY):
    """
    Fact-checks a given sentence using the Google Fact Check Tools API.
    Instead of simply joining the key phrases with the sentence,
    this function extracts key facts using `extract_key_facts()`, which
    captures both numeric segments and the context surrounding any key phrases
    (up to 7 non-space tokens before and after).
    The resulting extracted text is URL-encoded and sent as the query to the API.
    If any fact-checked claims are returned, the function returns True;
    otherwise, it returns False.
    """
    # Extract key facts from the sentence using the provided key phrases for context.
    key_facts = extract_key_facts(sentence, key_phrases=key_phrases)
    # Build the query string by joining the extracted segments.
    query_str = " ".join(key_facts)
    if not query_str:
        print("Empty query string generated; skipping fact check.")
        return None
    # URL-encode the query string
    query = requests.utils.quote(query_str)
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # print("Keyfact query:", query_str, True if "claims" in data and data["claims"] else False)
            return True if "claims" in data and data["claims"] else False
        else:
            print(f"Fact Check API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error during fact check API call: {e}")
        return None


##########################################
# Summarization Functions                #
##########################################

def summarize_text(text, lang="en"):
    """Generates a concise summary using Facebook's Bart Large CNN model."""
    try:
        print("Summarizing text with Bart...")
        text = truncate_text_for_model(text)  # Ensure compatible text length
        input_length = len(text.split())
        if input_length < 50:
            print("Input too short for summarization. Returning input as summary.")
            return text
        max_summary_length = min(1024, max(input_length // 2, 50))
        min_summary_length = min(max_summary_length // 2, 30)
        if lang == "vi":
            # Use a Vietnamese summarization model
            summarization_model = pipeline("summarization", model="VietAI/vit5-large-vietnews-summarization", tokenizer="VietAI/vit5-base")
        else:
            summarization_model = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
        summary = summarization_model(
            text,
            max_length=max_summary_length,
            min_length=min_summary_length,
            do_sample=False,
            num_beams=4,
            length_penalty=1.5
        )
        # print(f"Summary: {summary[0]['summary_text']}")
        return summary[0]["summary_text"]
    except IndexError:
        print("IndexError during summarization. Returning fallback truncated text.")
        return truncate_text_for_model(text, 512)
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return "Summarization unavailable."

def enhanced_summarize_text(text, lang="en"):
    """
    Generates an enhanced summary using a model with a higher token allowance (e.g., Longformer).
    Note: Replace with your preferred long-document summarization model if available.
    """
    try:
        print("Summarizing text with enhanced model (Longformer)...")
        text = truncate_text_for_model(text, max_length=3500)  # Longformer can accept more tokens
        if lang == "vi":
            summarization_model = pipeline("summarization", model="VietAI/vit5-large-vietnews-summarization", tokenizer="VietAI/vit5-large-vietnews-summarization")
        else:
            summarization_model = pipeline("summarization", model="allenai/longformer-base-4096", tokenizer="allenai/longformer-base-4096")
        summary = summarization_model(
            text,
            max_length=300,
            min_length=50,
            do_sample=False,
            num_beams=4,
        )
        return summary[0]["summary_text"]
    except Exception as e:
        print(f"Error in enhanced summarization: {e}")
        return summarize_text(text, lang=lang)  # Fallback to the standard summarizer


##########################################
# Writing Analysis Results               #
##########################################

output_file = "obsei_analysis.txt"

# Write analysis to txt
def write_analysis(data, output_path, sentiment, persuasive_contexts, summary, sentence_sentiment):
    """Writes analysis results into an output file."""
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            print("Writing analysis results...")
            for insight in data:
                # Optionally remove tonal marks for English articles
                processed_text = filter_text(insight.processed_text)
                language = detect_language(processed_text)
                if language == "en":
                    processed_text = remove_tonal_marks(processed_text)
                # Start writing
                file.write("----\n")
                file.write(f"Processed Text:\n{processed_text}\n\n")
                # Sentiment chart section
                file.write("Sentiment Analysis:\n")
                # file.write(f"  Positive: {sentiment.get('positive', 0) * 100:.2f}%\n")
                # file.write(f"  Negative: {sentiment.get('negative', 0) * 100:.2f}%\n\n")
                file.write(f"  Positive: {sentence_sentiment['positive']:.2f}%\n")
                file.write(f"  Negative: {sentence_sentiment['negative']:.2f}%\n\n")
                # Key phrases section
                key_phrases = extract_key_phrases(processed_text)
                file.write("Key Phrases:\n")
                file.write(f"{', '.join(key_phrases)}\n\n")
                # Named entities section
                named_entities = perform_ner(processed_text)
                cleaned_entities = filter_named_entities(named_entities)
                file.write("Named Entities:\n")
                for entity in cleaned_entities:
                    file.write(f"  {entity['word']} ({entity['entity']})\n")
                file.write("\n")
                # Persuasive context section
                file.write("Persuasive Contexts:\n")
                file.write("\n".join(persuasive_contexts) + "\n")
                file.write("----\n\n")
                # Extreme negative content section
                file.write("Extreme Negative Sentences:\n")
                for s in sentence_sentiment["extreme_negative_sentences"]:
                    file.write(f"  {s}\n")
                file.write("----\n\n")
                # Summary section
                file.write("Summary:\n")
                file.write(f"{summary}\n")
                file.write("----\n\n")
        print(f"Results written to {output_path}")
    except Exception as e:
        print(f"Error writing analysis: {e}")


##########################################
# Flask Routes                           #
##########################################

@app.route("/")
def index():
    """Render the main HTML page."""
    return render_template("index.html")

@app.route("/analyze", methods=["GET", "POST"])
def main():
    if request.method == "GET":
        # For GET requests, simply render the index page
        return render_template("index.html")
    # Otherwise, handle POST requests:
    url = request.form.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        raw_text = fetch_raw_content(url)
        if raw_text:
            detected_language = detect_language(raw_text)
            # For English content, remove tonal marks from the raw text
            if detected_language == "en":
                raw_text = remove_tonal_marks(raw_text)
            data = fetch_cleaned_content(url, detected_language)
        else:
            print("Unable to fetch raw content for language detection.")
            data = []
        if data:
            # Split text into sentences using '.' and ';' as delimiters.
            analyzed_data = analyze_data(analyzer, data, analyzer_config)
            sentiment = analyzed_data[0].segmented_data.get("classifier_data", {})
            # Analyse text
            analyzed_text = analyzed_data[0].processed_text
            processed_text = filter_text(analyzed_text)
            if detected_language == "en":
                processed_text = remove_tonal_marks(processed_text)
            # Key phrases
            key_phrases = extract_key_phrases(processed_text)
            named_entities = perform_ner(processed_text)
            named_entities_cleaned = filter_named_entities(named_entities)
            persuasive_contexts = extract_persuasive_contexts(processed_text)
            summary = enhanced_summarize_text(processed_text, lang=detected_language)
            # Perform sentence-level sentiment analysis for more granular evaluation.
            sentence_sentiment = analyze_sentence_sentiments(processed_text, lang=detected_language, article_key_phrases=key_phrases)
            # Write result to file
            write_analysis(analyzed_data, output_file, sentiment, persuasive_contexts, summary, sentence_sentiment)
            # Prepare JSON response (including extreme negative sentences has been fact checked) - Removed since not using Facebook model anymore
            # positive = sentiment.get("positive", 0) * 100
            # negative = sentiment.get("negative", 0) * 100
            # Prepare the JSON result to be sent to the frontend.
            analysis_result = {
                "article_url": url, 
                "language": detected_language,
                "positive": sentence_sentiment["positive"],
                "negative": sentence_sentiment["negative"],
                "neutral": sentence_sentiment["neutral"],
                "key_phrases": key_phrases,
                "named_entities": [entity["word"] for entity in named_entities_cleaned],
                "persuasive_contexts": persuasive_contexts,
                "summary_contexts": summary,
                "extreme_negative_sentences": sentence_sentiment["extreme_negative_sentences"]
            }
            # Store the analysis result in a global variable for later saving (e.g., save to DB).
            global latest_analysis
            latest_analysis = analysis_result
            # Send json result object
            return jsonify(analysis_result)
        else:
            print("No data available for analysis.")
            return jsonify({"error": "No data available for analysis"}), 500
    except Exception as e:
        print(f"Error in /analyze: {e}")
        return jsonify({"error": str(e)}), 500

# Download txt file route
@app.route("/download", methods=["GET"])
def download():
    """Download the analysis file."""
    return send_file(output_file, as_attachment=True, download_name="obsei_analysis.txt")

# Save data to MongoDB route
@app.route("/save", methods=["GET"])
def save_to_db():
    """
    Saves the latest analysis result (JSON) to MongoDB.
    The saved document will include all the fields from the analysis result along with a UTC timestamp.
    """
    global latest_analysis
    if not latest_analysis:
        return jsonify({"error": "No analysis data available to save."}), 400
    article_url = latest_analysis.get("article_url")
    if not article_url:
        return jsonify({"error": "No article URL found in analysis."}), 400
    try:
        # Use the helper function to get a database using the article URL.
        db = get_db_for_url(article_url)
        collection = db.analysis  # Use a fixed collection name (e.g., "analysis")
        # Create a document from the latest analysis data and add a timestamp.
        doc = latest_analysis.copy()
        doc["timestamp"] = datetime.datetime.utcnow()
        result = collection.insert_one(doc)
        return jsonify({
            "message": "Analysis saved to MongoDB",
            "id": str(result.inserted_id)
        })
    except Exception as e:
        print(f"Error saving analysis to MongoDB: {e}")
        return jsonify({"error": f"Failed to save analysis to MongoDB: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5002)
