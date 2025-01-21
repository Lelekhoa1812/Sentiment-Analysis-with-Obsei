# Sentiment Analysis with Obsei

This project leverages the power of the Obsei framework and Hugging Face Transformers to perform sentiment analysis on web articles. The extracted insights, key phrases, named entities, and summarized content provide a comprehensive understanding of the text.

## **Objective**
Analyze web articles for sentiment and extract actionable insights, including:
- Sentiment polarity (positive and negative scores).
- Key phrases and entities.
- Summarized content.

## **How It Works**
1. **Content Extraction**:
   - `TrafilaturaCrawlerSource` fetches the main text content from the specified URL.

2. **Sentiment Analysis**:
   - `TransformersSentimentAnalyzer` classifies the sentiment of the text using a pre-trained multilingual transformer model (`distilbert-base-multilingual-cased`).

3. **Additional Analysis**:
   - Key phrase extraction using NER.
   - Named Entity Recognition (NER) for identifying specific entities like people, organizations, and locations.
   - Summarization for concise representation of the content.

4. **Output**:
   - Results are written to a text file `output_example.txt` for further review.

## **Project Structure**
- `sentiment_analysis.py`: Main script for performing sentiment analysis and additional insights extraction.
- `output_example.txt`: Contains results of the analysis for the given article URL.

## **Technologies Used**
- **Obsei Framework**: Streamlined pipelines for data collection and analysis.
- **Hugging Face Transformers**: Pre-trained models for NER and summarization tasks.

## **Acknowledgments**
This project was developed as an internship product for **iNet Solution** to demonstrate the potential of AI-powered sentiment analysis and text processing.

## **How to Run**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/sentiment-analysis-obsei.git
   cd sentiment-analysis-obsei
   ```
2. Install required dependencies:
    ```bash
    pip install transformers obsei trafilatura
    ```

3. Run the script:
    ```bash
    python sentiment_analysis.py
    ```

Review the results in output_example.txt.
...

## **Future Enhancements**
- Fine-tune transformer models for domain-specific sentiment analysis.
- Automate integration with additional sources (e.g., RSS feeds, Twitter).
- Dynamically detect the language and apply direct language processor into the configs (now have to specify language manually). Solve this:  
WARNING - trafilatura.utils -   Language detector not installed, skipping detection
- Extend analysis for multilingual content.

