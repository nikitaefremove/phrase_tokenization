# Phrase Similarity Analyzer

Phrase Similarity Analyzer is a tool that helps in finding phrases in a text sample that are similar to a predefined set of standard phrases using cosine similarity with BERT embeddings.

## Requirements

Before running the script, make sure to install the necessary Python packages. You can install them with the following command:

```
fastapi==0.103.1
nltk==3.8.1
numpy==1.26.0
pydantic==2.3.0
scikit-learn==1.3.0
scipy==1.11.2
spacy==3.6.1
torch==2.0.1
tqdm==4.66.1
transformers==4.33.2
uvicorn==0.23.2

```

## Usage

To use the Phrase Similarity Analyzer, run the script in a Python environment 
where all the necessary packages are installed. 
You will be prompted to enter a text sample, 
and the script will then analyze the sample 
and print any phrases that are similar to the standard phrases.

```
python script_name.py
```

## How It Works

The script tokenizes the input text into sentences using SpaCy.
It creates phrases from each sentence using a sliding window approach.
The script then computes BERT embeddings for each phrase in the input text and each standard phrase.
It computes cosine similarity scores between each pair of phrases (one from the input text and one from the standard phrases).
If the cosine similarity score is above a predefined threshold, the script prints both phrases along with the score.
