# Phrase Similarity Analyzer

Phrase Similarity Analyzer is a tool that helps in finding phrases in a text sample that are similar to a predefined set of standard phrases using cosine similarity with BERT embeddings.

## Requirements

Before running the script, make sure to install the necessary Python packages. You can install them with the following command:

```
annotated-types==0.5.0
anyio==3.7.1
blis==0.7.10
catalogue==2.0.9
certifi==2023.7.22
charset-normalizer==3.2.0
click==8.1.7
confection==0.1.3
cymem==2.0.8
fastapi==0.103.1
filelock==3.12.4
fsspec==2023.9.1
h11==0.14.0
huggingface-hub==0.17.2
idna==3.4
Jinja2==3.1.2
joblib==1.3.2
langcodes==3.3.0
MarkupSafe==2.1.3
mpmath==1.3.0
murmurhash==1.0.10
networkx==3.1
nltk==3.8.1
numpy==1.26.0
packaging==23.1
pathy==0.10.2
preshed==3.0.9
pydantic==2.3.0
pydantic_core==2.6.3
PyYAML==6.0.1
regex==2023.8.8
requests==2.31.0
safetensors==0.3.3
scikit-learn==1.3.0
scipy==1.11.2
smart-open==6.4.0
sniffio==1.3.0
spacy==3.6.1
spacy-legacy==3.0.12
spacy-loggers==1.0.5
srsly==2.4.7
starlette==0.27.0
sympy==1.12
thinc==8.1.12
threadpoolctl==3.2.0
tokenizers==0.13.3
torch==2.0.1
tqdm==4.66.1
transformers==4.33.2
typer==0.9.0
typing_extensions==4.8.0
urllib3==2.0.4
uvicorn==0.23.2
wasabi==1.1.2

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
