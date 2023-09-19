import warnings

warnings.filterwarnings('ignore')

import torch
import spacy
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk

nltk.download('punkt')

sample_text = input("Please enter the text to analyze: ")

standard_phrases = [
    "Optimal performance",
    "Utilise resources",
    "Enhance productivity",
    "Conduct an analysis",
    "Maintain a high standard",
    "Implement best practices",
    "Ensure compliance",
    "Streamline operations",
    "Foster innovation",
    "Drive growth",
    "Leverage synergies",
    "Demonstrate leadership",
    "Exercise due diligence",
    "Maximize stakeholder value",
    "Prioritise tasks",
    "Facilitate collaboration",
    "Monitor performance metrics",
    "Execute strategies",
    "Gauge effectiveness",
    "Champion change",
]

# Use GPU if it's available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Bert model for making embeddings
# Spacy for splitting text into phrases
nlp = spacy.load('en_core_web_sm')
model = BertModel.from_pretrained('bert-base-uncased').to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

doc = nlp(sample_text)
sentences = [sent.text for sent in doc.sents]
sample_phrases = []

for sentence in sentences:

    # Tokenize the sentence into words using Spacy
    doc = nlp(sentence)
    tokens = [token.text for token in doc]

    # Create phrases using a sliding window of size between 2 to 10
    for window_size in range(2, 11):
        phrases = [' '.join(tokens[i:i + window_size]) for i in range(0, len(tokens) - window_size + 1)]

        # Add the phrases to the sample_phrases list
        sample_phrases.extend(phrases)

cosine_similarity_threshold = 0.75


# Define the embeddings for all phrases in both lists
def get_embeddings(phrases_list):
    embeddings = []
    for phrase in phrases_list:
        inputs = tokenizer(phrase, return_tensors='pt', max_length=512, truncation=True)
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}  # Move inputs to the correct device
        with torch.no_grad():
            output = model(**inputs)
        embeddings.append(output.last_hidden_state.mean(dim=1))
    return embeddings


standard_phrases_embeddings = get_embeddings(standard_phrases)
sample_phrases_embeddings = get_embeddings(sample_phrases)

# Calculate the cosine similarity for each pair of phrases
for i, sample_embedding in enumerate(sample_phrases_embeddings):
    for j, standard_embedding in enumerate(standard_phrases_embeddings):

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(sample_embedding.cpu().numpy(), standard_embedding.cpu().numpy())[0][0]

        # If cosine similarity is above the threshold, print the phrases and their similarity score
        if cosine_sim > cosine_similarity_threshold:
            print(f"Sample phrase: '{sample_phrases[i]}'")
            print(f"Standard phrase: '{standard_phrases[j]}'")
            print(f"Cosine similarity score: {cosine_sim:.4f}")
            print("-" * 80)
