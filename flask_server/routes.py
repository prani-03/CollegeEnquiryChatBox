from flask import Blueprint, request, jsonify, render_template
import json
import random
from sentence_transformers import SentenceTransformer, util
import re


# Create a Flask blueprint for routes
main_routes = Blueprint('main', __name__)

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load intents data from intents.json
with open('newintents.json', 'r') as f:
    intents = json.load(f)

# Precompute embeddings for all patterns in intents.json
patterns = []
for intent in intents['intents']:
    patterns.extend(intent['patterns'])

# Create embeddings for the patterns
pattern_embeddings = model.encode(patterns, convert_to_tensor=True)

def is_meaningful_input(user_input):
    """Check if the input is a meaningful English phrase."""
    # Check if the input is empty or consists only of symbols
    if not user_input.strip() or not any(char.isalnum() for char in user_input):
        return False
    
    # Check for valid words (using a simple regex to find words)
    words = re.findall(r'\b\w+\b', user_input)  # Find all words in the input
    return len(words) > 1  # Ensure there are at least two words


def get_response(user_input):
    """Get a response based on user input using NLP techniques."""

    # Check if the user input is meaningful
    if not is_meaningful_input(user_input):
        return "I'm sorry, I didn't understand that."

    # Generate embedding for user input
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    # Compute cosine similarities between user input and pattern embeddings
    similarities = util.pytorch_cos_sim(user_embedding, pattern_embeddings)

    # Find the index of the highest similarity score
    best_match_index = similarities.argmax()

    # Get the corresponding intent based on the best match index
    best_match_intent = ""
    for intent in intents['intents']:
        if best_match_index < len(intent['patterns']):
            best_match_intent = intent
            break
        best_match_index -= len(intent['patterns'])

    # Return a response from the best matching intent if found
    if best_match_intent:
        return random.choice(best_match_intent['responses'])
    
    else:
        return "I'm sorry, I didn't understand that."

@main_routes.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint to receive user messages and return responses."""
    user_input = request.json.get('message')
    response = get_response(user_input)
    return jsonify({'response': response})

@main_routes.route('/')
def home():
    """Home route to serve the main page."""
    return render_template('index.html')  # Ensure this file exists in templates/