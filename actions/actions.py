# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa-pro/concepts/custom-actions

import sys
sys.modules["sqlite3"] = __import__("pysqlite3")
import chromadb
import numpy as np
import google.generativeai as genai
from rasa_sdk import Action
from sentence_transformers import SentenceTransformer
import os
from rasa_sdk.events import SlotSet, FollowupAction

#from .utils import *

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up Google Gemini API Key
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

# Connect to ChromaDB
VECTOR_DB_PATH = "vector_store"
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = chroma_client.get_collection(name="class_materials")


# === ACTION 1: FETCH RAW MATERIAL === #
class ActionFetchClassMaterial(Action):
    def name(self):
        return "action_fetch_class_material"

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get("text")  # Get user query
        print(f"\n🧒 User query: '{query}'")
        query = treat_raw_query(query)

        # === DENSE (Vector) SEARCH === #
        print(f"\n🔛 Getting query embeddings for query: '{query}'\n...")
        
        query_embedding = model.encode(query, convert_to_numpy=True).tolist()
        vector_results = collection.query(query_embeddings=[query_embedding], n_results=20)

        vector_docs = vector_results["documents"][0]
        vector_metadata = vector_results["metadatas"][0]
        vector_scores = vector_results["distances"][0]  # Lower is better (L2 distance)


        # === Perform Hyvrid BM25 search === #
        print(f"\n🔛 Getting BM25 sparse vectors...")

        # Step 1: Extract both complex & simple tokens
        complex_tokens = extract_complex_tokens(query)  # e.g., ["pestel analysis"]
        simple_tokens = extract_simple_tokens(query)  # e.g., ["pestel", "analysis"]
        print(f"📖 Finding material location for:\n - {complex_tokens}\n - {simple_tokens}")

        # Step 3: Expand using **weighted synonyms** (prefer closer meanings)
        expanded_complex = expand_query_with_weighted_synonyms(complex_tokens)
        expanded_simple = expand_query_with_weighted_synonyms(simple_tokens)
        print(f"🔄 Expanded tokens with synonyms:\n - {expanded_complex}\n - {expanded_simple}")

        specific_terms = [word for word in expanded_simple if not is_common_word(word)]
        generic_terms = [word for word in expanded_simple if is_common_word(word)]

        print(f"\n🔍 Specific terms: {specific_terms}")
        print(f"📌 Generic terms: {generic_terms}")

        print(f"    🔛 Getting BM25 sparse vectors for both:\n - {expanded_complex}\n - {specific_terms}")
        

        # Step 4: Perform BM25 Search
        bm25_scores_complex = bm25_index.get_scores(expanded_complex)
        bm25_scores_simple = bm25_index.get_scores(specific_terms)

        # Step 5: Combine Scores (Weighting Complex Matches Higher)
        final_scores = 1.5 * bm25_scores_complex + 1.0 * bm25_scores_simple  # Give priority to complex matches
        top_bm25_indices = np.argsort(final_scores)[::-1][:20]  # Top 20 results

        bm25_docs = [bm25_documents[i] for i in top_bm25_indices]
        bm25_metadata_selected = [bm25_metadata[i] for i in top_bm25_indices]
        bm25_scores_selected = [final_scores[i] for i in top_bm25_indices]


        # === NORMALIZE SCORES === #
        max_vec_score = max(vector_scores) if vector_scores else 1
        vector_scores = [1 - (score / max_vec_score) for score in vector_scores]  # Convert L2 to similarity

        max_bm25_score = max(bm25_scores_selected) if bm25_scores_selected else 1
        bm25_scores_selected = [score / max_bm25_score for score in bm25_scores_selected]  # Normalize BM25 scores

        # === MERGE & RE-RANK RESULTS === #
        hybrid_results = []
        alpha = 0.7  # Balance factor between vector & keyword search

        for doc, meta, score in zip(vector_docs, vector_metadata, vector_scores):
            hybrid_score = alpha * score + (1 - alpha) * 0  # BM25 score unavailable for vectors
            hybrid_results.append((doc, meta, hybrid_score))

        alpha = 0.3 
        for doc, meta, score in zip(bm25_docs, bm25_metadata_selected, bm25_scores_selected):
            hybrid_score = alpha * 0 + (1 - alpha) * score  # Vector score unavailable for BM25
            hybrid_results.append((doc, meta, hybrid_score))

        # Sort results by hybrid score
        hybrid_results = sorted(hybrid_results, key=lambda x: x[2], reverse=True)

        # === ADAPTIVE THRESHOLDING BASED ON PERCENTILE === #
        scores = [score for _, _, score in hybrid_results]
        max_score = max(scores) if scores else 1
        adaptive_threshold = max(np.mean(scores) + 0.5 * np.std(scores), np.percentile(scores, 80), 0.7 * max_score)
        print(f"\n📊 Adaptive Threshold: {np.mean(scores) + 0.5 * np.std(scores):.3f}, Percentile_80 threshold: {np.percentile(scores, 80):.3f}, Max Score Threshold: {0.7 * max_score:.3f} \nFinal Threshold => {adaptive_threshold:.3f}")

        # Filter results
        selected_results = [(doc, meta, score) for doc, meta, score in hybrid_results if score >= adaptive_threshold]

        if len(selected_results) == 0:
            dispatcher.utter_message(text="I couldn't find relevant class materials for your query.")
            print("\n🚨 No relevant materials found!")
        else: 
            print("\n✅ Selected Pages After Filtering:")
            document_entries = []
            idx = 1
            for doc, meta, score in selected_results:
                document_entries.append((meta['file'], meta['page']))
                print(f"{idx}. 📄 PDF: {meta['file']} | Page: {meta['page']} | Score: {score:.4f}")
                idx += 1
            
            # **Sort by PDF name (A-Z) and then by page number (ascending)**
            document_entries.sort(key=lambda x: (x[0].lower(), x[1]))
            gemini_results = group_pages_by_pdf(document_entries)
            
            # Format results
            results_text = []
            for text_chunk, meta, score in selected_results:
                file_name = meta["file"]
                page_number = meta["page"]
                results_text.append(f"📄 **From {file_name} (Page {page_number})**:\n{text_chunk}")

            # === PREPARE QUERY FOR GEMINI === #
            raw_text = "\n".join(results_text)
            prompt = f"Use the following raw educational content to answer the student query: '{query}'. Make the provided content more readable to the student: \n{raw_text} "

            print("\n📢 Sending to Gemini API for Summarization...")
            print(f"🔹 Prompt: {prompt[:200]}\n")  # Show only first 200 chars for readability

            try:
                g_model = genai.GenerativeModel("gemini-pro")
                response = g_model.generate_content(prompt)

                if hasattr(response, "text") and response.text:
                    print("\n🎯 Gemini Response Generated Successfully!")
                    formatted_response = format_gemini_response(response.text)
                    print(formatted_response)
                    dispatcher.utter_message(text=formatted_response)
                else:
                    print("\n⚠️ Gemini Response is empty.")
                    dispatcher.utter_message(text="Sorry, I couldn't generate a response.")
            except Exception as e:
                dispatcher.utter_message(text="Sorry, I couldn't process that request.")
                print(f"\n❌ Error calling Gemini API: {e}")

            # Call the new action for material location
            #ActionGetClassMaterialLocation().run(dispatcher, tracker, domain)

        return  [
            SlotSet("user_query", query),  # Store the query
            SlotSet("materials_location", gemini_results),  # Store selected materials
            FollowupAction("action_get_class_material_location")  # Call the next action
            ]


# === ACTION 2: GET PDF NAMES & PAGE LOCATIONS === #
class ActionGetClassMaterialLocation(Action):
    def name(self):
        return "action_get_class_material_location"

    def run(self, dispatcher, tracker, domain):

        selected_materials = tracker.get_slot("materials_location")
        query = tracker.get_slot("user_query") # already treated in last function

        print(f"\n\n 🔖 --------- Getting class materials location --------- 🔖 ")

        # === Perform Hyvrid BM25 search === #
        # Step 1: Extract both complex & simple tokens
        complex_tokens = extract_complex_tokens(query)  # e.g., ["pestel analysis"]
        simple_tokens = extract_simple_tokens(query)  # e.g., ["pestel", "analysis"]
        print(f"\n📖 Finding material location for:\n - {complex_tokens}\n - {simple_tokens}")

        # Step 3: Expand using **weighted synonyms** (prefer closer meanings)
        expanded_complex = expand_query_with_weighted_synonyms(complex_tokens)
        expanded_simple = expand_query_with_weighted_synonyms(simple_tokens)
        print(f"🔄 Expanded tokens with synonyms:\n - {expanded_complex}\n - {expanded_simple}")
        
        # Step 4: Perform BM25 Search
        bm25_scores_complex = bm25_index.get_scores(expanded_complex)
        bm25_scores_simple = bm25_index.get_scores(expanded_simple)

        # Step 5: Combine Scores (Weighting Complex Matches Higher)
        final_scores = 1.5 * bm25_scores_complex + 1.0 * bm25_scores_simple  # Give priority to complex matches

        top_indices = np.argsort(final_scores)[::-1][:10]  # Top 10 results

        specific_terms = [word for word in expanded_simple if not is_common_word(word)]
        generic_terms = [word for word in expanded_simple if is_common_word(word)]

        print(f"\n🔍 Specific terms: {specific_terms}")
        print(f"📌 Generic terms: {generic_terms}")
    

        location_results = []
        document_entries = []  # Store documents before sorting

        for i in top_indices:
            file_name = bm25_metadata[i]["file"]
            page_number = bm25_metadata[i]["page"]
            document_text = bm25_documents[i]

            # Tokenize the document text
            document_tokens = extract_complex_tokens(document_text)

            # Perform fuzzy matching -> solves matches like 'external environment analysis\npestel analysis'
            if fuzzy_match(expanded_complex, document_tokens):
                document_entries.append((file_name, page_number))
        
        if len(document_entries) == 0:
            print("\n👻 --> No matching for Complex Tokens")
            for i in top_indices:
                file_name = bm25_metadata[i]["file"]
                page_number = bm25_metadata[i]["page"]
                document_text = bm25_documents[i]

                # Tokenize the document text
                simple_document_tokens = extract_simple_tokens(document_text)

                # ✅ Ensure at least one "specific" word is found before allowing generic matches
                contains_specific = any(fuzzy_match([word], simple_document_tokens) for word in specific_terms)
                contains_generic = any(fuzzy_match([word], simple_document_tokens) for word in generic_terms)

                if contains_specific or (contains_generic and len(specific_terms) == 0):
                    document_entries.append((file_name, page_number))


        # **Sort by PDF name (A-Z) and then by page number (ascending)**
        document_entries.sort(key=lambda x: (x[0].lower(), x[1]))  

        # Format results
        location_results = group_pages_by_pdf(document_entries)

        if location_results:

            if len(document_entries) > len(selected_materials) * 3: # means that the tokenization went wrong
                location_results = selected_materials

            print("\n🎯 Material location for query found!")
            print("\n📌 FINAL SORTED RESULTS:")
            for result in location_results:
                print(result)
            print()
            dispatcher.utter_message(text="You can find more information in:\n" + "\n".join(location_results))
        else:
            print("\n⚠️  No exact references found, but you might check related PDFs.")
            dispatcher.utter_message(text="I couldn't find specific page references, but check related PDFs.")

        return []


import spacy
from fuzzywuzzy import fuzz
import pickle
from difflib import get_close_matches
import re

nlp = spacy.load("en_core_web_sm")

# Load BM25 index
with open("vector_store/bm25_index.pkl", "rb") as f:
    bm25_index, bm25_metadata, bm25_documents = pickle.load(f)


# === STEP 1: MULTI-WORD EXPRESSION (MWE) EXTRACTION === #

def extract_complex_tokens(query): # ['pestel analysis']
    """Extracts only meaningful subject keywords from a query."""
    doc = nlp(query.lower())  # Process query with NLP model
    keywords = []
    single_word_tokens = set()  # Store individual words temporarily

    # Extract noun phrases (multi-word terms)
    for chunk in doc.noun_chunks:
        keyword = chunk.text.strip()
        if len(keyword.split()) > 1:  # Only keep multi-word phrases
            keywords.append(keyword)
            single_word_tokens.update(keyword.split())  # Store individual words to avoid later

    # Extract single meaningful words (NOUN, PROPN) **if not part of a noun phrase**
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
            if token.text not in single_word_tokens:  # Exclude if part of a noun phrase
                keywords.append(token.text)

    # Remove duplicates while preserving order
    keywords = list(dict.fromkeys(keywords))
    return keywords


def extract_key_expressions(text): # ['pestel', 'analysis', 'pestel analysis']
    """ Extracts key multi-word expressions using NLP phrase detection. """
    doc = nlp(text.lower())
    key_expressions = set()

    # Extract multi-word phrases (noun chunks)
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if len(phrase.split()) > 1:  
            key_expressions.add(phrase)  

    # Add important single words
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
            key_expressions.add(token.text)

    return list(key_expressions)


# === EXPAND SYNONYMS === #
from itertools import product
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')  # Optional, for additional synonyms
from nltk.corpus import wordnet

def get_synonyms(word):
    """Fetch synonyms from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))  # Replace underscores in multi-word terms
    return list(synonyms)

def expand_query_with_synonyms(query_expressions):
    """Expand expressions with their known synonyms, preserving multi-word structure."""
    expanded_queries = set()

    for expr in query_expressions:
        words = expr.split()  # Split phrase into individual words
        synonym_options = []

        for word in words:
            synonyms = get_synonyms(word)  # Fetch synonyms dynamically
            synonym_options.append([word] + synonyms)  # Include original word + synonyms

        # Generate all possible replacements (cartesian product)
        for combination in product(*synonym_options):
            expanded_queries.add(" ".join(combination))  # Rebuild phrase

    return list(expanded_queries)

def expand_query_with_weighted_synonyms(query_expressions):
    """Expand query with weighted synonyms: prioritize more relevant expansions."""
    expanded_queries = set()
    
    for expr in query_expressions:
        words = expr.split()
        synonym_options = []

        for word in words:
            synonyms = get_synonyms(word)
            strong_synonyms = synonyms[:3]  # Limit to top 3 most relevant synonyms

            # Prioritize original word, then close synonyms
            synonym_options.append([word] + strong_synonyms)

        for combination in product(*synonym_options):
            expanded_queries.add(" ".join(combination))

    return list(expanded_queries)

import re

def format_gemini_response(text: str) -> str:
    """
    Replace triple backticks (```) with ** for bold formatting.
    
    Args:
        text (str): The response text from Gemini.
    
    Returns:
        str: The formatted text with ** tags instead of triple backticks.
    """
    text = re.sub(r'```(.*?)```', r'**\1**', text, flags=re.DOTALL)
    return text


def lemmatize_word(word):
    """Returns the lemma of a given word (e.g., 'methods' → 'method')."""
    doc = nlp(word)
    return doc[0].lemma_  # Return the base form (lemma)

def fuzzy_match(query_tokens, document_tokens, threshold=85):
    """Matches query tokens against document tokens, handling lemmatization & fuzzy similarity."""
    
    doc_text = " ".join(document_tokens).lower()  # Join doc tokens into full text
    query_tokens = [qt.lower() for qt in query_tokens]  # Lowercase query tokens

    for query_token in query_tokens:
        lemma_query = lemmatize_word(query_token)  # Convert to base form
        
        if " " in query_token:  # If query token is a phrase (e.g., "pestel framework")
            if query_token in doc_text: # or lemma_query in doc_text:  # Check for phrase
                print(f"\n  💚 Match in query_token '{query_token}':\n{doc_text}")
                return True
        else:  # Single word matching
            for doc_token in document_tokens:
                lemma_doc = lemmatize_word(doc_token)  # Lemmatize doc token
                # Allow exact match, lemmatized match, or fuzzy match
                if (query_token == doc_token or  
                    lemma_query == lemma_doc or  
                    fuzz.token_set_ratio(query_token, doc_token) >= threshold):
                    print(f"\n  📗 Match in query_token '{query_token}'=='{doc_token}' or lemma_query '{lemma_query}'=='{lemma_doc}'")

                    return True  

    return False  # No match found

from wordfreq import word_frequency

def is_common_word(word, threshold=0.00001):
    """
    Check if a word is common based on its frequency in large corpora.
    Lower threshold = more words classified as common.
    """
    freq = word_frequency(word, 'en')  # Get word frequency
    return freq > threshold  # If frequency is high, it's a common word



def extract_simple_tokens(query): # ['pestel', 'analysis']
    """Extracts only meaningful single-word tokens from a query (excluding stopwords & phrases)."""
    doc = nlp(query.lower())  # Process query with NLP model
    keywords = []
    
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
            keywords.append(token.text)

        # Include adjectives that appear **before** a noun (e.g., "financial management")
        elif token.pos_ == "ADJ" and token.dep_ in {"amod", "compound"}:
            keywords.append(token.text)

    # Remove duplicates while preserving order
    keywords = list(dict.fromkeys(keywords))
    return keywords

# Collect all unique words from BM25 documents to compare for spell correction
VALID_SIMPLE_WORDS = set()
for doc_text in bm25_documents:
    VALID_SIMPLE_WORDS.update(extract_simple_tokens(doc_text))
VALID_WORDS = set()
for doc_text in bm25_documents:
    VALID_WORDS.update(extract_complex_tokens(doc_text))


def group_pages_by_pdf(document_entries):
    """
    Groups consecutive pages for the same PDF into a range format.
    Example:
        Input: [("file1.pdf", 1), ("file1.pdf", 2), ("file1.pdf", 3), ("file2.pdf", 10), ("file2.pdf", 12)]
        Output: ["file1.pdf (Pages 1-3)", "file2.pdf (Pages 10, 12)"]
    """
    grouped_results = []
    current_pdf = None
    current_pages = []

    for file_name, page in document_entries:
        if file_name != current_pdf:  
            # If switching to a new PDF, store the previous result
            if current_pdf:
                grouped_results.append(format_page_range(current_pdf, current_pages))
            # Reset tracking for new PDF
            current_pdf = file_name
            current_pages = [page]
        else:
            current_pages.append(page)

    # Add the last processed PDF
    if current_pdf:
        grouped_results.append(format_page_range(current_pdf, current_pages))

    return grouped_results

def format_page_range(file_name, pages):
    """
    Converts a list of page numbers into a formatted string.
    Example:
        Input: "file1.pdf", [1, 2, 3, 5, 6, 8]
        Output: "📄 file1.pdf (Pages 1-3, 5-6, 8)"
    """
    pages.sort()
    ranges = []
    start = pages[0]

    for i in range(1, len(pages)):
        if pages[i] != pages[i - 1] + 1:  # Break in sequence
            if start == pages[i - 1]:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{pages[i - 1]}")
            start = pages[i]

    # Add the final range
    if start == pages[-1]:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{pages[-1]}")

    return f"📄 **{file_name} (Pages {', '.join(ranges)})**"

def treat_raw_query(query):
    # === Treat user query === #
    print(f"\n📍 Raw query: {query}")

    query_tokens = [token.text for token in nlp(query)]
    print(f"    📖 Query tokens: {query_tokens}")

    imp_tokens = extract_simple_tokens(query) # Extract meaningful keywords
    imp_tokens_dict = {}
    for token in imp_tokens:
        # Correct potential misspellings in the student query 
        imp_tokens_dict.update({token: correct_spelling(token)})

    updated_query_tokens = []
    for token in query_tokens:
        if imp_tokens_dict.get(token):
            updated_query_tokens.append(imp_tokens_dict.get(token))
        else: 
            updated_query_tokens.append(token)
    print(f"    ✅ Corrected Tokens After Spell Check: {updated_query_tokens}")
 
    corrected_query = " ".join(updated_query_tokens)
    print(f"📍 Treated query: {corrected_query}")
    
    return corrected_query

# === SPELL CORRECTION === #

def correct_spelling(word, set=VALID_SIMPLE_WORDS): # TODO: check if no need to use VALID_WORDS !?
    """Corrects spelling by finding the closest valid match."""
    closest_match = get_close_matches(word, set, n=1, cutoff=0.8)  # 80% similarity threshold
    if closest_match:
        print(f"    - 🐟 best match for '{word}': {closest_match[0]}")
    return closest_match[0] if closest_match else word  # Return the match or original word

# === SPELL CHECK WRAPPER === #
def correct_query_tokens(tokens, set):
    """Corrects a list of tokens for spelling mistakes."""
    return [correct_spelling(token, set) for token in tokens]