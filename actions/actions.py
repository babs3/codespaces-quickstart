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
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
import os
import re
import spacy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Define known synonyms (expand this list over time)
SYNONYM_MAP = {
    "framework": ["analysis", "model", "methodology"],
    "evaluation": ["assessment", "review"],
    "financial ratios": ["accounting ratios"]
}

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up Google Gemini API Key
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

# Connect to ChromaDB
VECTOR_DB_PATH = "vector_store"
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = chroma_client.get_collection(name="class_materials")

# Load BM25 index
with open("vector_store/bm25_index.pkl", "rb") as f:
    bm25_index, bm25_metadata, bm25_documents = pickle.load(f)

nlp = spacy.load("en_core_web_sm")

def tokenize(query):
    doc = nlp(query.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return tokens

# === STEP 1: MULTI-WORD EXPRESSION (MWE) EXTRACTION === #
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_keywords(query):
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


# === STEP 2: EXPAND SYNONYMS === #
from itertools import product

# === STEP 2: EXPAND SYNONYMS === #
def expand_query_with_synonyms(query_expressions):
    """Expand expressions with their known synonyms, preserving multi-word structure."""
    expanded_queries = set()

    for expr in query_expressions:
        words = expr.split()  # Split phrase into individual words
        synonym_options = []

        for word in words:
            if word in SYNONYM_MAP:
                synonym_options.append([word] + SYNONYM_MAP[word])  # Include original word + synonyms
            else:
                synonym_options.append([word])  # Keep the word unchanged

        # Generate all possible replacements (cartesian product)
        for combination in product(*synonym_options):
            expanded_queries.add(" ".join(combination))  # Rebuild phrase

    return list(expanded_queries)


# === STEP 3: SPELL CORRECTION === #

from difflib import get_close_matches

def correct_spelling(word, set):
    """Corrects spelling by finding the closest valid match."""
    closest_match = get_close_matches(word, set, n=1, cutoff=0.8)  # 80% similarity threshold
    if closest_match:
        print(f"    - 🐟 best match for '{word}': {closest_match[0]}; {closest_match}")
    return closest_match[0] if closest_match else word  # Return the match or original word

# === SPELL CHECK WRAPPER === #
def correct_query_tokens(tokens, set):
    """Corrects a list of tokens for spelling mistakes."""
    return [correct_spelling(token, set) for token in tokens]



# === STEP 4: DOCUMENT FILTERING BASED ON MWE PRESENCE === #
def document_contains_expression(doc_text, expressions, threshold=85):
    """Ensures that at least one key expression exists in the document."""
    for expr in expressions:
        for sentence in doc_text.split("."):  # Check each sentence separately
            if fuzz.token_set_ratio(expr.lower(), sentence.lower()) >= threshold:
                return True
    return False


# === ACTION 1: FETCH RAW MATERIAL === #
class ActionFetchClassMaterial(Action):
    def name(self):
        return "action_fetch_class_material"

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get("text")  # Get user query
        print(f"\n🧒 User query: '{query}'")

        # === DENSE (Vector) SEARCH === #
        print(f"\n🔛 Getting query embeddings...")

        query_tokens = query.split()  # Extract meaningful keywords
        print(f"    📖 Query tokens: {query_tokens}")

        corrected_tokens = correct_query_tokens(query_tokens, VALID_SIMPLE_WORDS) # Correct potential misspellings in the student query
        print(f"    ✅ Corrected Tokens After Spell Check: {corrected_tokens}")

        query = " ".join(corrected_tokens)
        print(f"    📍 Getting query embeddings for query: {query}")
        
        query_embedding = model.encode(query, convert_to_numpy=True).tolist()
        vector_results = collection.query(query_embeddings=[query_embedding], n_results=20)

        vector_docs = vector_results["documents"][0]
        vector_metadata = vector_results["metadatas"][0]
        vector_scores = vector_results["distances"][0]  # Lower is better (L2 distance)

        # === SPARSE (BM25) SEARCH === #
        #query_tokens = tokenize(query)
        print(f"\n🔛 Getting BM25 sparse vectors...")

        query_tokens = extract_simple_tokens(query)  # Extract meaningful keywords
        print(f"    📖 Query tokens: {query_tokens}")

        corrected_tokens = correct_query_tokens(query_tokens, VALID_SIMPLE_WORDS) # Correct potential misspellings in the student query
        print(f"    ✅ Corrected Tokens After Spell Check: {corrected_tokens}")

        expanded_tokens = expand_query_with_synonyms(corrected_tokens)  # Expand with synonyms
        print(f"    🔄 Expanded keywords with synonyms: {expanded_tokens}")
        print(f"    📍 Getting bm25_scores for tokens: {expanded_tokens}")

        bm25_scores = bm25_index.get_scores(expanded_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:20]  # Top 20 results

        bm25_docs = [bm25_documents[i] for i in top_bm25_indices]
        bm25_metadata_selected = [bm25_metadata[i] for i in top_bm25_indices]
        bm25_scores_selected = [bm25_scores[i] for i in top_bm25_indices]

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
            idx = 1
            for doc, meta, score in selected_results:
                print(f"{idx}. 📄 PDF: {meta['file']} | Page: {meta['page']} | Score: {score:.4f}")
                idx += 1

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
            print(f"🔹 Prompt: {prompt[:300]}...")  # Show only first 300 chars for readability

            try:
                g_model = genai.GenerativeModel("gemini-pro")
                response = g_model.generate_content(prompt)

                if hasattr(response, "text") and response.text:
                    print("\n🎯 Gemini Response Generated Successfully!")
                    dispatcher.utter_message(text=response.text)
                else:
                    print("\n⚠️ Gemini Response is empty.")
                    dispatcher.utter_message(text="Sorry, I couldn't generate a response.")
            except Exception as e:
                dispatcher.utter_message(text="Sorry, I couldn't process that request.")
                print(f"\n❌ Error calling Gemini API: {e}")

            # Call the new action for material location
            ActionGetClassMaterialLocation().run(dispatcher, tracker, domain)

        return []


# Function to check if any query token loosely matches document tokens
def fuzzy_match(query_tokens, document_tokens, threshold=95):
    """Ensures that multi-word terms appear as full expressions in the document text."""
    
    doc_text = " ".join(document_tokens).lower()  # Join doc tokens into full text
    query_tokens = [qt.lower() for qt in query_tokens]  # Lowercase query tokens

    for query_token in query_tokens:
        if " " in query_token:  # If query token is a phrase (e.g., "pestel framework")
            if query_token in doc_text:  # Check if entire phrase appears in doc
                print(f"\n✅ Exact phrase match found! Token: '{query_token}'")
                print(f"📄 Context: {doc_text[:500]}")  # Print first 500 chars for debugging
                return True
        else:  # If single word, apply fuzzy matching
            for doc_token in document_tokens:
                if fuzz.token_set_ratio(query_token, doc_token) >= threshold:
                    print(f"\n✅ Match found! Token: '{query_token}' in '{doc_token}'.")
                    print(f"📄 Context: {doc_text[:500]}")
                    return True  
                    
    return False  # If no match found


def extract_simple_tokens(query):
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
    VALID_WORDS.update(extract_keywords(doc_text))


# === ACTION 2: GET PDF NAMES & PAGE LOCATIONS === #
class ActionGetClassMaterialLocation(Action):
    def name(self):
        return "action_get_class_material_location"

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get("text")  

        # treat user query:
        query_tokens = query.split()  # Extract meaningful keywords
        print(f"    📖 Query tokens: {query_tokens}")

        corrected_tokens = correct_query_tokens(query_tokens, VALID_SIMPLE_WORDS) # Correct potential misspellings in the student query
        print(f"    ✅ Corrected Tokens After Spell Check: {corrected_tokens}")

        query = " ".join(corrected_tokens)
        print(f"    📍 Treated query: {query}")

        query_tokens = extract_simple_tokens(query)  # Extract meaningful keywords
        print(f"\n📖 Finding material location for: {query_tokens}")

        corrected_tokens = correct_query_tokens(query_tokens, VALID_SIMPLE_WORDS) # Correct potential misspellings in the student query
        print(f"✅ Corrected Tokens After Spell Check: {corrected_tokens}")

        expanded_tokens = expand_query_with_synonyms(corrected_tokens)  # Expand with synonyms
        print(f"🔄 Expanded keywords with synonyms: {expanded_tokens}")
        
        # Perform BM25 search
        bm25_scores = bm25_index.get_scores(expanded_tokens) # tem de ser dos tokens individuais
        top_indices = np.argsort(bm25_scores)[::-1][:10]  # Top 10 matches

        print('_'*80)

        query_tokens = extract_keywords(query)  # Extract meaningful keywords
        print(f"\n📖 Finding material location for: {query_tokens}")

        corrected_tokens = correct_query_tokens(query_tokens, VALID_WORDS) # Correct potential misspellings in the student query
        print(f"✅ Corrected Tokens After Spell Check: {corrected_tokens}")

        expanded_tokens = expand_query_with_synonyms(corrected_tokens)  # Expand with synonyms
        print(f"🔄 Expanded keywords with synonyms: {expanded_tokens}")
    

        location_results = []
        document_entries = []  # Store documents before sorting

        for i in top_indices:
            file_name = bm25_metadata[i]["file"]
            page_number = bm25_metadata[i]["page"]
            document_text = bm25_documents[i]

            # Tokenize the document text
            document_tokens = extract_keywords(document_text)

            # Perform fuzzy matching -> solves matches like 'external environment analysis\npestel analysis'
            if fuzzy_match(expanded_tokens, document_tokens):
                document_entries.append((file_name, page_number))

            #for token in corrected_tokens:
            #    if token in document_tokens:
            #        print(f"✅ Match found! Token: '{token}' in document: '{file_name}' (Page {page_number})")
            #        print(f"📄 Context: {' '.join(document_tokens)}")  # Print full document tokens for debugging
            #        document_entries.append((file_name, page_number))
            #        break  # Stop after the first match


        # **Sort by PDF name (A-Z) and then by page number (ascending)**
        document_entries.sort(key=lambda x: (x[0].lower(), x[1]))  

        # Format results
        location_results = [f"📄 **{entry[0]} (Page {entry[1]})**" for entry in document_entries]

        if location_results:
            print("\n🎯 Material location for query found!")
            print("\n📌 FINAL SORTED RESULTS:")
            for result in location_results:
                print(result)
            dispatcher.utter_message(text="You can find more information in:\n" + "\n".join(location_results))
        else:
            print("\n⚠️  No exact references found, but you might check related PDFs.")
            dispatcher.utter_message(text="I couldn't find specific page references, but check related PDFs.")

        return []
