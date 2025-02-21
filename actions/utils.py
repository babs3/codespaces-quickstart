import spacy
from fuzzywuzzy import fuzz
import pickle
from difflib import get_close_matches

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