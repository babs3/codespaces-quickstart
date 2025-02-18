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

class ActionFetchClassMaterial(Action):
    def name(self):
        return "action_fetch_class_material"

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get("text")  # Get user query
        print(f"🧒 User query: '{query}'")

        # === DENSE (Vector) SEARCH === #
        query_embedding = model.encode(query, convert_to_numpy=True).tolist()
        vector_results = collection.query(query_embeddings=[query_embedding], n_results=10)

        vector_docs = vector_results["documents"][0]
        vector_metadata = vector_results["metadatas"][0]
        vector_scores = vector_results["distances"][0]  # Lower is better (L2 distance)

        # === SPARSE (BM25) SEARCH === #
        query_tokens = query.lower().split()
        bm25_scores = bm25_index.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:10]  # Top 10 results

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
        percentile_cutoff = 70  # Retrieve top 30% of most relevant results
        threshold = np.percentile(scores, percentile_cutoff)  # Set threshold at top 30% of results

        print(f"📊 Using {percentile_cutoff}th Percentile as Threshold: {threshold:.4f}")

        # Filter results based on PERCENTILE as threshold
        selected_results = [(doc, meta, score) for doc, meta, score in hybrid_results if score >= threshold]

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
            prompt = f"Use the following raw educational content to answer the student query: '{query}'. Make the provided content more readable to the student and don't forget to mention the PDF name and page numbers where the student could find more information: \n{raw_text} "

            print("\n📢 Sending to Gemini API for Summarization...")
            print(f"🔹 Prompt: {prompt[:500]}...")  # Show only first 500 chars for readability

            try:
                # Call Gemini API
                g_model = genai.GenerativeModel("gemini-pro")
                response = g_model.generate_content(prompt)

                # ✅ Extract response text correctly
                if hasattr(response, "text") and response.text:
                    print("\n🎯 Gemini Response Generated Successfully!")
                    dispatcher.utter_message(text=response.text)
                else:
                    dispatcher.utter_message(text="Sorry, I couldn't generate a response.")
                    print("\n⚠️ Gemini Response is empty.")
            except Exception as e:
                dispatcher.utter_message(text="Sorry, I couldn't process that request.")
                print(f"\n❌ Error calling Gemini API: {e}")

        return []
