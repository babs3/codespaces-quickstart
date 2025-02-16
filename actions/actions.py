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
import os

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up Google Gemini API Key
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

# Connect to ChromaDB
VECTOR_DB_PATH = "vector_store"
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = chroma_client.get_collection(name="class_materials")

class ActionFetchClassMaterial(Action):
    def name(self):
        return "action_fetch_class_material"

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get("text")  # Get user query
        query_embedding = model.encode(query, convert_to_numpy=True).tolist()  # ✅ Ensure it's 1D

        # Search in ChromaDB - Get top 20 candidates
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10  # Retrieve more pages first
        )

        # Extract scores and metadata
        scores = search_results["distances"][0]  # Similarity scores for retrieved pages
        documents = search_results["documents"][0]  # Retrieved text chunks
        metadata = search_results["metadatas"][0]  # Metadata (file names, pages)

        #print("\n🔍 Retrieved Results from ChromaDB:")
        #for i, (doc, meta, score) in enumerate(zip(documents, metadata, scores)):
        #    print(f"{i+1}. PDF: {meta['file']} | Page: {meta.get('page', 'unknown')} | Score: {score:.4f}")

        # Normalize scores (FAISS distances are L2 distances, so we convert to similarity)
        max_score = max(scores) if scores else 1  # Avoid division by zero
        print("\n✏️  max_score: " + str(max_score))
        #normalized_scores = [(1 - (score / max_score)) for score in scores]  # Convert to similarity (higher is better)

        # Set a dynamic score threshold (e.g., retrieve pages with at least 80% of max relevance)
        threshold = 0.8 * max_score  
        print("🧻 Threshold: " + str(threshold))
        selected_results = [
            (documents[i], metadata[i], scores[i])
            for i in range(len(scores))
            if scores[i] >= threshold  # Keep only relevant results
        ]

        # Ensure at least 1 result is returned
        if len(selected_results) == 0:
            selected_results = [(documents[0], metadata[0], scores[0])]

        print("\n✅ Selected Pages After Filtering:")
        idx = 1
        for doc, meta, score in selected_results:
            print(f"{idx}. 📄 PDF: {meta['file']} | Page: {meta.get('page', 'unknown')} | Score: {score:.4f}")
            idx += 1

        # Format results
        results_text = []
        for text_chunk, meta, score in selected_results:
            file_name = meta["file"]
            page_number = meta["page"] if "page" in meta else "unknown"
            results_text.append(f"📄 **From {file_name} (Page {page_number})**:\n{text_chunk}")

        if results_text:
            # Prepare final text for Gemini
            raw_text = "\n".join(results_text)
            #prompt = f"Summarize this educational content and make it more readable for students. Keep the PDF name and page numbers clear: \n{raw_text}."
            prompt = f"Use the following raw educational content to answer the student query: '{query}'. Make the provided content more readable to the student and don't forget to mention the PDF name and page numbers where the student could find more information: \n{raw_text} "
            print("\n📢 Sending to Gemini API for Summarization...")
            print(f"🔹 Prompt: {prompt[:300]}...")  # Show only first 300 chars for readability

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
        else:
            dispatcher.utter_message(text="I couldn't find relevant class materials for your query.")
            print("\n🚨 No relevant materials found!")

        return []

