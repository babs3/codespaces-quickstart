# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa-pro/concepts/custom-actions


import os
import sys
sys.modules["sqlite3"] = __import__("pysqlite3")
import chromadb
import numpy as np
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="vector_store/chroma_db")
collection = chroma_client.get_collection(name="class_materials")

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up Google Gemini API Key
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

class ActionFetchClassMaterial(Action):
    def name(self):
        return "action_fetch_class_material"

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get("text")  # Get user query
        query_embedding = model.encode(query).tolist()

        # Retrieve top 3 results from ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3  # Retrieve top 3 chunks
        )

        retrieved_texts = []
        file_names = set()

        if results["ids"]:
            for i in range(len(results["ids"][0])):
                file_name = results["metadatas"][0][i]["file"]
                chunk_text = results["metadatas"][0][i]["text"]
                file_names.add(file_name)
                retrieved_texts.append(f"From {file_name}:\n{chunk_text}")

            raw_text = "\n".join(retrieved_texts)
            file_names_str = ", ".join(file_names)  # List of PDFs used

            prompt = f"Summarize this educational content and make it more readable for students. The content comes from: {file_names_str}.\n\n{raw_text}\n\nIn the end of your response, please refer the pdfs where the relevant info came from."

            try:
                # Call Gemini API
                g_model = genai.GenerativeModel("gemini-pro")
                response = g_model.generate_content(prompt)

                # Extract response text correctly
                if hasattr(response, "text") and response.text:
                    dispatcher.utter_message(text=response.text)
                else:
                    dispatcher.utter_message(text="Sorry, I couldn't generate a response.")

            except Exception as e:
                dispatcher.utter_message(text="Sorry, I couldn't process that request.")
                print(f"Error: {e}")
        else:
            dispatcher.utter_message(text="I couldn't find relevant class materials for your query.")

        return []



""" not used for now """
class ActionGenerateAnswer(Action): # only using gemini
    def name(self):
        return "action_generate_answer"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_query = tracker.latest_message.get("text")

        try:
            # Call Gemini API
            g_model = genai.GenerativeModel("gemini-pro")
            response = g_model.generate_content(user_query)

            # Send Gemini's response back to user
            dispatcher.utter_message(text=response.text)

        except Exception as e:
            dispatcher.utter_message(text="Sorry, I couldn't process that request.")
            print(f"Error: {e}")

        return []


from transformers import pipeline
from rasa_sdk import Action

class ActionAskMistral(Action):
    def name(self):
        return "action_ask_mistral"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get("text")

        #qa_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")
        qa_pipeline = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

        bot_response = qa_pipeline(user_message, max_length=100)[0]["generated_text"]

        dispatcher.utter_message(text=bot_response)
        return []


