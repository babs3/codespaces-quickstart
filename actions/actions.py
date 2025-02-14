# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa-pro/concepts/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

"""
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
#from rasa_sdk.events import SlotSet
from typing import Any, Text, Dict, List

from materials.auxiliar import *

class ActionFetchClassMaterial(Action):
    def name(self):
        return "action_fetch_class_material"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        user_query = tracker.get_slot("user_query")
        relevant_info = search_text(user_query)  # Call vector search function
        dispatcher.utter_message(f"Here's some information from class materials:\n\n{relevant_info}")

        #materials = "Here are your class materials: [link]"
        #dispatcher.utter_message(text=materials)
        return []
"""


import os
import faiss
import pickle
import numpy as np
from rasa_sdk import Action
from rasa_sdk.events import SlotSet
from sentence_transformers import SentenceTransformer

# Load the same model used for indexing
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and document store
INDEX_PATH = "vector_store/faiss_index"
DATA_PATH = "vector_store/documents.pkl"

index = faiss.read_index(INDEX_PATH)

with open(DATA_PATH, "rb") as f:
    documents, metadata = pickle.load(f)

    print("Metadata Type:", type(metadata))  # Should be a list
    print("Metadata Example:", metadata[0])  # Should be a dictionary with "file" and "chunk_id"


class ActionFetchClassMaterial(Action):
    def name(self):
        return "action_fetch_class_material"

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get("text")  # Get user query
        query_embedding = model.encode([query], convert_to_numpy=True)

        # Search in FAISS index
        _, indices = index.search(query_embedding, k=3)  # Retrieve top 3 chunks

        results = []
        for idx in indices[0]:
            file_name = metadata[idx]["file"]
            chunk_id = metadata[idx]["chunk_id"]
            text_chunk = documents[idx]
            results.append(f"📄 **{file_name}** (Part {chunk_id+1}):\n{text_chunk}...")  

        if results:
            response = "Here are the most relevant excerpts from class materials:\n\n" + "\n\n".join(results)
        else:
            response = "I couldn't find relevant class materials for your query."

        dispatcher.utter_message(text=response)
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


import google.generativeai as genai
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import os

# Set up Google Gemini API Key
GENAI_API_KEY = "AIzaSyCFVTmc30L2AZ76WaEC3VWrrsMGwbKDYpM"
genai.configure(api_key=GENAI_API_KEY)

class ActionGenerateAnswer(Action):
    def name(self):
        return "action_generate_answer"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_query = tracker.latest_message.get("text")

        try:
            # Call Gemini API
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(user_query)

            # Send Gemini's response back to user
            dispatcher.utter_message(text=response.text)

        except Exception as e:
            dispatcher.utter_message(text="Sorry, I couldn't process that request.")
            print(f"Error: {e}")

        return []
