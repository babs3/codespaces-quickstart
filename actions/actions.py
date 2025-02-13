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
    