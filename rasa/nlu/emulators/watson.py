from typing import Any, Dict, Text

from rasa.nlu.emulators.no_emulator import NoEmulator


class WatsonEmulator(NoEmulator):
    def __init__(self) -> None:

        super(WatsonEmulator, self).__init__()
        self.name = "watson"

    def normalise_request_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Transform data from watson format."""
        _data = {
            "text": data["input"]['text']
        }

        return _data

    def normalise_response_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Transform data to watson format."""
        return {
            "intents": [
                {
                    "intent": el["name"], 
                    "confidence": el["confidence"]
                } for el in data["intent_ranking"]],
            "entities": [
                {
                    "entity": e["entity"],
                    "location": [
                        e.get("start"),
                        (e["end"] - 1) if "end" in e else None
                    ],
                    "value": e["value"],
                    "confidence": e.get("confidence"),
                } for e in data["entities"]
            ],
            "input": {
                "text": data["text"]
            }
        }