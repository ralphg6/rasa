import json
import logging
import re
import typing
from typing import Any, Dict, Text

from rasa.nlu.training_data.formats.readerwriter import JsonTrainingDataReader

if typing.TYPE_CHECKING:
    from rasa.nlu.training_data import Message, TrainingData

logger = logging.getLogger(__name__)

class WatsonReader(JsonTrainingDataReader):
       

    def read_from_json(self, js: Dict[Text, Any], **kwargs: Any) -> "TrainingData":
        """Loads training data stored in the Watson Assistant data format."""
        from rasa.nlu.training_data import Message, TrainingData

        entity_synonyms, regex_features = WatsonReader._extract_entities(js)

        training_examples = WatsonReader._extract_intents(js)

        return TrainingData(
            training_examples,
            entity_synonyms,
            regex_features,
        )

    @staticmethod
    def _extract_entities(js):
        entity_synonyms = {}
        regex_features = []
        for entity in js.get('entities', []):
            entity_name = entity.get('entity')
            for value in entity.get('values', []):
                if (value.get('type') == "synonyms"):
                    entity_synonyms[value.get('value')] = entity_name
                    for synonym in value.get("synonyms", []):
                        entity_synonyms[synonym] = entity_name
                if (value.get('type') == "patterns"):
                    for pattern in value.get("patterns", []):
                        regex_features.append(
                            {"name": entity_name +'_' +value.get('value'), "pattern": '%s' % pattern}
                        )
            
        return entity_synonyms, regex_features

    @staticmethod
    def _extract_intents(js):
        from rasa.nlu.training_data import Message

        training_examples = []

        for intent in js.get("intents", []):
            intentName = intent.get('intent')
            for text in intent.get('examples', []):
                message = Message(text.get('text'), {"intent": intentName})
                training_examples.append(message)

        return training_examples
