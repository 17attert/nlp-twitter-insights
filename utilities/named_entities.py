from flair.data import Sentence
from flair.models import SequenceTagger
import pandas as pd
import requests

DB_SPOTLIGHT = "https://api.dbpedia-spotlight.org/en/annotate"


def request_dbspotlight(row):
    """Send a GET request to DBSpotlight with a Named Entity to search.
    Args:
        row: Row of a pandas.DataFrame to apply the function to.

    Returns:
        (str) Link to the DBSpotlight page corresponding to the named entity.
    """
    try:
        entity = row['Entity']

        # Make the request
        r = requests.get(DB_SPOTLIGHT, params={"text": entity}, headers={"accept": "application/json"})
        # Parse response
        link = r.json()["Resources"][0]['@URI']

        return link
    except KeyError:
        return None


def link_entities(sentence, thresh=0.5):
    """
    A function to link any named entities present in a piece of text to their appropriate DBSpotlight page.
    Args:
        sentence: (str) The Flair sentence object that has already been passed to the NER tagger.
        thresh: (float) A confidence threshold for filtering out uncertain Named Entity predictions.

    Returns:
        (pandas.DataFrame) A DataFrame containing the Named Entities extracted from the sentence together
                           with the DBSpoltight links to the Named Entities (if they exist).
    """
    # Extract the names entities
    named_entities = []
    entities = sentence.to_dict(tag_type='ner')['entities']

    for item in entities:
        entity = item['text']
        label = item['labels'][0].to_dict()['value']
        conf = item['labels'][0].to_dict()['confidence']
        named_entities.append({"Entity": entity, "Label": label, "Confidence": conf})
    named_entities = pd.DataFrame(named_entities)

    # Filter named entities by confidence
    named_entities = named_entities[named_entities['Confidence'] > thresh][['Entity', 'Label']]

    if not named_entities.empty:
        # Link the entities with the DBSpotlight page
        named_entities['link'] = named_entities.apply(request_dbspotlight, axis=1)

    return named_entities


def get_named_entities(texts, thresh=0.8):
    """
    A function to identify any named entities present in pieces of text.
    Args:
        texts: (list) A list of clean texts.
        thresh: (float) A confidence threshold for filtering NER predictions.

    Returns:
        (pandas.DataFrame) A DataFrame containing all named entities found. (The word, the type of entity, and the
                           confidence of the prediction).
    """

    # Load the NER tagger
    tagger = SequenceTagger.load('ner')

    # Create a list of sentences
    sentences = [0] * len(texts)
    for i, text in enumerate(texts):
        sentences[i] = Sentence(text)

    # Run NER over sentences
    tagger.predict(sentences, mini_batch_size=64)

    # Extract the names entities
    named_entities = []
    for sentence in sentences:
        entities = sentence.to_dict(tag_type='ner')['entities']
        for item in entities:
            entity = item['text']
            label = item['labels'][0].to_dict()['value']
            conf = item['labels'][0].to_dict()['confidence']
            # Keep only single-word named entities
            if len(entity.split()) == 1 and conf > thresh:
                named_entities.append({"word": entity, "ne": label, "ne_conf": conf})

    named_entities = pd.DataFrame(named_entities)

    # Return only the most confident entity tag per word
    named_entities = named_entities.groupby(by=['word', 'ne']).agg({'ne_conf': 'mean'}).reset_index()
    named_entities = named_entities.sort_values(by=['word', 'ne_conf'], ascending=[True, False])
    named_entities['most_conf_entity'] = named_entities.groupby(by='word')['ne'].transform('first')
    named_entities = named_entities[named_entities['ne'] == named_entities['most_conf_entity']]
    named_entities = named_entities[['word', 'ne', 'ne_conf']]

    return named_entities
