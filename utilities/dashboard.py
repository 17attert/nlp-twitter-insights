POS_MAP = {
    "nouns": ["NN", "NNS", "NNP", "NNPS"],
    "verbs": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
    "adjectives": ["JJ", "JJR", "JJS"],
    "adverbs": ["RB", "RBR", "RBS"]
}

NE_MAP = {
    "persons": "PER",
    "locations": "LOC",
    "organisations": "ORG",
    "miscellaneous": "MISC"
}

from flair.data import Sentence
from flair.models import SequenceTagger


def parse_mutliselect(selections, mapping_dict):
    """Parse the parameters selected on the dashboard to the same form as what the POS and NER models output.
    Parameters on the dashboard are a more user-friendly version of what the POS and NER models accept.
    Args:
        selections: (list) Raw selections from the dashboard.
        mapping_dict: (dict) A dictionary containing a mapping from the dashboard parameters to the parameters accepted
                             by the POS or NER models.

    Returns: (list) A list of parameters that can be used to filter output from the NER or POS models.

    """
    params = []
    for selection in selections:
        mapped_selection = mapping_dict[selection]
        if isinstance(mapped_selection, str):
            params.append(mapped_selection)
        else:
            params.extend(mapped_selection)
    return params


def get_named_entities_for_display(text):
    """
    Infer named entities from a user-specified string.
    Args:
        text: Text that the user would like NER tagged.

    Returns: (tuple): (The string with Named Entities tagged (if named entities were found), Flair sentence object).

    """
    # Load the NER tagger
    tagger = SequenceTagger.load('ner')
    sentence = Sentence(text)

    tagger.predict(sentence)

    if text == sentence.to_tagged_string():
        return "", None

    else:
        return sentence.to_tagged_string(), sentence
