# Maintainers
abrie@cape-ai.com
---
# The Super Amazing NLP Twitter Insights Project
Welcome Grasshopper. So you have come in search of the mystical skills needed to become an NLP Ninja? Well I am pleased to tell you that you have come to the right place. By going through this project, you will develop all the essential skills needed to become a true NLP warrior.

## Project Structure
- [Who is this project for?](#who-is-this-project-for)
- [Getting started](#getting-started)
- [Progressing through your belts](#progressing-through-your-belts)

## Who is this project for?
This project is for any young grasshopper who wants to immerse themselves into the field of NLP in a super practical way. You will learn to get comfortable handling text data and extracting meaningful information from it. There are no formal prerequisite for this project but if you are comfortable with Python and are prepared to do some deep-dive reading into all the shared links, you should be ready to go!

__WARNING: This project is not for the faint-hearted and will require a significant amount of time, dedication and perseverance. But I can promise you, the fruits will be worth the sacrifice.__

## Getting started
* Clone the repository on the `nlp-twitter-insights` branch to your local file system (preferably with SSH):
```commandline
git@gitlab.com:cape-ai/cape-ai-projects.git
```
* Create a new virtual environment with Python version `>=3.9` and pip version `>=20.2.2`.
* Give yourself a pat on the back for choosing to embark on this quest.

## Progressing through your belts

Grasshopper, to become an NLP Ninja, you need to complete a set of tasks for me. Each task you complete will earn you a belt. During your training, you will pick up valuable skills that will help you along your journey. To give you an overview of what the path looks like going ahead, here is the list of tasks at a glance:

1. <span style="color:rgb(212, 212, 212)">__White belt task__</span>: Use the Twitter API to gather tweets about a topic of your interest
2. <span style="color:yellow">__Yellow belt task__</span>: Create a wordcloud of all the nouns/verbs in the tweets
3. <span style="color:orange">__Orange belt task__</span>: Construct a vector representation for each tweet and view the embedding space
4. <span style="color:green">__Green belt task__</span>: Use the embedding space to do some topic modelling
5. <span style="color:purple">__Purple belt task__</span>: Use a pretrained model to analyse the tweet sentiment
6. <span style="color:blue">__Blue belt task__</span>: Extract entities from the tweets using a pretrained NER model
7. <span style="color:brown">__Brown belt task__</span>: Link the entities you uncovered to the appropriate Wikipedia page
8. <span style="color:red">__Red belt task__</span>: Finetune a BERT model to do text classification
9. <span style="color:black">__Black belt task__</span>: Present your insights in a Streamlit dashboard hosted on AWS

Stay nimble, young grasshopper, and let us begin your training.

______________________________________________________________________________________

<details>
  <summary style="color:rgb(212, 212, 212);font-size:20px;font-weight: bold;">1. White belt task</summary>
  
Grasshopper, the first task I would like you to complete is to gather tweets about a topic of your interest. To earn your white belt, you must:

1. Decide what topic you want to gather data on. I'd recommend choosing one that is perhaps related to current world events and that people are currently tweeting about. Gather a good amount of tweets, perhaps around 1 000 or so should be good.
2. Write a function called `get_tweets` that takes a list of `hashtags` and output all tweets containing those hashtags in a new folder called `data`. Choose a sensible timeframe from today backwards (say a week or a month) with which to gather tweets and store the results in JSON format.

__Useful info:__

- The [Twitter v2 API](https://developer.twitter.com/en/docs/twitter-api/early-access) has recently been released and, if we sign up for a developers account, we should be able to get access to the tweets we seek.
- A fellow NLP Ninja has been gracious enough to equip us with [a useful tool](https://github.com/JeannieDaniel/twitterati) to get us on our way.
- The function may be structured as follows:
```python
import json
import datetime
import twitterati


def get_tweets(hashtags: list[str],
               period: int):
    """
    A function to retrieve all tweets from a historical period.
    
    :param hashtags: The list of hashtags to search for
    :param period: The number of days before today to search for tweets
    """
    
    # Cool stuff goes here
```

<details>

  <summary style="font-size:15px;font-weight: bold;">I'm finished, Senpai!</summary>

I am proud of you, young grasshopper. You have earned this white belt.

![white_belt](pictures/white_belt.jpg)

</details>

</details>

______________________________________________________________________________________

<details>
  <summary style="color:yellow;font-size:20px;font-weight: bold;">2. Yellow belt task</summary>

Grasshopper, I'd like you to now create a wordcloud using the tweets you have acquired. To earn your yellow belt, you must:

1. Create a function called `show_wordcloud` that is able to take in a list of tweet texts and output a wordcloud image of the most frequently used words in the tweets.
2. This magic function should also provide us with the ability to filter the type of words that get displayed by their parts of speech (e.g. noun, verb, adjective).

__Useful info:__

- The [spaCy library](https://spacy.io/) is the Swiss army knife that every NLP Ninja should have in their back pocket. In this case, it might come in handy when trying to separate words into their different parts of speech.
- Wordclouds are often the first visualisation an NLP Ninja goes for to get an aggregated view of what a body of text contains. But never try and write anything from scratch! There are plenty of implementations [like this one](https://github.com/amueller/word_cloud) that can help us out.
- There are [many different types of text visualisations](https://textvis.lnu.se/) — perhaps give them a browse and maybe one of them comes in handy in the future.
- The function may be structured as follows:
```python
import spacy
import wordcloud


def show_wordcloud(text: list[str],
                   pos: list[str]):
    """
    A function to display a wordcloud of one or more parts of speech.
    
    :param text: The list of tweet texts
    :param pos: One or more parts of speech tags to include in the wordcloud
    """
    
    # Cool stuff goes here
```

<details>

  <summary style="font-size:15px;font-weight: bold;">I'm finished, Senpai!</summary>

I am proud of you, young grasshopper. You have earned this yellow belt.

![yellow_belt](pictures/yellow_belt.jpg)

</details>

</details>

______________________________________________________________________________________

<details>
  <summary style="color:orange;font-size:20px;font-weight: bold;">3. Orange belt task</summary>

Grasshopper, I'd like you to now construct a vector representation for each tweet in your dataset and view what this embedding space looks like. To earn your orange belt, you must:

1. Create a function called `get_tweet_embedding` that can take in a list of tweet texts and output a set of vector representations of the text. This function should ideally make use of a pretrained text embedding model — please don't train your own!
2. Create an additional function called `visualise_embedding` that takes the vector representation from the previous function as input, reduces the dimensionality of these vector points to two dimensions and plots them. We will likely get a much better idea of what this embedding space looks like after this!

__Useful info:__

- There are more than one way to convert a body of text into a vector representation. If you explore the [spaCy documentation](https://spacy.io/api/doc), you may come across a more simplistic way to do this. But what I'd really recommend is checking out how to use [more advanced transformer-based methods](https://github.com/UKPLab/sentence-transformers), as these typically produce more high-quality embeddings.
- The embedding space of the text will likely be much larger than two dimensions. Perhaps we can employ a dimensionality reduction technique like [UMAP](https://umap-learn.readthedocs.io/en/latest/) to get a better feel of what the embedding space looks like.
- I'd recommend using [Plotly](https://plotly.com/python/line-and-scatter/) for your visualisation as it provides us with interactive functionality which will prove useful later on.
- The function may be structured as follows:
```python
import numpy as np
import spacy
import sentence_transformers
import umap


def get_tweet_embedding(text: list[str]) -> list[np.array]:
    """
    A function to embed a set of tweets into an embedding space.
    
    :param text: The list of tweet texts
    :return: The set of embedding vectors
    """
    
    # Cool stuff goes here
    
    return embedding

def visualise_embedding(vectors: list[np.array]):
    """
    A function to reduce and visualise an embedding space.
    
    :param vectors: A set of vectors to visualise
    """
```

<details>

  <summary style="font-size:15px;font-weight: bold;">I'm finished, Senpai!</summary>

I am proud of you, young grasshopper. You have earned this orange belt.

![orange_belt](pictures/orange_belt.jpg)

</details>

</details>

______________________________________________________________________________________

<details>
  <summary style="color:green;font-size:20px;font-weight: bold;">4. Green belt task</summary>

Grasshopper, I'd like you to do some [topic modelling](https://en.wikipedia.org/wiki/Topic_model) on the tweets. To earn your green belt, you must:
  
1. Create a function called `get_topics` that takes in a list of tweets (or perhaps their vector representations?) and outputs a list of topic clusters.

__Useful info:__

- There are many different topic modelling approaches. Probably the most well-known is a technique called *[Latent Dirichlet Allocation](https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24)* (LDA). But what I really want you to do is use [a more recent method](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6) that leverages BERT sentence embeddings (your orange belt should come in handy here).
- The function may be structured as follows:
```python
import numpy as np
import umap
import hdbscan


def get_topics(vectors: list[np.array]) -> list[list[tuple[str, float]]]:
    """
    A function to uncover the topics present in a text embedding.
    
    :param vectors: A set of embedding vectors
    :return: The uncovered topics and their associated scores
    """
    
    # Cool stuff goes here
    
    return topic_clusters
```

<details>

  <summary style="font-size:15px;font-weight: bold;">I'm finished, Senpai!</summary>

I am proud of you, young grasshopper. You have earned this green belt.

![green_belt](pictures/green_belt.jpg)

</details>

</details>

______________________________________________________________________________________

<details>
  <summary style="color:purple;font-size:20px;font-weight: bold;">5. Purple belt task</summary>

Grasshopper, we're going to be doing some fun stuff now — I'd like you to do some [sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis) on your tweets. To earn your purple belt, you must:

1. Create a function called `get_sentiment` that takes in a tweet text and outputs a sentiment score between -1 and 1 indicating how negative or positive a tweet is, respectively.
2. Use this function to enhance your `show_wordcloud` function by colouring words according to the average sentiment of the tweets containing those words. Make positive words more green and negative words more red.

__Useful info:__

- The [Hugging Face Transformers library](https://github.com/huggingface/transformers) is arguably the most useful tool for any NLP practitioner. They provide a platform for anyone to easily utilize SOTA transformer models made available by big corporations like Google, Facebook, Microsoft, to name but a few. Perhaps give their [list of models](https://huggingface.co/models) a browse and see if you can find one that has been pretrained for sentiment analysis.
- It may also be worth checking out the [pipeline functionality](https://huggingface.co/transformers/main_classes/pipelines.html) provided by the transformers package — if you're just leveraging a pretrained model and not doing some fancy finetuning, this function makes things very easy!
- The function may be structured as follows:
```python
import transformers


def get_sentiment(text: str) -> float:
    """
    A function to predict the sentiment of a piece of text.
    
    :param text: The text
    :return: The sentiment score
    """
    
    # Cool stuff goes here
    
    return sentiment_score
```

<details>

  <summary style="font-size:15px;font-weight: bold;">I'm finished, Senpai!</summary>

I am proud of you, young grasshopper. You have earned this purple belt.

![purple_belt](pictures/purple_belt.jpg)

</details>

</details>

______________________________________________________________________________________

<details>
  <summary style="color:blue;font-size:20px;font-weight: bold;">6. Blue belt task</summary>

Grasshopper, we're now going to do some [_Named Entity Recognition_](https://en.wikipedia.org/wiki/Named-entity_recognition) (NER). Extract entities from the tweets using a pretrained NER model. To earn your blue belt, you must:

1. Create a function called `get_named_entities` that takes in a tweet text and outputs a list of entities with their associated label.
2. Further enhance your `show_wordcloud` function by allowing us to view only words that correspond to a particular entity class.

__Useful info:__

-  When doing NER, you can leverage [spaCy's NER functionality](https://spacy.io/api/entityrecognizer) or get more fancy using [a pretrained model](https://huggingface.co/models?pipeline_tag=token-classification) off Hugging Face. But what I'd really recommend is using a package called [Flair](https://github.com/flairNLP/flair) as they tend to produce SOTA results for NER tasks.
- The function may be structured as follows:
```python
import flair


def get_named_entities(text: str) -> list[tuple[str, str]]:
    """
    A function to identify any named entities present in a piece of text.
    
    :param text: The text
    :return: The list of entities and their associated class
    """
    
    # Cool stuff goes here
    
    return named_entities
```

<details>

  <summary style="font-size:15px;font-weight: bold;">I'm finished, Senpai!</summary>

I am proud of you, young grasshopper. You have earned this blue belt.

![blue_belt](pictures/blue_belt.jpg)

</details>

</details>

______________________________________________________________________________________

<details>
  <summary style="color:brown;font-size:20px;font-weight: bold;">7. Brown belt task</summary>

Grasshopper, I'd now like you to link the entities you uncovered to the appropriate Wikipedia page. To earn your brown belt, you must:

1. Write a function called `link_entities` that takes in a tweet text and outputs a list of entities that have a Wikipedia page.

__Useful info:__

- I'd recommend checking out an API called [DBpedia Spotlight](https://www.dbpedia-spotlight.org/api) for all your entity linking needs.
- The function may be structured as follows:
```python
import requests


def link_entities(text: str) -> list[dict]:
    """
    A function to link any named entities present in a piece of text to their appropriate Wikipedia page.
    
    :param text: The text
    :return: The list of entities and their associated Wikipedia page
    """
    
    # Cool stuff goes here
    
    return wiki_links
```

<details>

  <summary style="font-size:15px;font-weight: bold;">I'm finished, Senpai!</summary>

I am proud of you, young grasshopper. You have earned this brown belt.

![brown_belt](pictures/brown_belt.jpg)

</details>

</details>

______________________________________________________________________________________

<details>
  <summary style="color:red;font-size:20px;font-weight: bold;">8. Red belt task</summary>

Grasshopper, this is going to be quite a tough task, but I know you are more than capable. I'd like you to finetune a pretrained BERT model to do [text classification](https://developers.google.com/machine-learning/guides/text-classification). To earn your red belt, you must:

1. Finetune your own BERT model using a [`bert-base-cased`](https://huggingface.co/bert-base-cased) model from Hugging Face as a starting point. Make sure your best model checkpoints are saved during your training.
2. Write a function called `predict_toxic` that takes in a tweet text and outputs a list of toxicity tags the text subscribes to.

__Useful info:__

- Obviously you need some labelled training data. For this, take a look at the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) on Kaggle.
- [Venelin Valkov](https://www.youtube.com/watch?v=wG2J_MJEjSQ&ab_channel=VenelinValkov) also does a great job at showing you how to finetune your own BERT model using [Pytorch Lightning](https://www.pytorchlightning.ai/). He's also been so gracious enough to have complied a [complete tutorial and notebook](https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/) for us — what a champ!
- If your PC does not have a GPU, consider training your model using [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb). You can connect your notebook to a free GPU instance by selecting `Runtime` → `Change runtime type` → `GPU`.
- The function may be structured as follows:
```python
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from transformers import BertTokenizerFast as BertTokenizer


# Probably a lot of copy-pasted stuff here

def predict(text: str) -> list[str]:
    """
    A function to predict the toxicity tags of a piece of text.
    
    :param text: The text
    :return: The list of toxicity tags
    """
    
    # Cool stuff goes here
    
    return toxicity_tags
```

<details>

  <summary style="font-size:15px;font-weight: bold;">I'm finished, Senpai!</summary>

I am proud of you, young grasshopper. You have earned this red belt.

![red_belt](pictures/red_belt.jpg)

</details>

</details>

______________________________________________________________________________________

<details>
  <summary style="color:black;font-size:20px;font-weight: bold;">9. Black belt task</summary>

Grasshopper, I'd now like you to collect everything you have learnt and present your insights in an interactive dashboard. To earn your black belt, your dashboard must include:

1. A tab called `Wordcloud` for viewing the wordcloud visualisations created by your `show_wordclouds` function. Give users the ability to:
   1. Filter words based on their part of speech or entity class they subscribe to.
   2. Colour words according to the average sentiment of their origin tweets.
2. A tab called `Embedding` for viewing the two-dimensional embedding produced by your `visualise_embedding` function. Give users the ability to:
   1. Select one or more toxicity tags which should then highlight any tweet embeddings corresponding to these tags.
   2. Colour tweets according to the HDBSCAN topic cluster they subscribe to.
3. A tab called `Entities` that showcases your models. Allow a user to input their own text and then: 
   1. Display the named entities uncovered as well as any links to the entity's corresponding Wikipedia page (if applicable).
   2. Display what toxicity tag(s) the piece of text subscribes to.
4. Host your dashboard on the cloud and share the dashboard link with a fellow NLP Ninja to assess.
  
__Useful info:__

- As for the dashboard, I'd highly recommend reading up on [Streamlit](https://streamlit.io/) and all it's functionality. You'll be able to easily get a professional-looking dashboard set up in no time.
- For hosting, you may choose to spin up your own [EC2 instance](https://af-south-1.signin.aws.amazon.com/oauth?response_type=code&client_id=arn%3Aaws%3Aiam%3A%3A015428540659%3Auser%2Fec2&redirect_uri=https%3A%2F%2Faf-south-1.console.aws.amazon.com%2Fec2%2Fv2%2Fhome%3Fregion%3Daf-south-1%26state%3DhashArgs%2523Instances%253A%26isauthcode%3Dtrue&forceMobileLayout=0&forceMobileApp=0&code_challenge=m_joBVgOD3HWUHc_uO-5WFFFTH-CITEQG3ce3E9r10w&code_challenge_method=SHA-256) on AWS (__IMPORTANT: Please contact someone from the Tech Leadership Team and let them know you are about to do this!__). Alternatively, you can also host a free dashboard using [Heroku](https://towardsdatascience.com/a-quick-tutorial-on-how-to-deploy-your-streamlit-app-to-heroku-874e1250dadd).

<details>

  <summary style="font-size:15px;font-weight: bold;">I'm finished, Senpai!</summary>

I am proud of you, young grasshopper. You have earned this black belt and the title of __NLP Ninja__. Well done.

![black_belt](pictures/black_belt.jpg)

</details>

</details>
