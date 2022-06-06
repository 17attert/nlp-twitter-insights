import json
import datetime
import twitterati
from typing import List


def get_tweets(hashtags: List,
               period: int):
    """
    A function to retrieve all tweets from a historical period, specified in days. 
    
    :param hashtags: The list of hashtags to search for
    :param period: The number of days before today to search for tweets
    """
    
    # Query Twitter API for tweets


def save_tweets(hashtags: List):

    pass


if __name__ == "__main__":

    print("Main file executing...")

    hashtags = [
        "johnny depp amber heard trial"
        "johnny depp",
        "amber heard",
        "defamation trial"
    ]
    period = 30
    tweets = get_tweets(hashtags, period)
    save_tweets(tweets)