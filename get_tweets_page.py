from utilities.tweets import get_tweets
import streamlit as st

class GetTweetsPage:
    def __init__(self, hashtags, period, max_count, additional_options):
        self.hashtags = hashtags
        self.period = period
        self.max_count = max_count
        self.additional_options = GetTweetsPage.parse_additional_query_options(additional_options)

    def send_tweet_request(self):
        tweets = None
        try:
            with st.spinner("Getting tweets..."):
                tweets = get_tweets(hashtags=self.hashtags, period=self.period, **self.additional_options,
                                    max_count=self.max_count)
            st.write(f"{len(tweets)} tweets retrieved successfully!")
        except Exception as e:
            st.write("Oh no! Something went wrong... Error: ", e)
            print(e)
        return tweets

    @staticmethod
    def parse_additional_query_options(additional_options):
        query_options = dict(remove_retweets=True, remove_replies=True, remove_quote=True)
        for option in additional_options:
            option = option.lower()

            for key, value in query_options.items():
                if option == key.split("_")[1]:
                    query_options[key] = False

        return query_options
