from utilities.tweets import clean_tweet_texts, create_word_freq_df, tweet_to_wordlist
from utilities.named_entities import get_named_entities
from utilities.sentiment import score_dataset_with_sentiment
from utilities.plots import word_sentiment_to_colour
import wordcloud
from utilities.grouped_colour_func import GroupedColourFunc
import matplotlib.pyplot as plt
import streamlit as st
import json

class WordCloudPage:
    def __init__(self, tweets, pos=[], remove_stopwords=True, top_n=None, colour_by_sentiment=False, named_entities=[]):
        # Import and clean raw tweets
        self.tweets_df = clean_tweet_texts(tweets)

        # Create tokens from cleaned texts
        self.tweets_df['tokens'] = self.tweets_df['text'].apply(tweet_to_wordlist, args=(pos, remove_stopwords))

        # Create dataframe that will be used for creating the wordcloud
        self.word_freq_df = create_word_freq_df(self.tweets_df)

        self.colour_by_sentiment = colour_by_sentiment

        if colour_by_sentiment:
            # Get average sentiment per unique word
            with st.spinner("Scoring sentiment..."):
                self.word_freq_df = score_dataset_with_sentiment(self.tweets_df, self.word_freq_df)

        if named_entities:
            # Add NEs
            with st.spinner("Tagging named entities..."):
                nes = get_named_entities(self.tweets_df.text)
            self.word_freq_df = self.word_freq_df.merge(nes[['word', 'ne', 'ne_conf']], on='word', how='left').dropna(axis=0).reset_index(drop=True)

            # Filter by supplied NEs
            ne_mask = [entity in named_entities for entity in self.word_freq_df['ne']]
            self.word_freq_df = self.word_freq_df[ne_mask]

        # Retrieve top N words (if argument supplied)
        if top_n:
            try:
                self.word_freq_df = self.word_freq_df.sort_values(by='count', ascending=False).reset_index(drop=True)[
                                    :top_n]
            except IndexError:
                self.word_freq_df = self.word_freq_df.sort_values(by='count', ascending=False).reset_index(drop=True)

        # Create freqs
        self.freqs = {}
        for i in self.word_freq_df.index:
            self.freqs[self.word_freq_df.iloc[i]['word']] = self.word_freq_df.iloc[i]['count']

    def show_wordcloud(self):
        """
        A function to display a wordcloud of one or more parts of speech.

        Returns:

        """
        wc = wordcloud.WordCloud(width=1200, height=800, background_color="white", max_words=200, min_font_size=10) \
            .generate_from_frequencies(self.freqs)

        if self.colour_by_sentiment:
            # Create colour categories for each level of sentiment
            self.word_freq_df['colour'] = self.word_freq_df.apply(word_sentiment_to_colour, axis=1)

            # Create colour to word mapping
            colour_to_words = {}
            for colour in self.word_freq_df.colour.unique():
                colour_to_words[colour] = list(self.word_freq_df[self.word_freq_df['colour'] == colour]['word'].values)

            default_colour = 'grey'

            grouped_colour_func = GroupedColourFunc(colour_to_words, default_colour)

            wc.recolor(color_func=grouped_colour_func)

        fig = plt.figure(figsize=(16, 12))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        st.pyplot(fig)


if __name__ == "__main__":
    with open("/home/samuels/Documents/CapeAI/projects/nlp-twitter-insights/data/tweets-#f1 -is:retweet -is:reply -is:quote lang:en-None.json") as fp:
        tweet_texts = json.load(fp)
    page = WordCloudPage(
        tweet_texts,
        top_n=50,
        colour_by_sentiment=True,
        named_entities=["LOC"]
    )

    page.show_wordcloud()
