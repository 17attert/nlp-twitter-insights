"""Script that generates the NLP Twitter Insights Dashboard.
"""
import streamlit as st

from embedding_page import EmbeddingPage
from get_tweets_page import GetTweetsPage
from wordcloud_page import WordCloudPage
from toxicity_tagger.toxic_comment_tagger import LABEL_COLUMNS
from toxicity_tagger.toxicity import get_toxicity
from tweets import Tweets

from utilities.dashboard import parse_mutliselect, get_named_entities_for_display, NE_MAP, POS_MAP
from utilities.named_entities import link_entities

if __name__ == "__main__":
    st.title('NLP Twitter Insights Dashboard')

    add_radio = st.sidebar.radio("Choose one", ["Get Tweets", "Wordcloud", "Embeddings", "Entities"])

    tweets = Tweets()

    if add_radio == "Get Tweets":
        st.subheader("Retrieve Tweets using the Twitter API")

        hashtags = st.text_input('Input a list of hashtags that will be used to query the Twitter API. '
                                 'Separate your list with commas.',
                                 placeholder="#F1")

        # Clean hashtags
        hashtags = hashtags.split(",")
        if len(hashtags) > 1:
            hashtags = [hashtag.strip() for hashtag in hashtags]

        st.write("Your hashtags are: ", hashtags)

        period = st.number_input('Input the number of previous days to return tweets and '
                                 'metadata for (maximum is 6 days)', value=1, min_value=1,
                                 max_value=6, format="%i")

        max_count = st.number_input('Input the maximum number tweets you would like to retrieve', value=100, min_value=100,
                                    max_value=1000, format="%i")

        additional_options = st.multiselect(
            'Select additional types of tweets you wish to KEEP. Default behaviour is to remove these...',
            ["Retweets", "Replies", "Quotes"])

        get_tweets_page = GetTweetsPage(hashtags, period, max_count, additional_options)

        if st.button("Get Tweets!", help="Click me send your Twitter API request!"):
            query_result = get_tweets_page.send_tweet_request()
            for key, result in query_result.items():
                tweets[key] = result

    elif add_radio == "Wordcloud":
        st.subheader("See Your Tweets Wordcloud")

        if tweets:
            pos_options = st.multiselect(
                'Select your parts of speech...',
                POS_MAP.keys())

            ne_options = st.multiselect(
                'Select your named entities...',
                NE_MAP.keys())

            remove_stopwords = st.selectbox(
                'Remove stopwords from the wordcloud?', ('Yes', 'No'), index=0,
                help="Should words like 'a', 'the', 'is', and 'are' be removed from the analysis?")

            colour_by_sentiment = st.selectbox(
                'Colour the wordcloud by sentiment?', ('Yes', 'No'), index=1,
                help="Should sentiment analysis be performed on the uploaded tweets and the result be used to \
                    color the wordcloud?")

            top_n = st.number_input('Input the number of words you wish to see in your cloud', value=100,
                                    min_value=0,
                                    max_value=1000, format="%i")

            if st.button("Generate Wordcloud", help="Click me to generate your wordcloud!"):
                pos = parse_mutliselect(pos_options, POS_MAP)
                ne = parse_mutliselect(ne_options, NE_MAP)

                with st.spinner("Generating wordcloud..."):
                    wordcloud_page = WordCloudPage(tweets, pos=pos, remove_stopwords=(remove_stopwords == "Yes"),
                                                   top_n=top_n, colour_by_sentiment=(colour_by_sentiment == "Yes"),
                                                   named_entities=ne)

                    wordcloud_page.show_wordcloud()
        else:
            st.write("Navigate to the Get Tweets page and get some tweets first!")

    elif add_radio == "Embeddings":
        st.subheader("Visualise Your Tweet Embeddings")

        if tweets:
            toxicity_options = []

            colour_by = st.selectbox(
                'Highlight the embedding plot by...', ('toxicity', 'cluster'), index=1,
                help="Should the embeddings by highlighted according to the toxicity tags the tweets subscribe"
                     "to? Or should the embeddings be highlighted by the topic cluster the tweets belong to?")

            if colour_by == "toxicity":
                toxicity_options = st.multiselect(
                    'Select your toxicity tags...',
                    LABEL_COLUMNS)

            if st.button("Visualise Embeddings", help="Click me to visualise your tweet embedding space!"):
                embedding_page = EmbeddingPage(tweets)

                if toxicity_options:
                    embedding_page.visualise_embeddings(selected_toxicity=toxicity_options)
                elif colour_by == "cluster":
                    embedding_page.visualise_embeddings(selected_toxicity=[])
                else:
                    st.write("Please select at least one toxicity tag!")
        else:
            st.write("Navigate to the Get Tweets page and get some tweets first!")

    elif add_radio == "Entities":
        st.subheader("Test the Model Suite")

        text_input = st.text_input('Input some text you would like named entity recognition and toxicity tagging '
                                   'performed on...')

        if st.button("Tag Text", help="Click me to tag your input text!"):
            with st.spinner("Tagging named entities..."):
                named_entity_str, sentence = get_named_entities_for_display(text_input)
                if named_entity_str != "":
                    st.write("Named entity tagged input text:")
                    st.write(named_entity_str)

                    with st.spinner("Linking named entities..."):
                        linked_entities_df = link_entities(sentence)
                        st.write("Linked entities:")
                        if not linked_entities_df.empty:
                            st.dataframe(linked_entities_df)
                        else:
                            st.write("No links found!")
                else:
                    st.write("No named entities tagged!")

            with st.spinner("Tagging toxicity..."):
                toxicity_tags = get_toxicity([text_input])[0]
                if len(toxicity_tags) > 0:
                    st.write(f"Toxicity tags: {', '.join(toxicity_tags)}")
                else:
                    st.write("Hooray! No toxicity was detected.")
