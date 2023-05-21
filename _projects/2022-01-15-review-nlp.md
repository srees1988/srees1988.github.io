---
title: 'Audience Text Mining'
subtitle: 'Customer Feedback Analysis using Pre-trained Classifiers'
date: 2022-01-15 00:00:00
featured_image: '/images/projects/5.reviews_nlp/1.nlp.jpg'
---

<style>
body {
text-align: justify}
</style>

### Introduction

In today's world, businesses are inundated with vast amounts of unstructured data, including user reviews, social media posts, and customer feedback. The challenge is in extracting valuable insights from this unstructured data to make informed decisions. This is where topic modeling, summarization, keyword extraction, and sentiment analysis come in. With these techniques, you can analyze large volumes of unstructured data and uncover hidden patterns and trends.

However, until a few years ago, performing these NLP techniques involved labor-intensive manual tagging and classification. Thankfully, with the advent of pre-trained classifiers, we can now get ready-made solutions that make analyzing unstructured data a breeze.

In this blog post, I will walk you through a step-by-step guide on how to perform these NLP techniques using pre-trained classifiers ([Hugging Face Zero-Shot Classifier](https://huggingface.co/facebook/bart-large-mnli)) on a [public Yelp reviews dataset](https://github.com/srees1988/reviews-nlp-py) using Python. By the end of this tutorial, you will have gained practical knowledge that you can apply to your own datasets to gain valuable insights. So, let's get started!


![](/images/projects/5.reviews_nlp/2.sentiment_analysis.JPG)


### Data Preparation:

To begin analyzing the unstructured customer review data, we will utilize a publicly available [YELP reviews dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) from Kaggle, which is accessible via the [GitHub Repository](https://github.com/srees1988/reviews-nlp-py) used in this tutorial. However, before we proceed, it is necessary to preprocess the data to ensure it is in a suitable format for analysis.

In this section, I will present the Python code we used to clean and prepare the data, along with explanatory comments and a brief summary of the key steps, to facilitate comprehension. So, without further ado - let's get this show on the road!
 
 
### Step 0:Restart the session and clear all temporary variables:

```
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

```

### Step 1: Import relevant libraries:

Import all the relevant Python libraries to conduct natural language processing tasks.

```
# Step 1: Import libraries:

# General libraries
import os
import warnings

# Data processing and visualization libraries
import pandas as pd
import numpy as np 
import string
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go

# Natural language processing libraries
import re
import nltk
import spacy
from collections import Counter
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
spacy.cli.download("en_core_web_sm")


# Keyword extraction libraries
import yake
from keybert import KeyBERT
from rake_nltk import Rake

# Text summarization libraries
from sentence_transformers import SentenceTransformer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import pipeline


# Google BigQuery libraries
from google.oauth2 import service_account
from pandas.io import gbq

# Sentiment analysis libraries
import textstat
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Vectorization libraries
from operator import itemgetter
from operator import attrgetter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix


```

### Step 2: Import Zero-Shot Classification Model:

Here am basically importing a pre-trained classier called [Hugging Face Zero-Shot Classification Model](https://huggingface.co/facebook/bart-large-mnli) that helps us classify the user reviews into different categories. We're using something called a "pipeline" to do this, which is nothing but a series of steps that the pre-trained classifier follows to achieve its goal.


```
classifier = pipeline("zero-shot-classification", device = 0)

```

### Step 3: Import the dataset:

I've taken a subset of the Yelp reviews dataset, focusing specifically on user reviews for Domino's Pizza franchises located in and around the United States.

This dataset contains unique identifiers (such as review_id, business_id, and user_id), review features (like review_length, stars, latitude, and longitude, etc.) as well as review types (such as useful, funny, and cool). Additionally, the dataset includes the actual user reviews, which can be found under the label 'text'.


```
dataset = pd.read_csv("dominos_pizza_review.csv")

```

### Step 4: Clean the dataset:

Think of this section as if we're cleaning our room before starting a new project. We want to make sure that everything is in order, with no duplicates or irrelevant items lying around.

I have also defined a function here that checks if a review is valid based on certain criteria such as having at least 10 words and no URLs. The valid reviews are kept, converted to lowercase, and all punctuation is removed. By performing these cleaning steps, the dataset is now ready to be used for further analysis, and we have eliminated any noise or irrelevant data that could skew our results.


```
# Remove any rows with missing or null values in the 'text' column
dataset = dataset.dropna(subset=['text'])

# Drop any rows where the cleaned text is empty
dataset = dataset[dataset['text'].str.len() > 0]

#Drop Duplicates:
dataset.drop_duplicates(subset='text', inplace=True)

def is_valid_review(text):
    # Remove all non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # If the review contains fewer than 10 words, it's not valid
    if len(text.split()) < 10:
        return False
    
    # If the review contains any URLs, it's not valid
    if re.search(r'http\S+', text):
        return False
    
    return True

dataset = dataset[dataset['text'].apply(is_valid_review)]


# convert to lowercase
dataset['text'] = dataset['text'].str.lower()

# remove punctuation
dataset['text'] = dataset['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

#Reset Index:
dataset.reset_index(drop=True, inplace=True)

```

### Step 5: Exploratory Data Analysis:

If you are into machine learning, you know by now that Exploratory Data Analysis (EDA) is your new best friend! I know I've been emphasizing this section in all my blogs (such as in [this one](https://srees.org/project/predict-cltv) and [this one](https://srees.org/project/predict-churn) and yes, even [out here](https://srees.org/project/predict-sales) :D), but trust me, it's that important! EDA is the first step you should take before diving into any modeling initiatives. Why, you ask? Well, EDA helps us identify patterns, inconsistencies, and outliers in our datasets. By understanding our data end-to-end, we can significantly improve the quality of our results.

Spending time with your data before modeling is like putting on sunscreen before heading to the beach - it'll save you from a lot of pain later!

In this section, we will start by figuring out the number of user reviews per month, followed by checking the popular cities where we find the most reviews. Then, we'll take a look at the review length, user rating distribution, and geospatial mappings. By conducting this EDA, we can gain a comprehensive overview of the data, which can help us make more informed decisions when it comes to designing our models.

By the way, if you're looking for a no-code solution to perform this EDA on the YELP dataset, check out this interactive [Google Looker Studio dashboard](https://lookerstudio.google.com/embed/u/0/reporting/597cd46f-146f-4906-89e0-0ed0851594e6/page/PziPD) that I've built. The dashboard provides insights into the same data we'll be exploring in this blog post. It's self-intuitive and doesn't require any coding knowledge, so whether you're a seasoned data scientist or just getting started, you can easily uncover insights about the YELP dataset with this tool. Of course, I'll still be exploring the data in this blog post, but I wanted to provide an alternative solution for those looking for a more visual approach. Alrighty, let's go!

### Step 5.1: Distribution of User Reviews by Month:

```
# Convert the date column to datetime format
dataset['date'] = pd.to_datetime(dataset['date'])

# Group the data by month and count the number of reviews in each month
reviews_by_month = dataset.groupby(pd.Grouper(key='date', freq='M')).size()

# Plot the timeseries of reviews by month
plt.style.use('default')
fig, ax = plt.subplots(figsize=(17, 8))
reviews_by_month.plot(ax=ax,  colormap='Accent')
ax.set_title('User Reviews by Month\n', horizontalalignment='center', fontstyle='normal', fontsize=22, fontfamily='sans-serif')
ax.set_xlabel('Date', horizontalalignment='center', fontstyle='normal', fontsize=18, fontfamily='sans-serif', labelpad=20)
ax.set_ylabel('Number of Reviews', horizontalalignment='center', fontstyle='normal', fontsize=18, fontfamily='sans-serif', labelpad=20)
ax.tick_params(labelsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#3f3f3f')
ax.spines['left'].set_color('#3f3f3f')
ax.tick_params(axis='x', colors='#3f3f3f')
ax.tick_params(axis='y', colors='#3f3f3f')
plt.style.use('classic')
plt.xticks(rotation=0, horizontalalignment='center', fontsize=12)
plt.yticks(rotation=0, horizontalalignment='right', fontsize=12)
ax.autoscale(enable=True, axis='both', tight=False)
plt.grid(axis='y', color='#d4d4d4', linestyle='--')
plt.show()


```

![](/images/projects/5.reviews_nlp/3.user_reviews_by_month.jpg)

The distribution of Domino's Pizza user reviews by month on the Yelp dataset unveils several intriguing insights:

1) First and foremost, the data covers a period from March 2006 to January 2022, spanning more than 15 years.

2) Secondly, the number of reviews increased steadily from 2006 to 2009. However, there was a significant increase in the number of user reviews from 2010 onwards.

3) A seasonal pattern exists in the number of user reviews, with the highest number of reviews being in the summer months (June to August) and the lowest number of reviews in the winter months (December to February).

4) There are several spikes in the number of reviews, with most of the spikes in December (in December 2012, December 2014, and December 2018).

5) Lastly, there appears to be a decline in the number of user reviews from 2018 onwards. The highest number of reviews occurred in December 2018, when the number of reviews reached 100. Since then, there has been a gradual decline in the number of reviews.


### Step 5.2: Distribution of User Reviews by Cities:

```
plt.style.use('default')
ax = dataset['city'].value_counts()[:10].plot.bar(
    title='Top Cities by User Reviews',  
    figsize=(17, 8),  
    colormap='Accent',  
    fontsize=15,  
    xlabel='Review Counts', 
    ylabel='City',  
    legend=True,  
    table=False,  
    grid=False,  
    subplots=False,  
    stacked=False,  
    linestyle='-',  
)

# Add the digit labels on top of the bars
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    
# set up the x and y labels
ax.set_xlabel('Review Counts', horizontalalignment='center', fontstyle='normal', fontsize='large', fontfamily='sans-serif', labelpad=20)
ax.set_ylabel('City', horizontalalignment='center', fontstyle='normal', fontsize='large', fontfamily='sans-serif', labelpad=20)

# set up the title, legends and theme
ax.set_title('Top Cities by User Reviews\n', horizontalalignment='center', fontstyle='normal', fontsize=22, fontfamily='sans-serif')
plt.legend(loc='upper right', fontsize="medium", frameon=False) 
plt.xticks(rotation=0, horizontalalignment='center', fontsize=12)
plt.yticks(rotation=0, horizontalalignment='right', fontsize=12)
plt.style.use('classic')
ax.autoscale(enable=True, axis='both', tight=False)
plt.grid(axis='y', color='#d4d4d4', linestyle='--')
plt.show()


```

![](/images/projects/5.reviews_nlp/4.top_cities_by_user_reviews.jpg)


1) From the plot below, we can clearly see that Philadelphia has the highest number of reviews (580), followed by New Orleans (386) and Indianapolis (374). These three cities seem to have the most active Domino's Pizza user communities on Yelp compared to the other cities listed in the data set.

2) Some of the cities with fewer user reviews include Avondale, Zephyrhills, Caseyville, and Boyertown. These cities may have smaller populations or less active Domino's Pizza user communities on Yelp.

3) The distribution of user reviews by the city is somewhat non-uniform. For example, there is a large gap between Philadelphia (580 reviews) and New Orleans (386 reviews), but the gap between New Orleans and Indianapolis (374 reviews) is smaller. This suggests that Yelp usage may be influenced by factors beyond population size, such as local culture, cuisine, or economic factors.

4) The top cities are all relatively large metropolitan areas. However, there are also smaller cities represented in the data set, such as Kennett Square and Mount Ephraim, which may have more concentrated Domino's Pizza user communities on Yelp.


### Step 5.3: Validate the Distribution of Review Length:

Of all the user reviews, 90% are less than 1000 characters, with the majority falling within the 200 to the 400-character range.

```
# Create a new column in the dataframe containing the length of each review
dataset['review_length'] = dataset['text'].apply(len)

# Set up the figure and axis
plt.style.use('default')
fig, ax = plt.subplots(figsize=(17, 8))
ax.set_title('Distribution of Review Length\n', horizontalalignment='center', fontstyle='normal', fontsize=22, fontfamily='sans-serif')
ax.set_xlabel('Review Length', horizontalalignment='center', fontstyle='normal', fontsize='large', fontfamily='sans-serif', labelpad=20)
ax.set_ylabel('Frequency', horizontalalignment='center', fontstyle='normal', fontsize='large', fontfamily='sans-serif', labelpad=20)

# Plot a histogram of review length distribution
ax.hist(dataset['review_length'], bins=50, color='#7fc97f', edgecolor='white', alpha=0.8)

# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set the background color to white
ax.set_facecolor('White')

# Add the digit labels on top of the bars
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, int(height), ha='center', va='bottom', fontfamily='sans-serif', fontsize=10)

# Set the x and y ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Set the style
plt.style.use('classic')
#ax.autoscale(enable=True, axis='both', tight=False)
plt.grid(axis='y', color='#d4d4d4', linestyle='--')
plt.show()

             
# Show the plot
plt.show()

```

![](/images/projects/5.reviews_nlp/5.distribution_of_review_length.jpg)


### Step 5.4: Number of Useful, Funny, and Cool Reviews by Month:

The number of reviews with useful and funny votes tends to be higher than those with cool votes. This suggests that users may be more likely to find reviews that are helpful or humorous than those that are simply cool.


```
# create a copy of the original dataset and set the date column as the index
df = dataset.copy()
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# group by month and sum the useful, funny, and cool columns
monthly_counts = df.groupby(pd.Grouper(freq='M')).sum()[['useful', 'funny', 'cool']]

# plot a multi-line graph of useful, funny, and cool reviews by month with a 30-day moving average
plt.style.use('default')
ax = monthly_counts.plot(figsize=(17, 8), colormap='Accent', fontsize=15, linestyle='-')
ax.set_title('Number of Useful, Funny, and Cool Reviews by Month\n', fontsize=22)
ax.set_xlabel('Date', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.legend(['Useful', 'Funny', 'Cool'], fontsize=15, loc='upper left')

# add a 30-day moving average line to the graph
rolling_mean = monthly_counts.rolling(window=30).mean()
rolling_mean.plot(ax=ax, color='k', linestyle='--', linewidth=2)

# set the style of the graph
plt.style.use('classic')
plt.xticks(rotation=0, horizontalalignment='center', fontsize=12)
plt.yticks(rotation=0, horizontalalignment='right', fontsize=12)

ax.autoscale(enable=False, axis='both', tight=False)
plt.show()

```

![](/images/projects/5.reviews_nlp/6.no_of_useful_funny_cool_reviews_by_month.jpg)


### Step 5.5: Correlation Matrix of Numeric Variables:

1) Review Count: The review count has a negative but weak correlation with the stars, indicating that Domino's Pizza franchises with more reviews tend to have slightly lower ratings in the US.

2) Useful: The "useful" feature has a strong positive correlation with "funny" and "cool", indicating that reviews that are perceived as useful also tend to be perceived as funny and cool.

```
plt.style.use('default')

# Select the numeric columns from the dataset
numeric_cols = ['review_count', 'stars', 'useful', 'funny', 'cool']

# Create a correlation matrix
sns.set(style="white")
corr_matrix = dataset[numeric_cols].corr()

# Generate a mask for the upper triangle:
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure and a diverging colormap:
f, ax = plt.subplots(figsize=(18, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio:
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Add the title
ax.set_title('Correlation Matrix of Numeric Variables\n', horizontalalignment='center', fontstyle='normal', fontsize=22, fontfamily='sans-serif')

```

![](/images/projects/5.reviews_nlp/7.correlation_matrix_of_numerical_variables.jpg)

Overall, these correlation matrices suggest that user reviews are more complex and multifaceted than we may presume. Often, user reviews are not solely based on star ratings; factors such as perceived usefulness, humor, and coolness can also influence how people perceive and engage with these reviews. In fact, this highlights the importance of using sentiment analysis and keyword extraction techniques, to gain a more nuanced understanding of user feedback.


### Step 5.6: Common words used in Reviews:

```
#Combine all text into a single string
all_text = ' '.join(dataset['text'].tolist())

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Accent', max_words=200).generate(all_text)

# Display the word cloud
import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words used in User Reviews\n', fontsize=22, fontstyle='normal', fontfamily='sans-serif')
plt.show()

```

![](/images/projects/5.reviews_nlp/8.common_words_used_in_reviews.jpg)

### Step 5.7: Validate the Distribution of Ratings:

```
plt.style.use('default')
fig, ax = plt.subplots(figsize=(17, 8))
ax.set_title('Distribution of Ratings\n', horizontalalignment='center', fontstyle='normal', fontsize=22, fontfamily='sans-serif')
ax.set_xlabel('Rating', horizontalalignment='center', fontstyle='normal', fontsize='large', fontfamily='sans-serif', labelpad=20)
ax.set_ylabel('Frequency', horizontalalignment='center', fontstyle='normal', fontsize='large', fontfamily='sans-serif', labelpad=20)

# Plot a histogram of rating distribution
ax.hist(dataset['stars'], bins=5, range=[0.5, 5.5], color='#7fc97f', edgecolor='white', alpha=0.8)

# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set the background color to white
ax.set_facecolor('White')

# Add the digit labels on top of the bars
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, int(height), ha='center', va='bottom', fontfamily='sans-serif', fontsize=10)

# Set the x and y ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Set the style
plt.style.use('classic')
#ax.autoscale(enable=True, axis='both', tight=False)
plt.grid(axis='y', color='#d4d4d4', linestyle='--')
plt.show()

```

![](/images/projects/5.reviews_nlp/9.distribution_of_ratings.jpg)

1) The vast majority of user reviews have a rating of 1 or 5 stars.

2) The number of 1-star reviews is significantly higher than the number of 2 or 3-star reviews. This suggests that users who have a negative experience are more likely to give a rating of 1 star than 2 or 3 stars.

3) There are fewer reviews with a 2, 3, or 4-star rating compared to the 1 or 5-star ratings. Extreme ratings seem to be common.

### Step 5.8: Visualize User Reviews on a Map:

Finally, I visualized the user reviews on a map to quickly identify popular areas and less popular ones. This can help businesses understand where they are receiving the most attention.

```
# Group the reviews by latitude and longitude, and count the number of reviews in each group
location_reviews = dataset.groupby(['latitude', 'longitude']).size().reset_index(name='count')

# Set up the map figure
fig = px.scatter_mapbox(location_reviews, lat='latitude', lon='longitude', size='count', hover_name='count', zoom=3, center={'lat': 37.0902, 'lon': -95.7129}, mapbox_style='carto-positron')

# Set the title and font
fig.update_layout(title='User Reviews by Location in the US', font=dict(family='sans-serif', size=22))

# Show the figure
fig.show()

```

![](/images/projects/5.reviews_nlp/10.user_reviews_on_a_map.jpg)


### Step 6: Auto Text Classification:

Topic modelling or text clustering, combined with keyword extraction and sentiment analysis, is a powerful approach that helps us understand what customers truly appreciate or criticize the most about product or service offerings. By automating the process of categorizing text, we can save time and effort while uncovering valuable insights from these unstructured data points.


Here what we do is take unstructured text data and use a [HuggingFace Zero-Shot Classification model](https://huggingface.co/facebook/bart-large-mnli) to classify it into predefined topics. The classification results, along with their precision rates, are stored in a temporary data frame before being merged back into the original dataset.

```
# extract text data from the dataset
sequence = list(dataset["text"])

# define the candidate labels/topics for classification
candidate_labels = ["Customer service", "Delivery experience", "Pizza Quality", "Value for money", "Online ordering experience", "Atmosphere", "Dietary restrictions", "Location"]

# apply the HuggingFace Zero Shot Classification method to classify the text data into candidate topics/labels
text_classification = classifier(sequence, candidate_labels)

# create a pandas dataframe to store the text classification results
text_classification = pd.DataFrame(text_classification)

# Select the first label:
text_classification["labels"] = text_classification["labels"].astype(str)

# Split the comma-separated labels into separate columns
interim = text_classification["labels"].str.split(',', expand = True)

select = [0]
interim = interim[select]

# Rename the columns and remove any quotes or brackets from the labels
interim = interim.rename(columns = {0: 'topic_clusters'})

interim["topic_clusters"] = interim["topic_clusters"].str.replace('[','')
interim["topic_clusters"] = interim["topic_clusters"].str.replace("'",'')

# Merge the raw dataset and the topic modelings
dataset = pd.merge(dataset, interim, left_index=True, right_index=True)


# Get the precision_rate:
text_classification["topic_modelling_precision"] = text_classification["scores"].astype(str)

# Split the comma-separated labels into separate columns
interim = text_classification["topic_modelling_precision"].str.split(',', expand = True)

select = [0]
interim = interim[select]

# Rename the columns and remove any quotes or brackets from the labels
interim = interim.rename(columns = {0: 'topic_modelling_precision'})

interim["topic_modelling_precision"] = interim["topic_modelling_precision"].str.replace('[','')

# Merge the raw dataset and the topic modelings
dataset = pd.merge(dataset, interim, left_index=True, right_index=True)

```

The treemap clearly shows that at least 7 out of every 10 users who wrote reviews about Domino's Pizza in the Yelp dataset focused on their delivery experience and customer service. The remaining segment discussed topics such as online ordering experience, value for money, and pizza quality. It's also worth noting that only a handful of users mentioned location and dietary restrictions. Let's move on to the next step and explore what users have said about each of these topics, as well as their sentiments.


![](/images/projects/5.reviews_nlp/11.text_Classification.JPG)

### Step 7: Sentiment Analysis:

Sentiment analysis is a powerful tool for understanding how customers feel about a business, product, or service. It allows us to quickly and easily understand the emotions and opinions expressed in text data, from glowing praise to scathing criticism.

```
# extract text data from the dataset
sequence = list(dataset["text"])

# Define the possible candidate labels for sentiment classification
candidate_labels = ["Positive", "Negative", "Neutral"]

# apply the HuggingFace Zero Shot Classification method to classify the text data into candidate topics/labels
text_classification = classifier(sequence, candidate_labels)

# create a pandas dataframe to store the text classification results
text_classification = pd.DataFrame(text_classification)

# Select the first label:
text_classification["labels"] = text_classification["labels"].astype(str)

# Split the comma-separated labels into separate columns
interim = text_classification["labels"].str.split(',', expand = True)

select = [0]
interim = interim[select]
# Rename the columns and remove any quotes or brackets from the labels
interim = interim.rename(columns = {0: 'sentiment_analysis'})

interim["sentiment_analysis"] = interim["sentiment_analysis"].str.replace('[','')
interim["sentiment_analysis"] = interim["sentiment_analysis"].str.replace("'",'')

# Merge the raw dataset and the topic modelings
dataset = pd.merge(dataset, interim, left_index=True, right_index=True)

# Get the sentiment analysis precision_rate:
text_classification["sa_precision"] = text_classification["scores"].astype(str)

# Split the comma-separated labels into separate columns
interim = text_classification["sa_precision"].str.split(',', expand = True)

select = [0]
interim = interim[select]

# Rename the columns and remove any quotes or brackets from the labels
interim = interim.rename(columns = {0: 'sa_precision'})

interim["sa_precision"] = interim["sa_precision"].str.replace('[','')

# Merge the raw dataset and the topic modelings
dataset = pd.merge(dataset, interim, left_index=True, right_index=True)

```

![](/images/projects/5.reviews_nlp/12.sentiment_analysis.JPG)

Here's what we could find about user sentiment from Domino's Pizza reviews in the Yelp dataset:

1) Positive reviews tend to be less common than negative reviews overall, with only a few topics (delivery experience, customer service, and online ordering experience) having a significant number of positive reviews.

2) Delivery experience and customer service are the most commonly discussed topics in the reviews, both of which received a significant number of negative reviews. This could indicate that these areas are particularly important for customers and may be areas of improvement for Domino's Pizza franchises.

3) Online ordering experience, value for money, and pizza quality are other important topics that appear in the reviews. While there are negative reviews for these areas, there are also a significant number of positive reviews, suggesting that businesses that excel in these areas may have a competitive advantage.

4) Some topics (such as pizza quality and value for money) appear to have a more nuanced sentiment, with both positive and negative reviews. This could suggest that these areas are particularly important to customers and may warrant further investigation or investment by businesses.

5) Finally, there are some areas (such as the atmosphere) that receive very few reviews overall, making it difficult to draw firm conclusions about customer sentiment in these areas. However, even a single positive or negative review could provide valuable insights for businesses looking to improve their offerings.


### Step 8: Keyword Extraction:

Now, let's try to extract the keywords in each one of those user reviews in the dataset. In this section, we are using pre-trained NLP tools to identify the top 5 most important words in a given piece of text. It does this by removing irrelevant words and counting the occurrences of specific parts of speech, like nouns and adjectives. The resulting top keywords are then stored in a new column of the original dataset.


```
# Load the pre-trained NLP classifier
nlp = spacy.load("en_core_web_sm")

# Load the NLTK tokenizer
tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Load a list of irrelevant words to remove
irrelevant_words = ["dominos" "pizza", "order", "delivery", "restaurant", "menu", "food", "service", "quality", "price", "place", "location"]

# Define a function to extract the top 5 most relevant keywords from a given text
def extract_keywords(text):
    # Use the NLTK tokenizer to split the text into sentences
    sentences = tokenizer.tokenize(text)
    
    # Create an empty Counter object to store the keyword counts
    keyword_counts = Counter()
    
    # Loop through each sentence in the text
    for sentence in sentences:
        # Use the spaCy model to parse the sentence
        doc = nlp(sentence)
        
        # Loop through each token in the sentence
        for token in doc:
            # Only consider certain parts of speech as keywords
            if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                # Ignore irrelevant words
                if token.text.lower() not in irrelevant_words:
                    # Increment the count for this keyword
                    keyword_counts[token.lemma_] += 1
    
    # Get the top 5 most frequent keywords
    top_keywords = keyword_counts.most_common(5)
    
    # Return a comma-separated string of the top keywords
    return ", ".join([keyword for keyword, count in top_keywords])


# Apply the extract_keywords function to the "text" column of the dataset
dataset["extracted_keywords"] = dataset["text"].apply(extract_keywords)

```

### Word Cloud of Keywords: 

If we analyze the word cloud over a period of time, it reveals compelling insights. For instance, in this case, I found that the number of negative keywords fluctuated over time. The trend appears to be relatively stable from 2012 to 2016 but then spikes from late 2016 to early 2019. There is a decrease in negative keywords from 2019 to 2020, which could be attributed to the COVID-19 pandemic era.

```
#Word cloud of Keywords
#Combine all text into a single string
all_text = ' '.join(dataset["extracted_keywords"].tolist())

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Accent', max_words=200).generate(all_text)

# Display the word cloud
import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words used in User Reviews\n', fontsize=22, fontstyle='normal', fontfamily='sans-serif')
plt.show()

```

![](/images/projects/5.reviews_nlp/13.wordcloud of keywords.jpg)


### Step 9: Summarization:

Lastly, Summarization - it is a powerful tool, especially in the world of eCommerce where showcasing product or SKU-level user reviews can make a big impact. In fact, it can be the key to effectively presenting relevant and useful information to potential customers. Summarized reviews can provide potential customers with quick and easily digestible information about a product or service, allowing them to gauge overall sentiment and make informed purchase decisions. By leveraging the insights gained from summarized reviews, businesses can enhance the customer experience and boost sales, making summarization a valuable asset in text analytics.

```
from transformers import pipeline
import warnings
import logging

# Set the logging level to only show errors and above
logging.basicConfig(level=logging.ERROR)

# Load the pre-trained summarization model
summarizer = pipeline("summarization")

# Define a function to summarize the text in each record
def summarize_text(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = summarizer(text, max_length=60, min_length=30, do_sample=False)[0]['summary_text']
    return summary

# Split the dataset into smaller subsets based on business_id, review_id, and user_id
groups = dataset.groupby(["business_id", "review_id", "user_id"])

# Define an empty list to store the summarized text
summaries = []

# Loop over each group and apply the summarize_text function
for name, group in groups:
    summary = summarize_text(group["text"].iloc[0])
    summaries.append(summary)

# Create a new DataFrame to store the summaries
summary_df = pd.DataFrame({
    "business_id": [name[0] for name in groups.groups.keys()],
    "review_id": [name[1] for name in groups.groups.keys()],
    "user_id": [name[2] for name in groups.groups.keys()],
    "summary": summaries
})

# Merge the summary_df with the original dataset on the "business_id", "review_id", and "user_id" columns
dataset = pd.merge(dataset, summary_df, on=["business_id", "review_id", "user_id"], how="left") 

```

### Step 10: Publish the output:

```
#Step 10: Publish the output:
dataset.to_csv('yelp_reviews_nlp.csv', index = False)

```

### What's Next?

1) Model Automation: Automate the whole framework and make sure it's doing batch predictions on the new reviews that are collected in your NPS survey management data servers every hour or day, depending on when they come in. This will provide a real-time understanding of customer sentiment and help your team stay on top of emerging issues.

2) Data Visualization & Training: Build a simple and easy-to-use visual analytics framework and train your team to use them effectively. By onboarding all relevant internal stakeholders, everyone will be able to reap near-real-time insights into how customers truly feel about your product or service offerings. To make this even more convenient, you can embed the dashboards into your internal web portals and schedule regular deliveries. Just to give you an idea, here's a snippet of the [Yelp Reviews NLP](https://lookerstudio.google.com/embed/u/0/reporting/597cd46f-146f-4906-89e0-0ed0851594e6/page/p_7pxapygy5c) dashboard that I have visualized and embedded in this article.

3) Analyze and Be the Bridge: Dive deep into the data cubes to analyze customer sentiment over time. By acting as a bridge between internal customer support and marketing/sales teams, you could help them identify negative sentiment trends and guide them through any issues that customers are facing. For example, if you were working with Domino's Pizza, you could highlight how negative sentiments are dominating over positive ones and guide the team through the specific issues that customers are facing in certain franchises or cities. Additionally, you could help them spot seasonality and cyclicity in the data, such as recurring patterns of negative reviews during certain months of the year, like December and January, etc.

![](/images/projects/5.reviews_nlp/2.sentiment_analysis.jpg)

4) Potential for Further Analysis: The data you provide will serve as a starting point for further analysis, including identifying businesses that receive the most negative reviews, understanding the reasons behind the negative sentiment, and assessing the impact of negative sentiment on the bottom line. Furthermore, you can combine this data with other sources, such as business locations, to analyze regional trends in sentiment.

5) Replicate the Similar Model for Other Services: You can replicate this model for a variety of other business use cases. For example, you could use similar methods for brand monitoring via social media feeds or for audience text mining via chat bot's conversational feeds. At any given time, this will enable you to gain a deep understanding of what your customers value the most or find fault with in your product or service offerings. This understanding can be foundational in shaping your business strategies and improving your overall customer experience.


### Conclusion

To wrap it up, I hope you found this blog post useful and enjoyed reading it! I've covered everything from collecting and preprocessing data to using pre-trained classifiers for sentiment analysis, keyword extraction, and topic modeling of unstructured user reviews using a Yelp dataset.

By following the steps outlined in this guide, you can quickly unlock the full potential of customer reviews and gain a competitive edge in your industry. So, start analyzing your customer reviews today and discover the insights that can take your business to the next level!

Thank you and write to you soon.

### GitHub Repository

 
I have learned (and continue to learn) from many folks in Github. Hence sharing my entire python script and supporting files in a public [GitHub Repository](https://github.com/srees1988/reviews-nlp-py) in case if it benefits any seekers online. Also, feel free to reach out to me if you need any help in understanding the fundamentals of supervised machine learning algorithms in Python. Happy to share what I know:) Hope this helps!
 




