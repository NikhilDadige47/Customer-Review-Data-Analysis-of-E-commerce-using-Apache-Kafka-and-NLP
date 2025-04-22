import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

df = pd.read_csv('enriched_reviews.csv')

# 1. Sentiment Distribution
sentiment_counts = df['sentiment_label'].value_counts()
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Sentiment Distribution')
plt.show()

# 2. Rating Analysis
plt.figure(figsize=(8, 6))
df['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Rating Distribution')
plt.show()

# 3. Correlation: Sentiment Score vs. Rating
plt.figure(figsize=(8, 6))
sns.boxplot(x='rating', y='sentiment_score', data=df)
plt.title('Sentiment Score by Rating')
plt.show()

# 4. Top Cities with Positive Reviews
positive_reviews = df[df['sentiment_label'] == 'POSITIVE']
top_cities = positive_reviews['city'].value_counts().head(10)
top_cities.plot(kind='bar', title='Top Cities with Positive Reviews')

# 5. Word Cloud for Positive/Negative Reviews
def generate_wordcloud(texts, title):
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(texts))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis('off')
    plt.show()

positive_texts = df[df['sentiment_label'] == 'POSITIVE']['text'].dropna()
negative_texts = df[df['sentiment_label'] == 'NEGATIVE']['text'].dropna()

generate_wordcloud(positive_texts, 'Positive Reviews Word Cloud')
generate_wordcloud(negative_texts, 'Negative Reviews Word Cloud')

# 6. Time Trends (Convert 'date' to datetime first)
df['date'] = pd.to_datetime(df['date'], format='%b, %Y')
df.set_index('date')['sentiment_score'].resample('ME').mean().plot(title='Sentiment Over Time')

# 7. Certified Buyer Impact
certified_stats = df.groupby('certified_buyer')['sentiment_score'].mean()
certified_stats.plot(kind='bar', title='Sentiment by Certified Buyer Status')
