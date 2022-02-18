# All Imported Libraries

import pandas as pd
from tabulate import tabulate
import numpy as np
import sys
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
import warnings
from datetime import date

# All the imported Libraries
warnings.simplefilter('ignore')

#Function For Debug
def Debugger(input):
    with open("debug"+str(date.today())+".txt", "w+") as output:
        sys.stdout = output  # Change the standard output to the file we created.
        print(str(input))
        sys.stdout = original_stdout  # Reset the standard output to its original value
#Function End

# Function for particular Genre
def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][
        ['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified['weight_ratio'] = qualified.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + (m / (m + x['vote_count']) * C),
        axis=1)
    qualified = qualified.sort_values('weight_ratio', ascending=False).head(250)

    return qualified

# Function Ends

# The weighted rating Function
def weighted_rating(x):
    # Calculates the rating base on 'vote count' and 'vote average'.
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
# Function Ends

# Function Get Recommendation
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]
# Function Ends

# Function to get Director
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
# Function Ends

# Function to Filter
def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words
# Function Ends

# Function For a better Recommendation
def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[
        (movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified
# Function End

# Read the .csv file to a variable
full_data = pd.read_csv('movies_metadata.csv')

# View all the column in the output
pd.set_option('display.max_columns', None)

# Data Cleaning step
full_data['genres'] = full_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# Variable declared for vote count and average removing the null values
vote_counts = full_data[full_data['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = full_data[full_data['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
print(C)

m = vote_counts.quantile(0.95)
print("This is the 95th Percentile Limit of vote Count" + str(m))

full_data['year'] = pd.to_datetime(full_data['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

# All the columns that we'll be showing in the output.
qualified = full_data[(full_data['vote_count'] >= m) & (full_data['vote_count'].notnull()) & (full_data['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
print('This is the Dimension of the Dataframe : ' + str(qualified.shape))

qualified['weight_ratio'] = qualified.apply(weighted_rating, axis=1)

qualified = qualified.sort_values('weight_ratio', ascending=False).head(250)

print(tabulate(qualified.head(10), headers='keys', tablefmt='psql'))

s = full_data.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = full_data.drop('genres', axis=1).join(s)

print(tabulate(build_chart('Thriller').head(10),headers='keys',tablefmt='psql'))

original_stdout = sys.stdout # Save the reference to the original standard output

links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
full_data = full_data.drop([19730, 29503, 35587])

#Check EDA Notebook for how and why I got these indices.
full_data['id'] = full_data['id'].astype('int')

smd = full_data[full_data['id'].isin(links_small)]
print(smd.shape)

smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])

print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

print(cosine_sim[0])

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

print(tabulate(get_recommendations('The Godfather').head(10), headers='keys', tablefmt='psql'))

print(tabulate(get_recommendations('The Dark Knight').head(10), headers='keys', tablefmt='psql'))

credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
full_data['id'] = full_data['id'].astype('int')

print(full_data.shape)

full_data = full_data.merge(credits, on='id')
full_data = full_data.merge(keywords, on='id')

smd = full_data[full_data['id'].isin(links_small)]
print(smd.shape)

smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))

smd['director'] = smd['crew'].apply(get_director)

smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)

smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x,x, x])

s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'

s = s.value_counts()
print(s[:5])

s = s[s > 1]

stemmer = SnowballStemmer('english')
print(stemmer.stem('dogs'))

smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

print(tabulate(get_recommendations('The Dark Knight').head(10), headers='keys', tablefmt='psql'))
