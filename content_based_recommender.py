#%matplotlib inline
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

import warnings; warnings.simplefilter('ignore')

# from unidecode import unidecode

def read_metadata(path='data/movies_metadata.csv'):  # metadata = md
	return pd.read_csv(path)

def read_links_small(path='data/links.csv'):  # links_small = ls
	return pd.read_csv(path)

def read_credits(path='/Users/Subhalingam/Downloads/credits.csv'):  # credits = cr
	return pd.read_csv(path)

def read_keywords(path='data/keywords.csv'):  # keywords = kw
	return pd.read_csv(path)

def preprocess_md(md):
	md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [el['name'] for el in x] if isinstance(x, list) else [])
	md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
	# md = md.drop([19730, 29503, 35587])
	# md['id'] = md['id'].astype('int')
	# print(md['id'])
	return md

def preprocess_ls(ls):
	ls = ls[ls['tmdbId'].notnull()]['tmdbId'].astype('int').astype('str')
	# print(ls)
	return ls

def preprocess_cr(cr):
	cr['id'] = cr['id'].astype('str')
	# print(cr['id'])
	return cr

def preprocess_kw(kw):
	kw['id'] = kw['id'].astype('str')
	# print(kw['id'])
	return kw

def get_recommendations_movie_desc(title):
	md = preprocess_md(read_metadata())
	ls = preprocess_ls(read_links_small())
	smd = md[md['id'].isin(ls)]
	smd['tagline'] = smd['tagline'].fillna('')
	smd['description'] = smd['overview'] + smd['tagline']
	smd['description'] = smd['description'].fillna('')
	tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
	tfidf_matrix = tf.fit_transform(smd['description'])
	cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
	smd = smd.reset_index()
	titles = smd['title']
	indices = pd.Series(smd.index, index=smd['title'])

	idx = indices[title]
	sim_scores = list(enumerate(cosine_sim[idx]))
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	sim_scores = sim_scores[1:31]
	movie_indices = [i[0] for i in sim_scores]
	
	movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
	vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
	vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('float')
#	C = vote_averages.mean()
#	m = vote_counts.quantile(0.60)
	qual = movies[(movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
	qual['vote_count'] = qual['vote_count'].astype('int')
	qual['vote_average'] = qual['vote_average'].astype('float')
#	qual['wr'] = qual.apply(lambda x: weighted_rating(x,m,C), axis=1)
#	qual = qual.sort_values('wr', ascending=False).head(10)
	return qual

#	return titles.iloc[movie_indices]

def get_director(x):
	for i in x:
            		if i['job'] == 'Director':
            			return i['name']
	return np.nan

def filter_keywords(x,s):
	words = []
	for i in x:
        		if i in s:
            			words.append(i)
	return words

def weighted_rating(x,m,C):
    	v = x['vote_count']
    	R = x['vote_average']
    	return (v/(v+m) * R) + (m/(m+v) * C)

def get_recommendations_metadata(title):
	md = preprocess_md(read_metadata())
	ls = preprocess_ls(read_links_small())
	cr = preprocess_cr(read_credits())
	kw = preprocess_kw(read_keywords())
	md = md.merge(cr, on='id')
	md = md.merge(kw, on='id')
	smd = md[md['id'].isin(ls)]
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
	s = s[s > 1]

	stemmer = SnowballStemmer('english')
	smd['keywords'] = smd['keywords'].apply(lambda x: filter_keywords(x,s))
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

	idx = indices[title]
	sim_scores = list(enumerate(cosine_sim[idx]))
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	sim_scores = sim_scores[1:26]
	movie_indices = [i[0] for i in sim_scores]
	
	movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
	vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
	vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('float')
	C = vote_averages.mean()
	m = vote_counts.quantile(0.60)
	qual = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
	qual['vote_count'] = qual['vote_count'].astype('int')
	qual['vote_average'] = qual['vote_average'].astype('float')
	qual['wr'] = qual.apply(lambda x: weighted_rating(x,m,C), axis=1)
	qual = qual.sort_values('wr', ascending=False).head(10)
	return qual



print(get_recommendations_metadata('The Hangover Part II'))
