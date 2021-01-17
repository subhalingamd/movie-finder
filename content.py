import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer

from helper import *

#from unidecode import unidecode


def content_desc(title,md,ls,filters={}):
	smd1 = md[md['id'].isin(ls)]
	if not len(smd1[smd1['title']==title]):
		print("Error: I haven't seen this movie title in my collection before. Can't proceed.")
		return

	smd = filter_data(smd1,filters)
	if not len(smd[smd['title']==title]):
		print("Warning: Given title not in required filters! I'm proceeding with the query but the results might not be correct.")
		smd = pd.concat([smd,smd1[smd1['title']==title]])
	
	smd['desc'] = (smd['overview'].fillna('') + smd['tagline'].fillna('')).fillna('')
	tfidf = TfidfVectorizer(analyzer='word', stop_words="english", ngram_range=(1, 2))
	matrix = tfidf.fit_transform(smd['desc'])
	cosine = linear_kernel(matrix, matrix)

	smd = smd.reset_index()
	indices = pd.Series(smd.index, index=smd['title'])
	# print(smd.reindex(['title'],axis="columns")['title'])
	idx = indices[title]

	scores = list(enumerate(cosine[idx]))
	scores = sorted(scores, key=lambda x: x[1], reverse=True)
	scores = scores[1:501]
	#print(scores)
	movies = [i[0] for i in scores]
	
	return smd.iloc[movies]


def content_metadata(title,md,ls,cr,kw,filters={}):
	md = md.merge(cr, on='id').merge(kw, on='id')
	smd1 = md[md['id'].isin(ls)]

	if not len(smd1[smd1['title']==title]):
		print("Error: I haven't seen this movie title in my collection before. Can't proceed.")
		return

	smd = filter_data(smd1,filters)
	if not len(smd[smd['title']==title]):
		print("Warning: Given title not in required filters! I'm proceeding with the query but the results might not be correct.")
		smd = pd.concat([smd,smd1[smd1['title']==title]])

	smd['cast'] = smd['cast'].apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []).apply(lambda x: x[:3] if len(x) >=3 else x).apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
	smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
	smd['crew'] = smd['crew'].apply(literal_eval)
	smd['crew_size'] = smd['crew'].apply(lambda x: len(x))
	smd['director'] = smd['crew'].apply(get_director).astype('str').apply(lambda x: str.lower(x.replace(" ", ""))).apply(lambda x: [x,x,x])
	smd['keywords'] = smd['keywords'].apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
	
	s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
	s.name = 'keyword'
	s = s.value_counts()
	s = s[s > 1]

	stemmer = SnowballStemmer('english')
	smd['keywords'] = smd['keywords'].apply(lambda x: filter_keywords(x,s)).apply(lambda x: [stemmer.stem(i) for i in x]).apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
	if "genres" in filters.keys():
		smd['soup'] = (smd['keywords'] + smd['cast'] + smd['director']).apply(lambda x: ' '.join(x))
	else:
		smd['soup'] = (smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']).apply(lambda x: ' '.join(x))

	count = CountVectorizer(analyzer='word', stop_words="english", ngram_range=(1, 2))
	matrix = count.fit_transform(smd['soup'])
	cosine = cosine_similarity(matrix, matrix)
	smd = smd.reset_index()
	indices = pd.Series(smd.index, index=smd['title'])

	idx = indices[title]
	scores = list(enumerate(cosine[idx]))
	scores = sorted(scores, key=lambda x: x[1], reverse=True)
	scores = scores[1:501]
	#print(scores)
	movies = [i[0] for i in scores]
	
	movies = smd.iloc[movies]
	vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int').quantile(0.5)
	vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('float').mean()
	movies = movies[(movies['vote_count'].notnull()) & (movies['vote_average'].notnull()) & (movies['vote_count'] >= vote_counts)]
	movies['vote_count'], movies['vote_average'] = movies['vote_count'].astype('int'), movies['vote_average'].astype('float')
	movies['rating'] = movies.apply(lambda x: weighted_rating(x,vote_counts,vote_averages), axis=1)
	movies = movies.sort_values('rating', ascending=False)
	
	#print(movies.columns)
	return movies


if __name__ == "__main__": # See analysis.ipynb for more
	from helper import *

	md = preprocess_md(read_metadata())
	ls = preprocess_ls(read_links_small())
	cr = preprocess_cr(read_credits(path="/Users/Subhalingam/Downloads/credits.csv"))
	kw = preprocess_kw(read_keywords())


	print(content_desc('The Hangover Part III',md,ls))
	print(content_metadata('The Hangover Part III',md,ls,cr,kw))
