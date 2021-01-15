import pandas as pd
from ast import literal_eval
import numpy as np
# from unidecode import unidecode

def read_metadata(path='data/movies_metadata.csv'):  # metadata = md
	return pd.read_csv(path)

def read_links_small(path='data/links.csv'):  # links_small = ls
	return pd.read_csv(path)

def read_credits(path='data/credits.csv'):  # credits = cr
	return pd.read_csv(path)

def read_keywords(path='data/keywords.csv'):  # keywords = kw
	return pd.read_csv(path)


def preprocess_md(md):
	md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [el['name'] for el in x] if isinstance(x, list) else [])
	md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
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


def filter_data(data,filters):
	"""
	Filter the given data based on constraints
	"""
	for cat_class,cat_type in filters.items():
		if cat_class == 'genres':
			s = data.apply(lambda x: pd.Series(x[cat_class]),axis=1).stack().reset_index(level=1, drop=True)
			s.name = cat_class
			data = data.drop(cat_class, axis=1).join(s)
		data = data[data[cat_class] == cat_type]
	return data


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