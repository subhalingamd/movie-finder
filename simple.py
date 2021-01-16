import pandas as pd
import numpy as np

from helper import filter_data

def simple_recommender(data,filters={},percentile=0.925):
	"""
	IMDb Ranking function as stated in https://en.wikipedia.org/wiki/IMDb#Rankings
	"""
	data = filter_data(data,filters)

	m = data[data['vote_count'].notnull()]['vote_count'].astype('int').quantile(percentile)
	C = data[data['vote_average'].notnull()]['vote_average'].astype('float').mean()

	
	res = data[(data['vote_count'].notnull()) & (data['vote_average'].notnull()) & (data['vote_count'] >= m)][['id', 'title'] +list(filters.keys())+ ['vote_count', 'vote_average', 'popularity']]
	res['vote_count'], res['vote_average'] = res['vote_count'].astype('int'), res['vote_average'].astype('float')

	res['rating'] = res.apply(lambda x: (x['vote_average']*x['vote_count'] + m*C)/(x['vote_count']+m), axis=1)
	res = res.sort_values('rating', ascending=False)
	
	#return (R*v + C*m)/(v+m)
	return res


if __name__ == "__main__": # See analysis.ipynb for more
	from helper import *

	md = preprocess_md(read_metadata())
	
	#top_movs = top_movies(md)
	#print(top_movs.head(10)) # top 10

	top_rom = simple_recommender(md,filters={"genres":"Romance"},percentile=0.75)
	print(top_rom.head(10))
