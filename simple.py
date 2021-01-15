import pandas as pd
import numpy as np

from helper import filter_data

def simple_recommender(data,filters={},percentile=0.95):
	"""
	IMDb Ranking function as stated in https://en.wikipedia.org/wiki/IMDb#Rankings
	"""
	data = filter_data(data,filters)

	m = data[data['vote_count'].notnull()]['vote_count'].astype('int').quantile(percentile)
	C = data[data['vote_average'].notnull()]['vote_average'].astype('float').mean()

	

	qual = data[(data['vote_count'] >= m) & (data['vote_count'].notnull()) & (data['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
	qual['vote_count'], qual['vote_average'] = qual['vote_count'].astype('int'), qual['vote_average'].astype('float')


	qual['rating'] = qual.apply(lambda x: (x['vote_average']*x['vote_count'] + m*C)/(x['vote_count']+m), axis=1)
	qual = qual.sort_values('rating', ascending=False)
	
	#return (R*v + C*m)/(v+m)
	return qual


if __name__ == "__main__": # See analysis.ipynb for more
	from helper import *

	md = preprocess_md(read_metadata())
	
	#top_movs = top_movies(md)
	#print(top_movs.head(10)) # top 10

	top_rom = simple_recommender(md,filters={"genres":"Romance"},percentile=0.75)
	print(top_rom.head(10))
