import pandas as pd
import numpy as np
# from scipy import stats


def rank(data,filters={},percentile=0.95):
	"""
	IMDb Ranking function as stated in https://en.wikipedia.org/wiki/IMDb#Rankings
	"""
	for cat_class,cat_type in filters.items():
		if cat_class == 'genres':
			s = data.apply(lambda x: pd.Series(x[cat_class]),axis=1).stack().reset_index(level=1, drop=True)
			s.name = cat_class
			data = data.drop(cat_class, axis=1).join(s)
		data = data[data[cat_class] == cat_type]

	m = data[data['vote_count'].notnull()]['vote_count'].astype('int').quantile(percentile)
	C = data[data['vote_average'].notnull()]['vote_average'].astype('float').mean()

	

	qual = data[(data['vote_count'] >= m) & (data['vote_count'].notnull()) & (data['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
	qual['vote_count'], qual['vote_average'] = qual['vote_count'].astype('int'), qual['vote_average'].astype('float')


	qual['rating'] = qual.apply(lambda x: (x['vote_average']*x['vote_count'] + m*C)/(x['vote_count']+m), axis=1)
	qual = qual.sort_values('rating', ascending=False)
	
	#return (R*v + C*m)/(v+m)
	return qual

