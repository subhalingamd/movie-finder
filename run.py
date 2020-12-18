import pandas as pd
from ast import literal_eval
import numpy as np
from main import *

def read_metadata(path='data/movies_metadata.csv'):  # metadata = md
	return pd.read_csv(path)

def preprocess_md(md):
	md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [el['name'] for el in x] if isinstance(x, list) else [])
	md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
	return md


if __name__ == "__main__": # See analysis.ipynb for more
	md = preprocess_md(read_metadata())
	
	#top_movs = top_movies(md)
	#print(top_movs.head(10)) # top 10

	top_rom = top_rom(md)
	print(top_rom.head(10))
