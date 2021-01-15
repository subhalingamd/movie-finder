import pandas as pd

from surprise import Reader, Dataset
# from surprise.model_selection import cross_validate, train_test_split
# from surprise import accuracy


from main import *
from run import *


def build_collabrative_model(userData,mode='svd'):
	mode_opts = ['knn','knn_baseline','knn_with_means','svd','svd++']
	assert mode in mode_opts, "Invalid mode. Choose from "+str(mode_opts)
	
	reader = Reader()
	userData = Dataset.load_from_df(userData[['userId', 'movieId', 'rating']].astype('str'), reader)
	
	# trainset, testset = train_test_split(userData, test_size=0.1)
	trainset = userData.build_full_trainset()

	model = None

	if mode == "knn":
		from surprise import KNNBasic

		model = KNNBasic(verbose=True)

	elif mode == "knn_baseline":
		from surprise import KNNBaseline

		model = KNNBaseline(verbose=True)

	elif mode=='knn_with_means':
		from surprise import KNNWithMeans

		# To use item-based cosine similarity use user_based = False
		sim_options = {
		    "name": "cosine",
		    "user_based": True,  # Compute  similarities between items
		}
		model = KNNWithMeans(verbose=True,sim_options=sim_options)
	

	elif mode == "svd":
		from surprise import SVD

		model = SVD(verbose=True)

	elif mode == "svd++":
		from surprise import SVDpp

		model = SVDpp(verbose=True)

	
	model.fit(trainset)

	# predictions = model.test(testset)
	# print(accuracy.rmse(predictions))
	# predictions = model.test(trainset)
	# print(accuracy.rmse(predictions))

	return model


def collabrative(moviesData,userData,userID,model=None,filters={},mode='svd'):

	if model is None:
		model = build_collabrative_model(userData,mode=mode)

	moviesData = filter_data(moviesData,filters)
	moviesData['id'] = moviesData['id'].astype('str')
	userData = userData[userData['userId']==userID]
	data = moviesData[~(moviesData['id'].isin(userData['movieId']))]
	data['estRating'] = data.apply(lambda x: model.predict(userID,x['id']).est, axis=1)
	


	#data = data.apply(lambda x: float(x['estRating'].split(',')[3]) if not None else -1)
	data = data.sort_values('estRating', ascending=False)

	return data[['id','original_title','estRating']]


if __name__ == "__main__": # See analysis.ipynb for more
	
	import yaml

	config = open("_config.yaml")
	config = yaml.load(config, Loader=yaml.FullLoader)



	data = pd.read_csv(config["ratings"])
	data2=pd.read_csv(config["metadata"])

	# build_collabrative_model(data,mode='svd1')
	print(build_collabrative_model(data))
	# print(collabrative(preprocess_md(data2),data,'120',filters={},mode='knn_baseline').head())
