from surprise.model_selection import cross_validate
from surprise import SVD, Dataset, Reader
from helpers import *
import sys

def preplace(s):
  sys.stdout.write('\r' + s)
  sys.stdout.flush()

preplace("Loading data...")
data = load_data("data/data_train.csv")

preplace("Building a dataset...")
dataset = Dataset.load_from_df(data[['user', 'movie', 'rating']], Reader(rating_scale=(1, 5)))
svd = SVD(n_epochs=250, n_factors=300, lr_all=0.0022, reg_all=0.089)

preplace("Cross validating SVD...")
cross_validate(svd, dataset, measures=['RMSE','MAE'], n_jobs=-1, cv=5, verbose=True)

preplace("Fitting SVD...")
svd.fit(dataset.build_full_trainset())

preplace("Generating submission.csv...")
gen_submission(r'data/submission.csv', svd)