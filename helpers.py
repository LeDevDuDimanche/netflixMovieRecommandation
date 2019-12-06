import pandas as pd

def load_data(f):
  raw = pd.read_csv(f, converters = {'Id' : lambda x: x.split('_')} )
  user_movie = pd.DataFrame(raw.Id.tolist(), columns = ['user', 'movie'])
  data = pd.concat([user_movie, raw], axis=1)
  data.drop(columns = ["Id"], inplace = True)
  data.rename(columns = { 'Prediction' : 'rating' }, inplace = True)
  return data

def gen_submission(f_name, algo):
  submission = pd.read_csv("data/sample_submission.csv")
  submission['Prediction'] = [int(round(algo.predict(user, movie).est)) for [user, movie] in submission['Id'].str.split('_')]
  submission.to_csv(f_name, index=False)
