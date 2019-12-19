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

# algos = [(algo, weight), i.e. (algo1, 0.5)]
def gen_submission_multi(f_name, algos):
  def process(user, movie):
    return sum([algo.predict(user, movie).est * weight for (algo, weight) in algos])

  submission = pd.read_csv("data/sample_submission.csv")
  submission['Prediction'] = [int(round(process(user, movie))) for [user, movie] in submission['Id'].str.split('_')]
  submission.to_csv(f_name, index=False)

def gen_submission_multi_with_train(f_name, algos):
  def process(user, movie):
    return sum([algo.predict(user, movie).est * weight for (algo, weight) in algos])

  submission = pd.read_csv("data/data_train.csv")
  submission['Prediction2'] = [int(round(process(user, movie))) for [user, movie] in submission['Id'].str.split('_')]
  submission.to_csv(f_name, index=False)  
