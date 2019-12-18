from ALS_helpers import *
import scipy

from surprise import AlgoBase, PredictionImpossible 
 
def init_MF(trainset, num_features, min_num_ratings, debug=False):
    """init the parameter for matrix factorization.""" 


    #builds a sparse lil matrix out of the surprise Trainset class.
    train = preprocess_trainset_data(trainset)
     
    if debug:
        allSame = True
        i=0
        for row, col, value in trainset.all_ratings():
            if train[row, col] != value:
                print("at {0}, {1} train = {2} and trainset = {3},     iter {4}".format(row, col, train[row,col], value, i))
                allSame = False
            i+=1

        if not allSame:
            raise Error("not all same")

    
    num_item, num_user = train.get_shape()

    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)

    # start by item features.
    item_nnz = train.getnnz(axis=1)
    item_sum = train.sum(axis=1)

    for ind in range(num_item):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    return user_features, item_features, train


class ALS(AlgoBase):
    def __init__(self, n_epochs=20, num_features=20, lambda_user=0.1, lambda_item=0.7, lambda_all=None, min_num_ratings=1, verbose=False, debug=False):
        self.num_features = num_features
        self.lambda_user = lambda_user if lambda_all == None else lambda_all
        self.lambda_item = lambda_item if lambda_all == None else lambda_all
        self.min_num_ratings = min_num_ratings
        self.n_epochs = n_epochs 
        self.debug = debug

        AlgoBase.__init__(self)

    def fit(self, trainset): 
        AlgoBase.fit(self, trainset)
        self.als(trainset)

        return self

    def als(self, train):
        """Alternating Least Squares (ALS) algorithm."""
        #train was a sp.lil_matrix
        # define parameters
        change = 1
        error_list = [0, 0]

        # set seed
        np.random.seed(988)

        # init ALS
        self.user_features, self.item_features, sp_lil_train = init_MF(train, self.num_features, self.min_num_ratings, debug=self.debug)

        # get the number of non-zero ratings for each user and item
        nnz_items_per_user, nnz_users_per_item = sp_lil_train.getnnz(axis=0), sp_lil_train.getnnz(axis=1)

        # group the indices by row or column index
        nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(sp_lil_train)

        # run ALS
        print("\nstart the ALS algorithm...")
        #use epoch instead of stop criterion like in the lab solution
        for current_epoch in range(self.n_epochs):
            # update user feature & item feature
            self.user_features = update_user_feature(
                sp_lil_train, self.item_features, self.lambda_user,
                nnz_items_per_user, nz_user_itemindices)
            self.item_features = update_item_feature(
                sp_lil_train, self.user_features, self.lambda_item,
                nnz_users_per_item, nz_item_userindices)

            train_error = compute_error(sp_lil_train, self.user_features, self.item_features, nz_train)
            print("RMSE on training set: {}.".format(train_error))
            error_list.append(train_error)
            change = np.fabs(error_list[-1] - error_list[-2])
        

    def estimate(self, user_index, item_index): 
        if not self.trainset.knows_user(user_index) or not self.trainset.knows_item(item_index):
            raise PredictionImpossible("User and item are unknown")

        user_info = self.user_features[:, user_index]
        item_info = self.item_features[:, item_index]
        return user_info.T.dot(item_info)


