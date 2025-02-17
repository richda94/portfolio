import DecisionTree as dt
import numpy as np
#import pandas as pd
import sklearn

class RandomForestClassifier():

    def __init__(self, n_base_learner=10, max_depth=5, min_samples_leaf=1, min_information_gain=0.0,
                 numb_of_features_splitting=None, bootstrap_sample_size=None) -> None:
        self.n_base_learner = n_base_learner
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.bootstrap_sample_size = bootstrap_sample_size

    def _create_bootstrap_samples(self, X, Y) -> tuple:
        '''
        returns a tuple of bootstrapped samples
        bootstrapping takes a number of randomly selected features and generates
        a 'forest' of decision trees,
        '''
        X_boot = []
        Y_boot = []

        for i in range(self.n_base_learner): #for each learner
            if not self.bootstrap_sample_size: #if we haven't selected a sample size, then use whole data set
                self.bootstrap_sample_size = X.shape[0] #all rows
            sampled_idx = np.random.choice(X.shape[0], size=self.bootstrap_sample_size, replace=True)
            X_boot.append(X[sampled_idx])
            Y_boot.append(Y[sampled_idx])
        return X_boot, Y_boot

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        X_boot, Y_boot = self._create_bootstrap_samples(X_train, Y_train)
        #make a list to append the trees
        self.base_learner_list = []

        for i in range(self.n_base_learner): #create and train n decision trees
            base_learner = dt.DecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                        min_information_gain=self.min_information_gain,
                                        numb_of_features_splitting=self.numb_of_features_splitting)
            base_learner.train(X_boot[i], Y_boot[i])
            self.base_learner_list.append(base_learner)

        ##need to revisit feature importances

    def _predict_proba_base(self, X_set) -> list:
        '''for all base learners, get the predicted probabilities'''
        pred_proba_list = []
        for learner in self.base_learner_list:
            pred_proba_list.append(learner.predict_proba(X_set))

        return pred_proba_list

    def _predict_proba(self, X_set):
        '''returns predicted probabilities for a given data set'''
        pred_probs = []
        base_pred_probs = self._predict_proba_base(X_set)
        for i in range(X_set.shape[0]):
            base_learner_probs = [a[i] for a in base_pred_probs]
            pred_probs.append(np.mean(base_learner_probs, axis=0))
        return pred_probs

    def predict(self, X_set):
        '''returns argmax of avg predictions ie the voted prediction'''
        pred_probs = self._predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)

        return preds


iris = sklearn.datasets.load_iris()
X = np.array(iris.data)
Y = np.array(iris.target)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y, test_size=.25, random_state=42)

rfc = RandomForestClassifier(n_base_learner=10, max_depth=5, min_samples_leaf=1, min_information_gain=0.0,
                 numb_of_features_splitting=None, bootstrap_sample_size=None)
rfc.train(X_train, Y_train)
testpreds = rfc.predict(X_test)

print("Test accuracy: ", sum(testpreds==Y_test)/len(Y_test))

