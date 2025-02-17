import numpy as np
import DecisionTree as dt

'''
inputs:
    {(x_i, y_i)} i: 1 -> n
    differentiable loss function L(y, F(x))
    number of iterations M
Goal:
    define x -> F(x) as sum i->M h_i(x) of weak learners (individual trees) note x is n-vector of features
algorithm:
    1) let F_1(x) =  mean of y (guess it can be any constant? or just a base tree i guess)
    for m = 1 ... M
        2) find r^_1_m = -[diff(L(y_i,y^_i))/diff(y^_i)] for i = 1...n and y^_i= F_1(x_i)
        
        3) train new weak learner h_m(x) on set {(x_i, r^_i_m)} i -> n
        4) solve gamma_m = argmin_gamma of sum i:1->n of L(yi, F_1(x_i) + gamma * h_m(x_i))
        5) update F_m(x) = F_m-1(x) + gamma_m * h_m(x)
'''

class GradientBoost():
    def __init__(self, n_base_learner=10, max_depth=5, min_samples_leaf=1, min_information_gain=0.0,
                 numb_of_features_splitting=None, bootstrap_sample_size=None) -> None:
        self.n_base_learner = n_base_learner
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.bootstrap_sample_size = bootstrap_sample_size

    def _new_tree(self, X_set, Y_set):
        tree = dt.DecisionTree(max_depth=2)
        tree.train(X_set, Y_set)
        return tree

    def _Loss(self, labels, predictions):
        n = len(labels)
        squareErrors = [] #[(y-fx)^2 for y,fx in labels, predictions]????
        for i in range(0,n - 1):
            squareErrors.append((labels[i] - predictions[i])**2)
        return squareErrors #1/n*sum(

    def pseudoResiduals(self, X_set, Y_set, tree: dt.DecisionTree) -> tuple:
        numerator = np.gradient(self._Loss(Y_set,  tree.predict(X_set)))
        denominator = np.gradient(tree.predict(X_set))
        return -numerator/denominator ##dont need to transpose bc np.gradient same shape as input

    def _GBTree(self, X_set, Y_set):
        gbTree = self._new_tree(X_set, Y_set)
        ###gamma vector of len(n_base_learner) -> gamVec[0] = 1
        ###learner vector of len(n_base_learner) or maybe list is fine -> append base tree
        for i in range(self.n_base_learner):#define and implement better stopping criteria
            residuals = self.pseudoResiduals(X_set, Y_set, gbTree)
            newTree = self._new_tree(X_set, residuals)
            ###learner list.append new tree
            # find minimum gamma for loss(predictions, basetree.predict(base x set) + gamma * newtree1(base x set)
                #gamma vector[i] = gamma
            ##somehow combine gbtree with new learner and gamma ... maybe split this function up 


    ###predict(X, _GBTree?) ->
    ###### construct predictor function as linear combination of gamma vector * learner vector elements
    ###### return predFunc(x)


    
