import numpy as np
import DecisionTree as dt
from scipy import optimize
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

    def pseudoResiduals(self, Y_set, pred_set) -> np.array:
        numerator = np.gradient(self._Loss(Y_set,  pred_set))
        denominator = np.gradient(pred_set)
        return -numerator/denominator ##dont need to transpose bc np.gradient same shape as input

    def objective_function(self, gamma, labels, inputs, f, h):
        gamma = float(gamma)  # Ensure gamma is a scalar
        loss = self._Loss(labels, f.predict(inputs) + gamma * h.predict(inputs))
        return loss

    def _GBTree(self, X_set, Y_set):
        gbTree = self._new_tree(X_set, Y_set)
        gammaVec = [1]
        learners = [gbTree]

        for i in range(1,self.n_base_learner):#define and implement better stopping criteria
            pred_set = gbTree.predict(X_set)
            residuals = self.pseudoResiduals(Y_set, pred_set)
            newTree = self._new_tree(X_set, residuals)
            learners.append(newTree)
            #newLearned = newTree.predict(X_set)
            gammaVec.append(optimize.minimize(self.objective_function, x0=np.asarray([0.01]), bounds=[(0, 1)],
                                              args=(Y_set, X_set, gbTree, newTree)).x[0])
            ##somehow combine gbtree with new learner and gamma ... maybe split this function up
            gbTree = ?


    ###predict(X) ->
    ###### construct predictor function as linear combination of gamma vector * learner vector elements
    ###### return predFunc(x)


    



