import numpy as np
from collections import Counter


class TreeNode():
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.left = None
        self.right = None


class DecisionTree():
    """
    Decision Tree Classifier
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    """

    def __init__(self, max_depth=4, min_samples_leaf=1,
                 min_information_gain=0.0, numb_of_features_splitting=None,
                 amount_of_say=None) -> None:
        """
        Setting the class with hyperparameters
        max_depth: (int) -> max depth of the tree
        min_samples_leaf: (int) -> min # of samples required to be in a leaf to make the splitting possible
        min_information_gain: (float) -> min information gain required to make the splitting possible
        num_of_features_splitting: (str) ->  when splitting if sqrt then sqrt(# of features) features considered,
                                                            if log then log(# of features) features considered
                                                            else all features are considered
        amount_of_say: (float) -> used for Adaboost algorithm
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.amount_of_say = amount_of_say

    def _entropy(self,
                 class_probabilities: list) -> float:  ##metric used for splitting. future work -> replace entropy with gini impurity for larger datasets
        return sum([-p * np.log2(p) for p in class_probabilities if
                    p > 0])  ## idea is that if there is a dominant class there will be low entropy
        #return 1-sum(np.square(class_probabilities))

    def _class_probabilities(self, labels: list) -> list:  ##basic probabilities of each outcome
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def _data_entropy(self, labels: list) -> float:  ##use class probabilities to calculate entropy
        return self._entropy(self._class_probabilities(labels))

    def _partition_entropy(self,
                           subsets: list) -> float:  ##find entropy for subsets. This is called below to compare splits
        """subsets = list of label lists (EX: [[1,0,0], [1,1,1])"""
        total_count = sum([len(subset) for subset in subsets])
        return sum([self._data_entropy(subset) * (len(subset) / total_count) for subset in subsets])  #sum of wgt avg

    def _split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]

        return group1, group2

    def _select_features_to_use(self, data: np.array) -> list:
        feature_idx = list(range(data.shape[1] - 1))
        ### will be used to extend the model with sqrt of num of features or log2 of number of features-- to start just use all features
        feature_idx_to_use = feature_idx
        return feature_idx_to_use

    def _find_best_split(self, data: np.array) -> tuple:
        min_part_entropy = 1e9
        feature_idx_to_use = self._select_features_to_use(data)
        '''
        for each feature index, calculate median/percentile/whatever
        for each feature value, partition the data above and below that value -> this will be the split value
        Once partitioned, calculate entropy of partition -> remember low entropy means there is a well defined split in the data
        if the entropy is lower than the minimum so far, this partition becomes the new minimum -> this will be the node
        '''
        for idx in feature_idx_to_use:
            #when extended to have multiple values like percentiles, will need to nest another for loop over those values
            feature_value = np.median(data[:, idx])
            g1, g2 = self._split(data, idx, feature_value)
            partition_entropy = self._partition_entropy([g1[:, -1], g2[:, -1]])
            if partition_entropy < min_part_entropy:
                g1_min, g2_min = g1, g2
                min_entropy_feature_idx = idx
                min_entropy_feature_val = feature_value
                min_part_entropy = partition_entropy
        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def _find_label_probs(self,
                          data: np.array) -> np.array:  ####this gets called below and is used to compare entropy values with the partition in order to find the information gain
        labels_as_integers = data[:, -1].astype(int)
        # Calculate the total number of labels
        total_labels = len(labels_as_integers)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        # Populate the label_probabilities array based on the specific labels
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def _create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        # Check if the max depth has been reached (stopping criteria)
        if current_depth >= self.max_depth:
            return None

        # Find best split
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self._find_best_split(data)

        # Find label probs for the node
        label_probabilities = self._find_label_probs(data)

        # Calculate information gain
        node_entropy = self._entropy(label_probabilities)
        information_gain = node_entropy - split_entropy  ###initial node entropy is total data entropy -> then recursively get smaller and smaller through calls below

        # Create node
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)

        # Check if the min_samples_leaf has been satisfied (stopping criteria)
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        # Check if the min_information_gain has been satisfied (stopping criteria)
        elif information_gain < self.min_information_gain:
            return node

        current_depth += 1
        node.left = self._create_tree(split_1_data, current_depth)
        node.right = self._create_tree(split_2_data, current_depth)

        return node

    def _predict_one_sample(self, X: np.array) -> np.array:
        """Returns prediction for 1 dim array"""
        node = self.tree

        # Finds the leaf which X belongs
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """
        Training is just creating the decision tree, since the create tree function finds the best split
        """

        # Concat features and labels
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

        # Start creating the tree
        self.tree = self._create_tree(data=train_data, current_depth=0)

        # # Calculate feature importance
        # self.feature_importances = dict.fromkeys(range(X_train.shape[1]), 0)
        # self._calculate_feature_importance(self.tree)
        # # Normalize the feature importance values
        # self.feature_importances = {k: v / total for total in (sum(self.feature_importances.values()),) for k, v in self.feature_importances.items()}

    def predict_proba(self, X_set: np.array) -> np.array:
        """Returns the predicted probs for a given data set"""

        pred_probs = np.apply_along_axis(self._predict_one_sample, 1, X_set)

        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """Returns the predicted probs for a given data set"""

        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)

        return preds
