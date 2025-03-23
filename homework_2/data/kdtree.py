import numpy as np      
import pandas as pd
from sklearn.metrics import mean_squared_error


def distance(x1, x2):
    np.sqrt(mean_squared_error(x1.to_numpy(), x2.to_numpy()))


class KDTree:
    def __init__(self, X, leaf_size):
        self.data = pd.DataFrame(X)
        self.leaf_size = leaf_size
        self.devided_number = None
        self.devided_axis = None
        self.left = None
        self.right = None
        if self.data.empty:
            return
        self.build_node()


    def build_node(self):
        self.choose_axis()
        self.devided_number = self.data[self.devided_axis].median()
        small_data = self.data[self.data[self.devided_axis] < self.devided_number]
        big_data = self.data[self.data[self.devided_axis] >= self.devided_number]
        if small_data.shape[0] < self.leaf_size:
            self.left = small_data
        else:
            self.left = KDTree(small_data, self.leaf_size)
        
        if big_data.shape[0] < self.leaf_size:
            self.right = big_data
        else:
            self.right = KDTree(big_data, self.leaf_size)


    def choose_axis(self):
        min_dict_coordinate = {}
        max_dict_coordinate = {}
        for col_name, data in self.data.items():
            min_dict_coordinate[col_name] = self.data[col_name].min()
            max_dict_coordinate[col_name] = self.data[col_name].max()

        max_difference_at_axis = 0

        max_axis = max_dict_coordinate.keys()[0]

        for col_name in max_dict_coordinate.keys():
            now_difference_at_axis = max_dict_coordinate[col_name] - min_dict_coordinate[col_name]
            if max_difference_at_axis < now_difference_at_axis:
                max_difference_at_axis = now_difference_at_axis
                max_axis = col_name

        self.devided_axis = max_axis


    def query(self, X, k):
        data = pd.DataFrame(X)
        neighborhood_dict = {}
        for index, point in data.iterrows():
            point
        

    def get_nearest(self, point, neighborhoods, k):
        if len(neighborhoods) < k:
            neighborhoods.append(self.dev)

        if self.left is not None:
            self.left.get_nearest(point, neighborhoods, k)
        if self.right is not None:
            self.right.get_nearest(point, neighborhoods, k)
        