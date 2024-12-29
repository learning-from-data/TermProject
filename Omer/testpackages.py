import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from scipy.sparse import csr_matrix  # Use sparse matrix for efficiency
import os