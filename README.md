# Social-Media-News-Generation
In this project "Unsupervised Learning" is used which is a Machine Learning concept, as the input dataset was not classified or labelled. Unlabelled data means data which is not tagged with any labels that will help in relating properties, features, etc. 

Important Points:
_____________________________________

1. KMeans Clustering:
•	It tries to divide ‘n’ inputs into ‘k’ clusters and attempts to put data in various clusters without being trained with labelled data.
•	We can make use of sklearn.cluster.KMeans library in python in order to implement it.
•	Sample code: KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
  where,
    n_clusters: Number of clusters and centroids to be formed.
    Init: Initialization method, we have used k-mean++ which smart way to converge.
    Max_iter: Maximum iterations in the algorithm in one go.
    N_init: Frequency of running algorithm with different centroid seeds.


2. Agglomerative Hierarchical Clustering which works by the iterative unions between the two nearest clusters reduce the number of clusters.

3. In order to work with K-Means and Agglomerative Clustering, we have used TfidfVectorizer which converts a collection of raw documents to a matrix of TF-IDF features.

______________________________________
Project Requirements:
1. Programming Language: python 3.0
2. Import below Libraries:
      import os
      import re
      import shutil
      import string
      import xlsxwriter
      import pandas as pd
      from pandas import DataFrame
      from nltk.corpus import stopwords
      from sklearn.cluster import KMeans
      from nltk.tokenize import word_tokenize
      from sklearn.cluster import AgglomerativeClustering
      from sklearn.feature_extraction.text import TfidfVectorizer

______________________________________
Run below from python command line 

	import nltk
	nltk.download('all')

