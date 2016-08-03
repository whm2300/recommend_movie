#!/usr/local/env python
#-*- coding:utf-8 -*-

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

conf = SparkConf().setMaster("spark://spark01:7077").setAppName("train movie")
sc = SparkContext(conf = conf)

data_path = "/home/michael/spark_code/movie_data/ml-10M100K/ratings.dat"
data = sc.textFile(data_path)
ratings = data.map(lambda x: x.split("::")).map(lambda x : Rating(int(x[0]), int(x[1]), float(x[2])))

#build the recommendation model using ALS
rank = 50
iterations = 10
alpha = 0.01
model = ALS.train(ratings, rank, iterations, alpha)

#evaluate the model on training data
test_data = ratings.map(lambda x: (x[0], x[1]))
predictions = model.predictAll(test_data).map(lambda x: ((x[0], x[1], x[2])))
predictions.saveAsTextFile("/home/michael/spark_code/movie_data/ml-10M100K/predi.dat")
#rates_preds = ratings.map(lambda x: ((x[0], x[1]), x[2])).join(predictions)
#MSE = rates_preds.map(lambda x: (x[1][0] - x[1][1])**2).mear()
#print("Mean Squared Error = " + str(MSE))

#save and load model
#model.save(sc, "/home/michael/spark_code/movie_data/ml-10M100K/model.dat")
#same_model = MatrixFactorizationModel.load(sc, "/home/michael/spark_code/movie_data/ml-10M100K/model.dat")
