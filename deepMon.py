import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from pandas.plotting import scatter_matrix
import warnings
import requests
from datetime import date, time, datetime
import workdays
import random
%matplotlib inline

def study():
    train = pd.read_csv("data/ghouls-goblins-and-ghosts-boo/train.csv")
    test = pd.read_csv("data/ghouls-goblins-and-ghosts-boo/test.csv")
    

    train["color"] = train["color"].map({"white" : 0, "black" : 1, "clear" : 2, "blue" : 3, "green" : 4, "blood" : 5})
    train["ghost"] = 0
    train["goblin"] = 0
    train["ghoul"] = 0
    train.ix[train["type"] == "Ghost", "ghost"] = 1
    train.ix[train["type"] == "Goblin", "goblin"] = 1
    train.ix[train["type"] == "Ghoul", "ghoul"] = 1
    #train["sum"] = train["bone_length"] + train["rotting_flesh"] + train["has_soul"] + train["ghoul"]

    test["color"] = test["color"].map({"white" : 0, "black" : 1, "clear" : 2, "blue" : 3, "green" : 4, "blood" : 5})

    display(train)

    predictors = train[["bone_length", "rotting_flesh", "hair_length", "has_soul", "color"]]

    classes = train[["ghost", "goblin", "ghoul"]]
    
    test_predictors = test[["bone_length", "rotting_flesh", "hair_length", "has_soul", "color"]]
    test["ghost"] = 0
    test["goblin"] = 0
    test["ghoul"] = 0
    test_classes = test[["ghost", "goblin", "ghoul"]]

    training_predictors = predictors
    training_classes = classes
    
    display(training_classes)

    num_predictors = len(training_predictors.columns)
    
    num_classes = len(training_classes.columns)

    session2 = tf.Session()

    feature = tf.placeholder(tf.float32, shape=(None, num_predictors))#4
    actual_classes = tf.placeholder(tf.float32,  shape=(None, num_classes))#3

    weights1 = tf.Variable(tf.truncated_normal([num_predictors, 50], stddev=0.0001))
    biases1 = tf.Variable(tf.ones([50]))

    weights2 = tf.Variable(tf.truncated_normal([50, 20], stddev=0.0001))
    biases2 = tf.Variable(tf.ones([20]))

    weights3 = tf.Variable(tf.truncated_normal([20, 3], stddev=0.0001))
    biases3 = tf.Variable(tf.ones([3]))

    hidden_layer_1 = tf.nn.relu(tf.matmul(feature, weights1) + biases1)
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights2) + biases2)
    model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)
    
    print(hidden_layer_2)
    print(weights3)
    print(tf.matmul(hidden_layer_2, weights3))
    print(actual_classes)

    cost = -tf.reduce_sum(actual_classes*tf.log(model))#ここでエラー

    train_op1 = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(cost)

    init = tf.global_variables_initializer()
    session2.run(init)
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    actual_classes_param = training_classes.values.reshape(len(training_classes.values), 3)

    for i in range(1, 15001):
        session2.run(
            train_op1,
            feed_dict={
              feature: training_predictors.values,
              actual_classes: actual_classes_param
            }
          )
        if i%5000 == 0:
            print( i, session2.run(
              accuracy,
              feed_dict={
                feature: training_predictors.values,
                actual_classes: actual_classes_param
              }
            ))
            
    print(test_classes)

    feed_dict= {
      feature: test_predictors.values,
      actual_classes: test_classes.values.reshape(len(test_classes.values), 3)
    }
    
    print(feed_dict)

    calc(model, session2, feed_dict, actual_classes)

def calc(model, session, feed_dict, actual_classes):
    predictions = tf.argmax(model, 1)
    print(feed_dict)
    print(session)
    probabilities = model.eval(feed_dict, session)
    print("probabilities:", probabilities)
    print("length:", len(probabilities))
    
    df = pd.DataFrame(columns=["type"])
    for p in probabilities:
        creature = None
        if p[0] > p[1] and p[0] > p[2]:
            creature = "Ghost"
        elif p[1] > p[0] and p[1] > p[2]:
            creature = "Goblin"
        elif p[2] > p[0] and p[2] > p[1]:
            creature = "Ghoul"
        
        df2 = pd.Series([creature], index=df.columns)
        df = df.append(df2, ignore_index=True)
    
    display(df)
    df.to_csv("my_tree_one.csv")

if __name__ == "__main__":
    study()