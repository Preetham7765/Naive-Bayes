from __future__ import division
import numpy as np
import math

'''
Naive Bayes algorithm on Iris data set
reference :https://machinelearningmastery.com/naive-bayes-tutorial-for-machine-learning/
'''

def load_data():
    filepath = "Iris.txt"
    data_set = []
    iris_labels = {"Iris-setosa":0,"Iris-versicolor" :1,"Iris-virginica":2}

    with open(filepath) as myfile:
        next(myfile)
        for line in myfile.readlines():
            l = line.split(",")
            feature = l[1:5]
            label =  iris_labels[l[-1].strip()]
            data_set.append(feature + [label])
    #print(data_set)
    data = np.array(data_set,dtype = float)
    np.random.shuffle(data)
    train_data = data[:120]
    test_data = data[120:]
    X_train = train_data[:,[0,1,2,3]]
    y_train = train_data[:,-1]
    X_test = test_data[:,[0,1,2,3]]
    y_test = test_data[:,-1]
    return train_data,test_data


def mean(data):
    return sum(data)/len(data)


def standard_deviation(data):
    mean_data = mean(data)
    variance = sum([pow((num-mean_data),2) for num in data])/float(len(data))
    std_dev = math.sqrt(variance)
    return std_dev

def getfeaturebyclass(train_data):

    label_feature = {}

    for i in range(0,120):
        feature = train_data[i]
        if(feature[-1] not in label_feature):
            label_feature[feature[-1]] = []
        label_feature[feature[-1]].append(feature[0:4])
    return label_feature

def summarize(data):
    summaries = []
    for row in zip(*data):
        summaries.append([mean(row),standard_deviation(row)])
    return summaries


def train_classifier(train_data):
    label_feature_data = getfeaturebyclass(train_data)
    #print(train_data)
    feature_label_distribution = {}
    for label,data in label_feature_data.items():
        feature_label_distribution[label] = summarize(data)
    return feature_label_distribution


def calculate_liklyhood(feature_val,mean,std_dev):
    exponent = math.exp(-(math.pow(feature_val-mean,2)/(2*math.pow(std_dev,2))))
    return (1 / (math.sqrt(2*math.pi) * std_dev)) * exponent

def calculate_probabilities(feature_vec,label_feature_distribution):
    probabilities = {}
    for label,feature_dist_values in label_feature_distribution.items():
        probabilities[label] =1
        for i in range(len(feature_dist_values)):
            mean,std_dev = feature_dist_values[i]
            feature_val = feature_vec[i]
            probabilities[label] *=calculate_liklyhood(feature_val,mean,std_dev)
    return probabilities

def predict(feature_vec,label_feature_distribution):
    probabilities = calculate_probabilities(feature_vec,label_feature_distribution)
    predict_label = None
    predict_prob = -1
    for label,probability in probabilities.items():
        if(predict_prob < probability):
            predict_prob = probability
            predict_label = label
    return predict_label

def make_predictions(test_data,label_feature_distribution):
    predictions = []
    for feature_vec in test_data:
        actual_label = feature_vec[-1]
        del feature_vec[-1]
        label = predict(feature_vec,label_feature_distribution)
        predictions.append((label,actual_label))
    return predictions

def calculate_accuray(prediction):
    count =0
    for labels in prediction:
        if(labels[0] == labels[1]):
            count+= 1
    print("Accuracy = ",(count/len(prediction))*100)

def main():
    train_data,test_data = load_data()
    label_feature_distribution = train_classifier(train_data.tolist())
    result = make_predictions(test_data.tolist(),label_feature_distribution)
    calculate_accuray(result)

if __name__ == "__main__":
    main()
