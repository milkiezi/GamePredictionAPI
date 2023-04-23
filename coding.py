import shap
import pickle
import pandas as pd
import csv
cb = pickle.load(open('resource/pc_game_with_rating_model.pkl', 'rb'))
cbnoratings = pickle.load(open('resource/pc_game_withOut_rating_model.pkl', 'rb'))
ratings_path = "resource/ratings_columns.csv"
noratings_path = "resource/no_ratings_columns.csv"

def CatboostPredict(input):
    predict = cb.predict(input)
    return(predict)

def ShapCalculate(input,classname):
    datacsv = pd.read_csv(ratings_path)
    explainer = shap.Explainer(cb)
    shap_values = explainer.shap_values([input])
    shap_values_row = shap_values[int(classname)][0]
    feature_names = datacsv.columns.tolist()
    features_with_shap = list(zip(feature_names, shap_values_row))
    sorted_features = sorted(features_with_shap, key=lambda x: abs(x[1]), reverse=True)
    positive = []
    negative = []
    for feature, shap_value in sorted_features:
        contribution = shap_value 
        if shap_value > 0:
            positive.append(f"{feature}: +{contribution:.2f}")
        else:
            negative.append(f"{feature}: -{abs(contribution):.2f}")
    positive5 = []
    for x in positive[0:5]:
        positive5.append(x)
    negative5 = []
    for x in negative[0:5]:
        negative5.append(x)

    return positive5,negative5


def CatboostPredictNoRatings(input):
    predict = cbnoratings.predict(input)
    return(predict)

def ShapCalculateNoratings(input,classname):
    datacsv = pd.read_csv(noratings_path)
    explainer = shap.Explainer(cbnoratings)
    shap_values = explainer.shap_values([input])
    shap_values_row = shap_values[int(classname)][0]
    feature_names = datacsv.columns.tolist()
    features_with_shap = list(zip(feature_names, shap_values_row))
    sorted_features = sorted(features_with_shap, key=lambda x: abs(x[1]), reverse=True)
    positive = []
    negative = []
    for feature, shap_value in sorted_features:
        contribution = shap_value 
        if shap_value > 0:
            positive.append(f"{feature}: +{contribution:.2f}")
        else:
            negative.append(f"{feature}: -{abs(contribution):.2f}")
    positive5 = []
    for x in positive[0:5]:
        positive5.append(x)
    negative5 = []
    for x in negative[0:5]:
        negative5.append(x)


    return positive5,negative5


def classdefine(classnum):
    classname = '0'
    if int(classnum) == 0:
        classname = '0-20000'
    elif int(classnum) == 1:
        classname = '20000-50000'
    elif int(classnum) == 2:
        classname = '50000-100000'
    elif int(classnum) == 3:
        classname = '100000-200000'
    elif int(classnum) == 4:
        classname = '200000-500000'
    elif int(classnum) == 5:
        classname = '500000-1000000'
    elif int(classnum) == 6:
        classname = '1000000-2000000'
    elif int(classnum) == 7:
        classname = '2000000-5000000'
    elif int(classnum) == 8:
        classname = '5000000-10000000'
    elif int(classnum) == 9:
        classname = '10000000-200000000'
    return classname


