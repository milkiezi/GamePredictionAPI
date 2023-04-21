import shap
import pickle
import pandas as pd
import csv
cb = pickle.load(open('F:\\python3.7.6\\api\\resource\\pc_game_with_rating_model.pkl', 'rb'))
cbnoratings = pickle.load(open('F:\\python3.7.6\\api\\resource\\pc_game_withOut_rating_model.pkl', 'rb'))
ratings_path = "F:\\python3.7.6\\api\\resource\\ratings_columns.csv"
noratings_path = "F:\\python3.7.6\\api\\resource\\no_ratings_columns.csv"

def CatboostPredict(input):
    predict = cb.predict(input)
    return(predict)

def ShapCalculate(input,classname):
    datacsv = pd.read_csv(ratings_path)
    datacsv = datacsv.iloc[:0, :]
    datacsv.to_csv(ratings_path, index=False)
    with open(ratings_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(input)
    datacsv = pd.read_csv(ratings_path)
    explainer = shap.Explainer(cb)
    shap_values = explainer.shap_values(datacsv)
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
    #five_positive = [x for x in positive5]
    #print("\n".join(five_positive))
    #five_negative = [x for x in negative5]
    #print("\n".join(five_negative))

    return positive5,negative5


def CatboostPredictNoRatings(input):
    predict = cbnoratings.predict(input)
    return(predict)

def ShapCalculateNoratings(input,classname):
    datacsv = pd.read_csv(noratings_path)
    datacsv = datacsv.iloc[:0, :]
    datacsv.to_csv(noratings_path, index=False)
    with open(noratings_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(input)
    datacsv = pd.read_csv(noratings_path)
    explainer = shap.Explainer(cbnoratings)
    shap_values = explainer.shap_values(datacsv)
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
    #five_positive = [x for x in positive5]
    #print("\n".join(five_positive))
    #five_negative = [x for x in negative5]
    #print("\n".join(five_negative))

    return positive5,negative5


