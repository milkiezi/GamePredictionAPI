from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import onnxruntime as rt
import json
import coding


app = Flask(__name__)

@app.route('/predict_ratings', methods=['POST'])
def predict():
    input_data = request.get_json()
    feainput = input_data['input']
    try: 
        estimated_sales = coding.CatboostPredict(feainput)
    except Exception as e: print(e)
    #print(estimated_sales)
    new_estimated_sales = (str(estimated_sales).lstrip('[').rstrip(']'))
    positive5,negative5 = coding.ShapCalculate(feainput,new_estimated_sales)

    return jsonify({'estimated sales': new_estimated_sales,
                    'positive1': positive5[0],
                    'positive2': positive5[1],
                    'positive3': positive5[2],
                    'positive4': positive5[3],
                    'positive5': positive5[4],
                    'negative1': negative5[0],
                    'negative2': negative5[1],
                    'negative3': negative5[2],
                    'negative4': negative5[3],
                    'negative5': negative5[4],
                    })


@app.route('/predict_noratings', methods=['POST'])
def predictno_ratings():
    input_data = request.get_json()
    feainput = input_data['input']
    try: 
        estimated_sales = coding.CatboostPredictNoRatings(feainput)
    except Exception as e: print(e)
    #print(estimated_sales)
    new_estimated_sales = (str(estimated_sales).lstrip('[').rstrip(']'))
    positive5,negative5 = coding.ShapCalculateNoratings(feainput,new_estimated_sales)

    return jsonify({'estimated sales': new_estimated_sales,
                    'positive1': positive5[0],
                    'positive2': positive5[1],
                    'positive3': positive5[2],
                    'positive4': positive5[3],
                    'positive5': positive5[4],
                    'negative1': negative5[0],
                    'negative2': negative5[1],
                    'negative3': negative5[2],
                    'negative4': negative5[3],
                    'negative5': negative5[4],
                    })



if __name__ == '__main__':
    app.run(debug=True)