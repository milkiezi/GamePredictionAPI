from flask import Flask, request, jsonify
import coding


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB


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
    text_estimated_sales = coding.classdefine(new_estimated_sales)
    return jsonify({'estimated_sales_class':new_estimated_sales,
                    'estimated_sales': text_estimated_sales,
                    'positive' : positive5,
                    'negative' :negative5
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
    text_estimated_sales = coding.classdefine(new_estimated_sales)
    return jsonify({'estimated_sales_class':new_estimated_sales,
                    'estimated_sales': text_estimated_sales,
                    'positive' : positive5,
                    'negative' :negative5
    })


@app.route('/')
def home():
    
    return "PopGame Prediction API"




if __name__ == '__main__':
    app.run(debug=True,port=1337)


