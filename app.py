from flask import Flask, request, jsonify
import coding


app = Flask(__name__)

@app.route('/predict_ratings', methods=['POST'])
def predict():
    input_data = request.get_json()
    feainput = input_data['input']
    text_estimated_sales = "0"
    try: 
        estimated_sales = coding.CatboostPredict(feainput)
    except Exception as e: print(e)
    #print(estimated_sales)
    new_estimated_sales = (str(estimated_sales).lstrip('[').rstrip(']'))
    positive5,negative5 = coding.ShapCalculate(feainput,new_estimated_sales)
    if int(new_estimated_sales) == 0:
        text_estimated_sales = '0-20000'
    elif int(new_estimated_sales) == 1:
        text_estimated_sales = '20000-50000'
    elif int(new_estimated_sales) == 2:
        text_estimated_sales = '50000-100000'
    elif int(new_estimated_sales) == 3:
        text_estimated_sales = '100000-200000'
    elif int(new_estimated_sales) == 4:
        text_estimated_sales = '200000-500000'
    elif int(new_estimated_sales) == 5:
        text_estimated_sales = '500000-1000000'
    elif int(new_estimated_sales) == 6:
        text_estimated_sales = '1000000-2000000'
    elif int(new_estimated_sales) == 7:
        text_estimated_sales = '2000000-5000000'
    elif int(new_estimated_sales) == 8:
        text_estimated_sales = '5000000-10000000'
    elif int(new_estimated_sales) == 9:
        text_estimated_sales = '10000000-200000000'
    return jsonify({'estimated sales class': new_estimated_sales,
                    'estimated sales': text_estimated_sales,
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
    if int(new_estimated_sales) == 0:
        text_estimated_sales = '0-20000'
    elif int(new_estimated_sales) == 1:
        text_estimated_sales = '20000-50000'
    elif int(new_estimated_sales) == 2:
        text_estimated_sales = '50000-100000'
    elif int(new_estimated_sales) == 3:
        text_estimated_sales = '100000-200000'
    elif int(new_estimated_sales) == 4:
        text_estimated_sales = '200000-500000'
    elif int(new_estimated_sales) == 5:
        text_estimated_sales = '500000-1000000'
    elif int(new_estimated_sales) == 6:
        text_estimated_sales = '1000000-2000000'
    elif int(new_estimated_sales) == 7:
        text_estimated_sales = '2000000-5000000'
    elif int(new_estimated_sales) == 8:
        text_estimated_sales = '5000000-10000000'
    elif int(new_estimated_sales) == 9:
        text_estimated_sales = '10000000-200000000'    
    return jsonify({'estimated sales class ': new_estimated_sales,
                    'estimated sales': text_estimated_sales,
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
