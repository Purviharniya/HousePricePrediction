from flask import Flask,request,jsonify
import util
app = Flask(__name__)

@app.route('/get_loc_names')
def get_loc_names():
    response = jsonify({
        'locations':util.get_loc_names()
    })

    response.headers.add('Access-Control-Allow-Origin','*')
    return response

@app.route('/predict',methods=['POST'])
def predict_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price':util.get_estimated_price(location,total_sqft,bhk,bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print("Server started! Peforming house price prediction")
    util.load_saved()
    app.run()