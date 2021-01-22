from flask import Flask, render_template, request
from flask_cors import cross_origin
#import pickle
import joblib
import numpy as np



app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("base1.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            CRIM=float(request.form['CRIM'])
            ZN = float(request.form['ZN'])
            INDUS = float(request.form['INDUS'])
            CHAS = float(request.form['CHAS'])
            NOX = float(request.form['NOX'])
            RM = float(request.form['RM'])
            AGE = float(request.form['AGE'])
            DIS = float(request.form['DIS'])
            RAD = float(request.form['RAD'])
            TAX = float(request.form['TAX'])
            PTRATIO = float(request.form['PTRATIO'])
            B = float(request.form['B'])
            LSTAT = float(request.form['LSTAT'])



            pred_arg = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]
            pred_arr = np.array(pred_arg)
            preds = pred_arr.reshape(1, -1)
            model = open("lir.pkl", "rb")
            lr_model = joblib.load(model)
            mod_prediction = lr_model.predict(preds)

            return render_template('base1.html', pred='Predicted House price will be {}'.format(mod_prediction))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('base1.html')
if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True)
    # running the app