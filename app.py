import pickle
from flask import Flask,render_template,request
import numpy as np

model=pickle.load(open('model.pkl','rb'))
app=Flask(__name__)
@app.route('/')
def man():
    return render_template("index.html")
@app.route('/predict',methods=["POST"])
def home():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output=prediction[0]
    return render_template ("index.html",prediction_text="the price of the house will be {}".format(output))
if __name__=="__main__":
    app.run(debug=True)