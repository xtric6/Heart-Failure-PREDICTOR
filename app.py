#load core packages
from flask import Flask,render_template,request
import pandas as pd


#initialize app
app = Flask(__name__)


#EDA PKGS
import os
import joblib
import numpy as np
import sklearn


#loading the model
def load_model(model_file):
	loaded_model= joblib.load(open(os.path.join(model_file),'rb'))
	return loaded_model

#get keys
def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key


#route
@app.route('/')
def index():
	return render_template('index.html')

#templating
@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/dataset')
def dataset():
	df = pd.read_csv('static/dataset/df_frontend.csv',index_col=[0])
	df.age = df['age'].round(0).astype(int)
	return render_template('Dataset.html',df_table=df)



@app.route('/predict', methods=['GET','POST'])
def predict():
	if request.method == 'POST':
		age = request.form['age']
		smoking = request.form['smoking']
		ejection_fraction = request.form['ejection_fraction']
		sex = request.form['sex']
		serum_creatinine = float(request.form['serum_creatinine'])
		creatinine_phosphokinase = request.form['creatinine_phosphokinase']
		time = request.form['time']
		pretty_data = {'age':age,'ejection_fraction': ejection_fraction,'smoking':smoking,'sex':sex,'serum_creatinine':serum_creatinine,'creatinine_phosphokinase':creatinine_phosphokinase,'time':time}
		sample_data = [age,ejection_fraction,smoking,sex,serum_creatinine,creatinine_phosphokinase,time]
		encoded_data = [float(int(i)) for i in sample_data]
		model =load_model('models/heart_failure_predictionBC4.pkl')
		prediction = model.predict(np.array(encoded_data).reshape(1,-1))

		prediction_label = {'You are at risk of a heart failure':1,'You are not at risk of a heart failure':0}

		final_result = get_key(prediction,prediction_label)
		predict_proba = model.predict_proba(np.array(encoded_data).reshape(1,-1))
		
		#pred_probability_score = {'Not at risk':predict_proba[0][0]*100 , 'At risk':predict_proba[0][1]*100 }

		pred_probability_score = ('Not at risk: {:.0f}%'.format(predict_proba[0][0]*100) , 'At risk: {:.0f}%'.format(predict_proba[0][1]*100))

		return render_template('index.html',pred_probability_score=pred_probability_score,final_result=final_result,pretty_data=pretty_data)



if __name__ == '__main__':
	app.run(debug=True)