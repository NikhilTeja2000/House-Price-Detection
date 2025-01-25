from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from joblib import dump
from joblib import load
import joblib


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Task %r>' % self.id

@app.route('/')
def home():
    return render_template('index.html')  # Main page

@app.route('/about')
def about():
    return render_template('about.html')  # About page

@app.route('/dataanalysis')
def datanalysis():
    return render_template('dataanalysis.html')  # About page


# Load the model
with open('best_pipeline.pkl', 'rb') as file:
    best_pipeline = pickle.load(file)

@app.route('/adventures', methods=['GET', 'POST'])
def adventures():
    prediction = None
    if request.method == 'POST':
        # Extract form data
        form_data = request.form
        input_data = {
            'OverallQual': float(form_data.get('OverallQual', 0)),
            'GrLivArea': float(form_data.get('GrLivArea', 0)),
            'GarageCars': float(form_data.get('GarageCars', 0)),
            'TotalSF': float(form_data.get('TotalSF', 0)),
            'YearBuilt': float(form_data.get('YearBuilt', 0))
        }
        predicted_price=predict(input_data)

        prediction = {'predicted_price': f"{predicted_price[0]:,.2f}"}
        return prediction

    return render_template('adventures.html', prediction=prediction)
@app.route('/contact')
def contact():
    return render_template('contact.html')  # Contact page

def predict(input_data):
    with open('best_pipeline.pkl', 'rb') as file:
        best_pipelines = pickle.load(file)
        dump(best_pipelines, 'best_pipeline.joblib')
    best_pipeline = load('best_pipeline.joblib')

    X_train_columns = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
                       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
                       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrArea', 'ExterQual',
                       'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                       'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
                       'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                       'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
                       'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
                       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
                       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
                       'SaleCondition', 'TotalSF']

    important_fields = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalSF', 'YearBuilt']

    input_dict = {col: np.nan for col in X_train_columns}
    for col in important_fields:
        input_dict[col] = input_data.get(col, np.nan)

    input_df = pd.DataFrame([input_dict], columns=X_train_columns)

    print("Input to model:\n", input_df)  # Debug input
    predicted_log_price = best_pipeline.predict(input_df)
    predicted_price = np.expm1(predicted_log_price)

    print(f"Predicted House Price: ${predicted_price[0]:,.2f}")
    return predicted_price

#print(predict())
"""

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        task_content = request.form['content']
        new_task = Todo(content=task_content)

        try:
            db.session.add(new_task)
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue adding your task'

    else:
        tasks = Todo.query.order_by(Todo.date_created).all()
        return render_template('index.html', tasks=tasks)


@app.route('/delete/<int:id>')
def delete(id):
    task_to_delete = Todo.query.get_or_404(id)

    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'There was a problem deleting that task'

@app.route('/update/<int:id>', methods=['GET', 'POST'])
def update(id):
    task = Todo.query.get_or_404(id)

    if request.method == 'POST':
        task.content = request.form['content']

        try:
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue updating your task'

    else:
        return render_template('update.html', task=task)

"""
if __name__ == "__main__":
    app.run(debug=True)
