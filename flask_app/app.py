from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv
from sqlalchemy import create_engine
import altair as alt
import json
import os

from database import query_db

# Create Flask webapp object
app = Flask(__name__)
load_dotenv()
URI_DB = os.getenv('URI_DB')
db_engine = create_engine(URI_DB)

variable_desc = {
    'TOTINDA': 'Lack of exercise',
    'FRUIT_inv': 'Fruit consumption (2000-2009)',
    'PNEUMO': 'Lack of pneumonia vaccination',
    'FLUSHOT': 'Lack of flu vaccination',
    'VEGE': 'Lack of vegetable consumption',
    'RFBING': 'Whether a binge drinker',
    'SMOKER': 'Smoker category',
    'HTCARE': 'Lack of healthcare coverage',
    'SLEPTIM': 'Sleep amount',
    'RFDRHV': 'Whether a heavy drinker',
    'RFSMOK': 'Whether a current smoker',
    'FRUIT': 'Lack of fruit consumption (2013-2019)'
}


# decorator syntax - associated a URL with the python function
# instead of dealing with redirects, we can associate multiple URLs to a function
@app.route('/')
@app.route('/entry', methods=['GET', 'POST'])
def entry_page() -> 'html':
    if request.method == 'GET':
        var = 'TOTINDA'
    if request.method == 'POST':
        var = request.form.get('dep_variable')
    n_samp = 100000

    with open('flask_app/static/chart_json/' + var + '_' + str(n_samp) + '_COVCHART.json') as f:
        cov_chart = f.readline()
    cov_data = query_db('COVDB', var, db_engine)
    with open('flask_app/static/chart_json/' + var + '_' + str(n_samp) + '_GEOCHART.json') as f:
        geo_chart = f.readline()
    geo_data = query_db('GEODB', var, db_engine)
    with open('flask_app/static/chart_json/' + var + '_' + str(n_samp) + '_CORRCHART.json') as f:
        corr_chart = f.readline()
    corr_data = query_db('CORRDB', var, db_engine)

    chart_list = [cov_chart, geo_chart, corr_chart]
    data_list = [cov_data, geo_data, corr_data]

    if os.path.exists('flask_app/static/chart_json/' + var + '_' + str(n_samp) + '_TSCHART.json'):
        with open('flask_app/static/chart_json/' + var + '_' + str(n_samp) + '_TSCHART.json') as f:
            ts_chart = f.readline()
        ts_data = query_db('TSDB', var, db_engine)
        chart_list.append(ts_chart)
        data_list.append(ts_data)

    return render_template('entry.html', the_title='Menrva Dashboard', the_var=var,
                           the_var_desc=variable_desc, the_charts=chart_list, the_data=data_list)


# app.run()
# If run locally, it will run in debug mode. Otherwise, the deployment will run
# its own version of app.run()
if __name__ == '__main__':
    app.run(debug=True)
