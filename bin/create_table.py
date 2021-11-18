#!/usr/bin/env python
"""
Create SQL table from an already existing database.
"""
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

load_dotenv()
URI = os.getenv('URI_DB')
data_dir = os.getenv('DATA_DIR')

variable_desc = {
    'FRUIT_inv': 'Lack of fruit consumption',
    'PNEUMO': 'Lack of pneumonia vaccination',
    'FLUSHOT': 'Lack of flu vaccination',
    'VEGE': 'Lack of vegetable consumption',
    'RFBING': 'Whether a binge drinker',
    'SMOKER': 'Smoker category',
    'HTCARE': 'Lack of healthcare coverage',
    'SLEPTIM': 'Sleep amount',
    'RFDRHV': 'Whether a heavy drinker',
    'RFSMOK': 'Whether a current smoker',
    'TOTINDA': 'Lack of exercise',
    'FRUIT': 'Lack of fruit consumption'
}


def create_cov_table(engine, n_samp):
    DB_data = []
    table_name = 'COVDB'
    for dep_variable in variable_desc:
        sub_df = pd.read_csv(data_dir + table_name + '/' + dep_variable + '_' + str(n_samp) + '_' + table_name + '.csv')
        sub_df.columns = ['beta', 'absTstat', 'covVar']
        sub_df['depVar'] = dep_variable
        DB_data.append(sub_df)

    DB_df = pd.concat(DB_data, axis=0)
    scaled_pvals = scipy.stats.norm.sf(DB_df['absTstat']) * 2 * DB_df.shape[0]
    DB_df['pValue'] = [min(pval, 1) for pval in scaled_pvals]
    DB_df['significant'] = DB_df['pValue'] < 0.01
    DB_df.to_sql(table_name, engine, if_exists='replace')
    return table_name


def create_geo_table(engine, n_samp):
    DB_data = []
    table_name = 'GEODB'
    for dep_variable in variable_desc:
        sub_df = pd.read_csv(data_dir + table_name + '/' + dep_variable + '_' + str(n_samp) + '_' + table_name + '.csv')
        first_year = sub_df.columns[2].split('_')[-1]
        last_year = sub_df.columns[4].split('_')[-1]
        sub_df.columns = ['stateId', 'stateAbbr', 'depVarFirst', 'residFirst', 'depVarLast', 'residLast', 'depVarChng', 'residChng']
        sub_df['depVar'] = dep_variable
        sub_df['firstYear'] = int(first_year)
        sub_df['lastYear'] = int(last_year)
        DB_data.append(sub_df)

    DB_df = pd.concat(DB_data, axis=0)
    DB_df.to_sql(table_name, engine, if_exists='replace')
    return table_name


def create_corr_table(engine, n_samp):
    DB_data = []
    table_name = 'CORRDB'
    for dep_variable in variable_desc:
        sub_df = pd.read_csv(data_dir + table_name + '/' + dep_variable + '_' + str(n_samp) + '_' + table_name + '.csv')
        sub_df.columns = ['beta', 'absTstat', 'pValue', 'indepVar']
        sub_df['depVar'] = dep_variable
        DB_data.append(sub_df)

    DB_df = pd.concat(DB_data, axis=0)
    scaled_pvals = DB_df['pValue']*DB_df.shape[0]
    DB_df['pValue'] = [min(pval, 1) for pval in scaled_pvals]
    DB_df['significant'] = DB_df['pValue'] < 0.01
    DB_df.to_sql(table_name, engine, if_exists='replace')
    return table_name


def create_ts_table(engine, n_samp):
    DB_data = []
    table_name = 'TSDB'
    for dep_variable in variable_desc:
        if os.path.exists(data_dir + table_name + '/' + dep_variable + '_' + str(n_samp) + '_' + table_name + '.csv'):
            sub_df = pd.read_csv(data_dir + table_name + '/' + dep_variable + '_' + str(n_samp) + '_' + table_name + '.csv')
            sub_df['depVar'] = dep_variable
            DB_data.append(sub_df)

    DB_df = pd.concat(DB_data, axis=0)
    DB_df.to_sql(table_name, engine, if_exists='replace')
    return table_name


if __name__ == '__main__':
    db_engine = create_engine(URI)
    n_samp = 100000
    create_cov_table(db_engine, n_samp)
    create_geo_table(db_engine, n_samp)
    create_corr_table(db_engine, n_samp)
    create_ts_table(db_engine, n_samp)
