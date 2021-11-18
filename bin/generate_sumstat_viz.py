"""
Main statistical pipelines for generating summary stats and vizualizations
for the Project Menrva dashboard.
Summary stats are saved and fed to the database while visualizations are
saved as json and served on the main site.
"""

import pickle
import sys
import altair as alt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from collections import namedtuple
from category_encoders import TargetEncoder
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests
from vega_datasets import data

# constants
years = range(2000, 2021)
proj_dir = 'C:/Users/brian/Desktop/Projects/CS_ML_DS/TDI_program/capstone/TDI_capstone_webapp/data/'
data_dir = 'D:/Projects/Project_data/Project_Menrva/'

# Column dictionary
with open(data_dir + 'Data/var_tables.pickle', 'rb') as f:
    [brf_dict, cov_dict] = pickle.load(f)

variable_list = list(set([v for v in brf_dict.values()]))

# Variable description table
variable_desc = {
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
    'TOTINDA': 'Lack of exercise',
    'FRUIT': 'Lack of fruit consumption (2013-2019)'
}

# For the purpose of time-series analysis, I opted for manual pairing of independent variable:
ts_pair = {
    'PNEUMO': 'aunempi',
    'FLUSHOT': 'aedpi',
    'RFBING': 'cwine',
    'SMOKER': 'aesapi',
    'RFDRHV': 'aunempi',
    'RFSMOK': 'aesapi',
    'TOTINDA': 'aesapi',
}


# Refer to the preprocessing scripts/notebook to see how this table was generated
state_policies = pd.read_csv(data_dir + 'Table/state_policy_tax_spending_merge.tsv', sep='\t', low_memory=False)


def load_datasets(var, n_samp='10000'):
    # Load the pre-processed data - they were saved by N and dep_variable
    with open(data_dir + 'Data/BRFSS_dataset/data_dumps/' + var + '_' + n_samp + '_peryear.pkl', 'rb') as f:
        master_table = pickle.load(f)
    # Save the hyp. testing results to COVDB
    with open(data_dir + 'Data/BRFSS_dataset/data_dumps/' + var + '_' + n_samp + '_dep_cov_imputed.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    data_dict['master_tbl'] = master_table
    return data_dict


def save_cov_tbl_viz(data_dict):
    X_scale, y_scale, cols, master_table, var, n_samp = \
        data_dict['X_scale'], data_dict['y_scale'], data_dict['cols'], data_dict['master_tbl'], data_dict['var'], \
        data_dict['n_samp']
    X_df = pd.DataFrame(X_scale)
    X_df.columns = cols + ['year']
    lin_model = sm.OLS(y_scale, X_df)
    lin_est = lin_model.fit()
    # print(lin_est.summary())
    result_df = pd.DataFrame({'beta': lin_est.params,
                              'abs(t)': abs(lin_est.tvalues)}) \
        .assign(cov_var=list(X_df.columns)).sort_values(by=['abs(t)'], ascending=False)
    result_df.to_csv(proj_dir + 'COVDB/' + var + '_' + str(n_samp) + '_COVDB.csv', index=False)

    # Save the altair plot to json
    supported_cols = list(set(result_df.index[result_df['abs(t)'] > 10]).union(['year']))
    # print(supported_cols)
    sub_dfs = []
    for cov in supported_cols:
        # TODO: expand to diff. types of plots, such as handling multiple categories of dep. vars:
        sub_tbl = master_table[[var, cov]].dropna().groupby(cov).agg({var: np.mean})
        sub_tbl[var] = sub_tbl[var] - 1
        sub_tbl['variable'] = cov
        sub_tbl['cov_value'] = sub_tbl.index.astype('int64')
        sub_dfs += [sub_tbl]

    combined_tbl = pd.concat(sub_dfs)
    COV_chart = make_cov_plot(var, combined_tbl, supported_cols, variable_desc[var])
    COV_chart.save(proj_dir + 'COVDB/' + var + '_' + str(n_samp) + '_COVCHART.json')


def make_cov_plot(var, combined_tbl, supported_cols, title):
    base = alt.Chart(combined_tbl).mark_bar().encode(
        alt.X('cov_value:O', title='Covariate category'),
        alt.Y(f'{var}:Q', title=title)
    ).properties(
        height=300,
        width=500
    )
    # A dropdown filter
    column_dropdown = alt.binding_select(options=supported_cols)
    column_select = alt.selection_single(
        fields=['variable'],
        on='doubleclick',
        clear=False,
        bind=column_dropdown,
        name='y',
        init={'variable': '_INCOMG'}
    )
    # Final chart
    filter_columns = base.add_selection(
        column_select
    ).transform_filter(
        column_select
    )

    return filter_columns


def save_geo_tbl_viz(data_dict):
    # First get the residuals:
    X_scale, y_scale, cols, var, n_samp, master_table = \
        data_dict['X_scale'], data_dict['y_scale'], data_dict['cols'], data_dict['var'], data_dict['n_samp'], data_dict[
            'master_tbl']
    X_df = pd.DataFrame(X_scale)
    # For the geographic map, we are not including year as covariate:
    X_df.columns = cols + ['year']
    X_omit_year = X_df[cols]
    lin_model = sm.OLS(y_scale, X_omit_year)
    lin_est = lin_model.fit()
    stat_df = master_table[[var, '_STATE', 'state_abbr', 'year']].assign(residual=list(lin_est.resid))
    pivot_tbl = stat_df.groupby(['_STATE', 'state_abbr', 'year']).mean().reset_index()
    min_year, max_year = min(pivot_tbl.year), max(pivot_tbl.year)
    tbl_1 = pivot_tbl.loc[pivot_tbl.year == min_year,].drop(['year'], axis=1)
    tbl_2 = pivot_tbl.loc[pivot_tbl.year == max_year,].drop(['year'], axis=1)
    plot_df = tbl_1.merge(tbl_2, on=['_STATE', 'state_abbr'], how='outer').dropna()
    plot_df = plot_df.assign(var_chng=lambda x: x[var + '_y'] - x[var + '_x'],
                             resid_chng=lambda x: x['residual_y'] - x['residual_x'])
    plot_df.columns = ['state_id', 'state_abbr', var + '_' + str(min_year), 'resid_' + str(min_year),
                       var + '_' + str(max_year), 'resid_' + str(max_year), 'var_chng', 'resid_chng']
    plot_df.to_csv(proj_dir + 'GEODB/' + var + '_' + str(n_samp) + '_GEODB.csv', index=False)

    # Save the altair plot to json
    supported_cols = [var + '_' + str(min_year), var + '_' + str(max_year), 'resid_chng']
    GEO_chart = make_geo_plot(var, plot_df, supported_cols, variable_desc[var])
    GEO_chart.save(proj_dir + 'GEODB/' + var + '_' + str(n_samp) + '_GEOCHART.json')


def make_geo_plot(var, plot_df, supported_cols, title):
    states = alt.topo_feature(data.us_10m.url, 'states')

    # A dropdown filter
    column_dropdown = alt.binding_select(options=supported_cols)
    column_select = alt.selection_single(
        fields=['variable'],
        bind=column_dropdown,
        name=var,
        init={'variable': 'resid_chng'}
    )

    filter_chart = alt.Chart(states).mark_geoshape().encode(
        color='value:Q',
        tooltip=['state_abbr:O', 'value:Q']
    ).transform_lookup(
        lookup='id',
        from_=alt.LookupData(plot_df, 'state_id', ['state_abbr'] + supported_cols)
    ).transform_fold(
        supported_cols,  # Preserve the State column, fold the rest
        ['variable', 'value']
    ).properties(
        width=500,
        height=300,
        title=title
    ).project(
        type='albersUsa'
    ).resolve_scale(
        color='independent'
    ).add_selection(
        column_select
    ).transform_filter(
        column_select
    )

    return filter_chart


def save_corr_tbl_viz(data_dict):
    X_scale, y_scale, cols, var, n_samp, master_table = \
        data_dict['X_scale'], data_dict['y_scale'], data_dict['cols'], data_dict['var'], data_dict['n_samp'], data_dict[
            'master_tbl']
    X_df = pd.DataFrame(X_scale)
    X_df.columns = cols + ['year']
    X_covs = X_df[cols]
    X_covs_t = X_df
    # Perform target encoding on state variables
    state_t = TargetEncoder()
    state_t.fit(np.array(master_table['state_abbr']).reshape(-1, 1),
                np.array(master_table[var]).reshape(-1, 1))
    X_states = state_t.transform(np.array(master_table['state_abbr']).reshape(-1, 1))
    scaler = StandardScaler()
    X_st_scale = scaler.fit_transform(X_states)
    X_covs_s = X_covs.assign(state_scale=X_st_scale)
    X_covs_st = X_covs_t.assign(state_scale=X_st_scale)

    quest_table = (master_table[[var, 'state_abbr', 'year']]) \
        .assign(resid=sm.OLS(y_scale, X_covs).fit().resid) \
        .assign(resid_s=sm.OLS(y_scale, X_covs_s).fit().resid) \
        .assign(resid_t=sm.OLS(y_scale, X_covs_t).fit().resid) \
        .assign(resid_st=sm.OLS(y_scale, X_covs_st).fit().resid)

    pivot_table = quest_table.groupby(['state_abbr', 'year']).mean()
    pivot_table = pivot_table.merge(right=state_policies, left_on=['state_abbr', 'year'],
                                    right_on=['st', 'year'], how='left')
    pivot_table = pivot_table.loc[~pd.isna(pivot_table['st']),]

    # Looks like regression on resid_st is enough:
    col = 'resid_st'
    test_table = pivot_table.loc[~pd.isna(pivot_table[col]),]
    test_cols = test_table.columns[7:]

    # Try first without multiprocessing
    def get_tstat(test_col):
        return sm.OLS(test_table[col], test_table[test_col], missing='drop').fit()

    test_models = list(map(get_tstat, test_cols))
    result_df = pd.DataFrame({'beta': [model.params[0] for model in test_models],
                              'abs(t)': [abs(model.tvalues[0]) for model in test_models],
                              'pvalue': [model.pvalues[0] for model in test_models]}).sort_values(by=['pvalue'])
    result_df = result_df.assign(indep=list(test_cols[result_df.index]))
    result_df.to_csv(proj_dir + 'CORRDB/' + var + '_' + str(n_samp) + '_CORRDB.csv', index=False)

    supported_cols = list(result_df.indep[:5])
    sub_dfs = []
    for indep in supported_cols:
        # TODO: expand to diff. types of plots, such as handling multiple categories of dep. vars:
        sub_tbl = test_table[['st', 'year', 'resid_st']].assign(variable=indep).assign(var_value=test_table[indep])
        sub_dfs += [sub_tbl]

    combined_tbl = pd.concat(sub_dfs)
    combined_tbl.columns = ['st', 'year', var, 'variable', 'var_value']
    CORR_chart = make_corr_plot(var, combined_tbl, supported_cols, variable_desc[var])
    CORR_chart.save(proj_dir + 'CORRDB/' + var + '_' + str(n_samp) + '_CORRCHART.json')


def make_corr_plot(var, combined_tbl, supported_cols, title):
    # dropdown filter
    column_dropdown = alt.binding_select(options=supported_cols)
    column_select = alt.selection_single(
        fields=['variable'],
        on='doubleclick',
        clear=False,
        bind=column_dropdown,
        name='x'
    )

    points = alt.Chart(combined_tbl).mark_point().encode(
        alt.X('var_value:Q', title='Independent variable value'),
        alt.Y(f'{var}:Q', title=title),
        color='year:O',
        tooltip=['st:N', 'year:O']
    )
    line = points.transform_regression('var_value', f'{var}').mark_line()

    combined = points + line
    combined = combined.properties(
        height=300,
        width=450
    ).add_selection(
        column_select
    ).transform_filter(
        column_select
    )

    return combined


def save_ts_tbl_viz(data_dict):
    Row = namedtuple('Row', ['indep', 'count', 'state', 'minPval', 'lag'])
    X_scale, y_scale, cols, var, n_samp, master_table = \
        data_dict['X_scale'], data_dict['y_scale'], data_dict['cols'], data_dict['var'], data_dict['n_samp'], data_dict[
            'master_tbl']
    X_df = pd.DataFrame(X_scale)
    X_df.columns = cols + ['year']
    X_covs = X_df[cols]
    quest_table = (master_table[[var, 'state_abbr', 'year']]) \
        .assign(resid=sm.OLS(y_scale, X_covs).fit().resid)
    pivot_table = quest_table.groupby(['state_abbr', 'year']).mean()
    pivot_table = pivot_table.reset_index()
    pivot_table.year = pivot_table.year+1
    pivot_table = pivot_table.merge(right = state_policies, left_on = ['state_abbr', 'year'],
                                    right_on = ['st', 'year'], how = 'left')
    pivot_table = pivot_table.loc[~pd.isna(pivot_table['state_abbr']),]
    # state policy data quality is really bad after 2015: cut off data until this year
    pivot_table = pivot_table.loc[pivot_table.year<=2015,]

    test_cols = pivot_table.columns[5:]
    n_lag = 3

    combined_dict = {}
    for indep_variable in test_cols:
        count = 0
        state_list = []
        pval_list = []
        lag_list = []
        for state in pd.unique(pivot_table['state_abbr']):
            sub_tbl = pivot_table.query(f'state_abbr == "{state}"')
            if sub_tbl.shape[0] > 0:
                sub_tbl = sub_tbl.query(indep_variable + ' > 0')
                query_tbl = pd.concat(
                    [pd.Series(np.diff(sub_tbl[var])), pd.Series(np.diff(sub_tbl[indep_variable]))], axis=1)
                if len(pd.unique(query_tbl.iloc[:, 1])) > 1:
                    try:
                        gc_model = grangercausalitytests(query_tbl, n_lag, verbose=False)
                    except:
                        continue
                    pvals = [gc_model[x + 1][0]['ssr_ftest'][1] for x in range(n_lag)]
                    if np.any([pval < 0.05 for pval in pvals]):
                        count += 1
                        state_list.append(state)
                        lag = int(np.where(pvals == min(pvals))[0])
                        lag_list.append(lag)
                        pval_list.append(pvals[lag])
        combined_dict[indep_variable] = {'count': count, 'state_list': state_list, 'pval_list': pval_list,
                                         'lag_list': lag_list}
    top_vars = sorted(list(test_cols), key=(lambda x: combined_dict[x]['count']), reverse=True)[:15]
    db_data = []
    for top_var in top_vars:
        index = combined_dict[top_var]
        if index['count'] > 0:
            min_val = min(index['pval_list'])
            ind = int(np.where(min_val == min(index['pval_list']))[0])
            db_data.append(
                Row(top_var, index['count'], index['state_list'][ind], index['pval_list'][ind], index['lag_list'][ind]))

    if len(db_data) > 0:
        DB_df = pd.DataFrame(db_data)
        DB_df.to_csv(proj_dir + 'TSDB/' + var + '_' + str(n_samp) + '_TSDB.csv', index=False)

        var_target = ts_pair[var]
        for row in db_data:
            if row[0] == var_target:
                break
        state = row[2]
        sub_tbl = pivot_table.query(f'state_abbr == "{state}"')[['year', dep_variable, var_target]]
        TS_chart = make_ts_plot(sub_tbl, var, var_target, variable_desc[var], state)
        TS_chart.save(proj_dir + 'TSDB/' + var + '_' + str(n_samp) + '_TSCHART.json')

    else:
        print('No time series analyses available!')


def make_ts_plot(sub_tbl, dep_variable, var_target, title, state):
    base = alt.Chart(sub_tbl).encode(
        alt.X('year:O', title='Year')
    ).properties(
        height=300,
        width=450,
        title='Plotting ' + dep_variable + ' and ' + var_target + ' for ' + state + ':'
    )
    l1 = base.mark_line(color='#57A44C').encode(
        alt.Y(f'{var_target}:Q', scale=alt.Scale(domain=[min(sub_tbl[var_target]), max(sub_tbl[var_target])])),
        tooltip=['year:O', f'{var_target}:Q']
    )
    l2 = base.mark_line(color='#5276A7').encode(
        alt.Y(f'{dep_variable}:Q', title=title,
              scale=alt.Scale(domain=[min(sub_tbl[dep_variable]), max(sub_tbl[dep_variable])])),
        tooltip=['year:O', f'{dep_variable}:Q']
    )
    combined = alt.layer(l1, l2).resolve_scale(
        y='independent'
    )

    return combined


if __name__ == '__main__':
    """
    Main pipeline - functionality to support which sumstats to accumulate:
    modes = cov, geo, corr, ts - covariate, geography, correlation and time-series analyses
    default: all - calculates sumstats for all four modes
    N = Number of data points per year to consider. A new value of N will output new intermediate data files
    default: 10000
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', dest='modes', type=str, help='Which summary stats and visualizations to generate?')
    parser.add_argument('--N', dest='N', type=str, help='How many samples to take per year')
    args = parser.parse_args()

    modes = []
    if args.modes == 'all':
        modes = ['cov', 'geo', 'corr', 'ts']
    else:
        modes = args.modes.split(',')
    print(modes)
    if len(set(modes).difference(['cov', 'geo', 'corr', 'ts'])) > 0:
        sys.exit('Inappropriate mode name(s)! Only "cov", "geo", "corr", "ts" or "all" allowed.')
    # Obtain number of samples per year - need to have run preprocessing script with this N value
    try:
        N = int(args.N)
        assert N >= 1000
    except (ValueError, AssertionError) as e:
        sys.exit('N should be a positive integer greater than 1000.')

    for dep_variable in variable_list:
        print('Processing: ' + dep_variable + ' with size ' + args.N)
        data_dict = load_datasets(dep_variable, args.N)
        if 'cov' in modes:
            save_cov_tbl_viz(data_dict)
        if 'geo' in modes:
            save_geo_tbl_viz(data_dict)
        if 'corr' in modes:
            save_corr_tbl_viz(data_dict)
        if 'ts' in modes:
            save_ts_tbl_viz(data_dict)
