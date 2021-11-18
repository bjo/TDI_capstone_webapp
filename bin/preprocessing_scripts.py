"""
Preprocessing scripts for Project Menrva
Takes the BRFSS dataset from CDC and CSPP dataset from MSU
and outputs them into data formats usable for the main script.
"""
import os
import pickle
import random
import sys
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# constants
years = range(2000, 2021)
data_dir = 'D:/Projects/Project_data/Project_Menrva/'
# Column dictionary - refer to the notebook for how these dictionaries were constructed
with open(data_dir + 'Data/var_tables.pickle', 'rb') as f:
    [brf_dict, cov_dict] = pickle.load(f)

variable_list = list(set([v for v in brf_dict.values()]))


def get_file_location(year):
    if year < 2011:
        fname = data_dir + 'Data/BRFSS_dataset/{}xpt/CDBRFS{}.XPT'.format(year, str(year)[2:4])
    else:
        fname = data_dir + 'Data/BRFSS_dataset/{}xpt/LLCP{}.XPT'.format(year, year)
    return fname


# This portion was run on the notebook
def obtain_data():
    # read in a website with state FIPS code:
    raw_tbl = pd.read_html('https://www.mcc.co.mercer.pa.us/dps/state_fips_code_listing.htm')[0]
    state_tbl = pd.concat([raw_tbl.iloc[1:28, 0:3].rename(columns={0: 'state_abbr', 1: 'FIPS Code', 2: 'state_name'}),
                           raw_tbl.iloc[1:29, 3:6].rename(columns={3: 'state_abbr', 4: 'FIPS Code', 5: 'state_name'})])
    state_tbl['FIPS'] = state_tbl['FIPS Code'].astype('int64')

    # Part 1 - Iterate through the years and output cleaned tables for BRF and covariates
    for year in years:
        print(year)
        df = pd.read_sas(get_file_location(year))
        # obtain brf and cov columns
        year_brf_table = df[[x for x in df.columns if x in brf_dict]]
        year_cov_table = df[[x for x in df.columns if x in cov_dict]]
        # rename columns and indexes
        year_brf_table.columns = [brf_dict[x] for x in year_brf_table.columns]
        year_brf_table.index = [str(year) + '_' + str(i + 1) for i in range(year_brf_table.shape[0])]
        year_cov_table.columns = [cov_dict[x] for x in year_cov_table.columns]
        year_cov_table.index = [str(year) + '_' + str(i + 1) for i in range(year_cov_table.shape[0])]
        # save brf and cov tables
        year_brf_table = year_brf_table.fillna(9999).astype('int64')
        year_brf_table.to_csv(data_dir + 'Table/BRFSS/cleaned_brf_{}.tsv'.format(year), sep='\t', index=False)

        year_cov_table = year_cov_table.fillna(9999).astype('int64').merge(right=state_tbl[['FIPS', 'state_abbr']],
                                                                           left_on='_STATE', right_on='FIPS',
                                                                           how='left').drop(columns=['FIPS'])
        year_cov_table.to_csv(data_dir + 'Table/BRFSS/cleaned_cov_{}.tsv'.format(year), sep='\t', index=False)


# Part 2 - Construct sub-setted (by year) pickle dumps for each variable
def brf_process_df(df):
    # Each dataframe has the following format:
    # first column is the depedent variable
    # last column is the state name
    # all columns in between, _XXX, are the covariates
    var_max_value = {}
    var_max_value['SLEPTIM'] = 24
    var_max_value['SMOKER'] = 4
    var_max_value['_AGEG5YR'] = 13
    var_max_value['_CHLDCNT'] = 6
    var_max_value['_EDUCAG'] = 4
    var_max_value['_INCOMG'] = 5
    var_max_value['_ASTHM'] = 3
    var_max_value['FRUIT_inv'] = 4
    var_set = set(v for v in brf_dict.values()).union(set(v for v in cov_dict.values()))

    for v in var_set:
        if v not in df.columns or v == '_STATE' or v == 'state_abbr':
            continue
        elif v == '_RACE':
            # collapse all non-white into one category for now
            df.loc[(df[v] > 1) & (df[v] <= 8), v] = 2
            df.loc[(df[v] > 8), v] = np.nan
        else:
            df.loc[(df[v] > var_max_value.get(v, 2)), v] = np.nan
    if '_SEX.1' in df.columns:
        df = df.drop(columns=['_SEX.1'])
    return df


def obtain_pkl_dump(dep_variable, N=10000):
    # N = 10000  # Vary this variable to control how many rows are to be taken from each year
    print('Pipeline for: ', dep_variable)
    yr_dfs = []
    for year in years:
        # we can expect ~300k to 450k rows per year, and we have data for a 21-year period
        # For prototyping, 10k was picked. For production, 100k was picked.
        reader = pd.read_csv(data_dir + 'Table/BRFSS/cleaned_brf_{}.tsv'.format(year), sep='\t', iterator=True)
        if dep_variable not in reader.get_chunk(5).columns:
            continue
        df_brf = pd.read_csv(data_dir + 'Table/BRFSS/cleaned_brf_{}.tsv'.format(year), sep='\t')
        df_cov = pd.read_csv(data_dir + 'Table/BRFSS/cleaned_cov_{}.tsv'.format(year), sep='\t')
        df = pd.concat([df_brf[[dep_variable]], df_cov], axis=1)
        # define a pre-processing pipeline function
        df = brf_process_df(df)
        df = df.loc[~np.isnan(df[dep_variable]),]
        # add year information to the df
        df['year'] = year
        random.seed(42)
        sub_df = df.sample(min(N, df.shape[0]))
        yr_dfs += [sub_df]
    combined_df = pd.concat(yr_dfs)
    print('Dumping combined dataset')
    pickle.dump(combined_df, open(data_dir + 'Data/BRFSS_dataset/data_dumps/' + dep_variable + '_' + str(N) + '_peryear.pkl', 'wb'))
    return combined_df


def obtain_imput_scaled_cov(var, master_tbl, N=10000):
    # If imputed table has already been saved, load it - otherwise, calculate and save
    write_loc = data_dir + 'Data/BRFSS_dataset/data_dumps/' + var + '_' + str(N) + '_dep_cov_imputed.pkl'
    if os.path.exists(write_loc):
        print('Imputed scaled covariates already exist!')
    else:
        # remove state variables, and impute missing values
        y = master_tbl[var]
        cols = list(set(master_tbl.columns).difference([var, '_STATE', 'state_abbr', 'year']))
        X = master_tbl[cols]
        # run standard scaling for years
        scale_year = MinMaxScaler()
        scale_year.fit(np.array(master_tbl['year']).reshape(-1, 1))
        X['year_scaled'] = scale_year.transform(np.array(master_tbl['year']).reshape(-1, 1))
        # we need imputation for missing values (nan)
        imp = IterativeImputer(max_iter=50, random_state=42)
        imp.fit(X)
        X_imput = imp.transform(X)
        scaler = StandardScaler()
        scaler.fit(X_imput)
        X_scale = scaler.transform(X_imput)
        scaler.fit(np.array(y).reshape(-1, 1))
        y_scale = scaler.transform(np.array(y).reshape(-1, 1))
        data_dict = {'X_scale': X_scale, 'y_scale': y_scale, 'cols': cols, 'n_samp': N, 'var': var}
        print('Dumping imputed scaled covariates')
        with open(write_loc, 'wb') as f:
            pickle.dump(data_dict, f)


# This portion was run on the notebook
def calculate_state_policies():
    # Merge two pre-processed documents: state_policy_spending.tsv and state_policy_tax.tsv
    state_spending = pd.read_csv(data_dir + 'Table/state_policy_spending.tsv', sep = '\t', low_memory = False).drop_duplicates(['year','st'])
    state_tax = pd.read_csv(data_dir + 'Table/state_policy_tax.tsv', sep = '\t', low_memory = False).drop_duplicates(['year','st'])
    state_merged = pd.concat([state_spending, state_tax.iloc[:,3:]], axis = 1).drop(columns=['state'])
    state_merged = state_merged.sort_values(by=['st', 'year'])
    state_merged.index = [i for i in range(state_merged.shape[0])]

    state_merged['cbeer'] = state_merged[['cbeertex', 'cbeerp','cbeert']].fillna(0).sum(axis=1)
    state_merged['cwine'] = state_merged[['cwinetex', 'cwinep','cwinet']].fillna(0).sum(axis=1)
    state_merged['cspir'] = state_merged[['cspirtex', 'cspirp','cspirt']].fillna(0).sum(axis=1)
    cols_to_drop = ['adebtpia', 'aasstpia', 'aincloc4', 'aincstat4', 'ainctot3', 'ainctot4', 'atotpib', 'atotpic',
                   'cigtax', 'icigtaxraw', 'beer_tax', 'beer_tax_rank', 'spirit_tax', 'spirit_tax_rank',
                   'wine_tax', 'wine_tax_rank', 'cbeertav', 'cwinetav', 'cspirtav',
                   'cbeertex', 'cbeerp','cbeert', 'cwinetex', 'cwinep','cwinet', 'cspirtex', 'cspirp', 'cspirt']
    state_merged = state_merged.drop(cols_to_drop, axis=1)

    # Design choice: some entries are NaN separated by two numbers - if so, fill them in with the average
    for col in state_merged.columns[2:]:
        for i in range(1, state_merged.shape[0]-1):
            if pd.isna(state_merged.loc[i, col]):
                prev_val = state_merged.loc[i-1, col]
                next_val = state_merged.loc[i+1, col]
                if (state_merged.loc[i-1, 'st'] == state_merged.loc[i+1, 'st']) and (not pd.isna(prev_val) and not pd.isna(next_val)):
                    state_merged.loc[i, col] = (prev_val + next_val) / 2
                    print('fill:', col, str(state_merged.loc[i-1, 'year']))
                    print(prev_val, next_val)

    state_merged.to_csv(data_dir + 'Table/state_policy_tax_spending_merge.tsv', sep = '\t', index = False)


if __name__ == '__main__':
    """
    Preprocessing script - Performs preprocessing portion of the data analysis pipeline
    N = Number of data points per year to consider. A new value of N will output new intermediate data files
    default: 10000
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', dest='N', type=str, help='How many samples to take per year')
    args = parser.parse_args()
    # Obtain number of samples per year
    try:
        N = int(args.N)
        assert N >= 1000
    except (ValueError, AssertionError) as e:
        sys.exit('N should be a positive integer greater than 1000.')

    for var in variable_list:
        master_table = obtain_pkl_dump(var, N)
        obtain_imput_scaled_cov(var, master_table, N)


