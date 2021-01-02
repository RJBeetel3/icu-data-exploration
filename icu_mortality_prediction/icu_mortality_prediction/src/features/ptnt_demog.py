import os
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import yaml
import numpy as np
from icu_mortality_prediction import DATA_DIR


def quant_cats(feature, Q1, Q2, Q3):
    
    if feature <=Q1:
        return 'Q0'
    elif (feature >Q1 and feature <= Q2):
        return 'Q1'
    elif (feature > Q2 and feature <= Q3):
        return 'Q2'
    elif feature > Q3:
        return 'Q3'


def import_demog_data():
    
    print("Importing patient demographic data")  
    ptnt_demog = pd.read_csv('../data/Ptnt_Demog_First24.csv')
    return ptnt_demog
    
    
def convert_datetimes(ptnt_demog2):
    
    dates_and_times = ['dob', 'admittime', 'dischtime', 'intime', 'outtime', 'deathtime']
    for thing in dates_and_times:
        print("converting {}".format(thing))
        new_series = pd.to_datetime(ptnt_demog2.loc[:,thing])
        ptnt_demog2.loc[:,thing] = new_series

    return ptnt_demog2


def calculate_age(row):

    if (pd.notnull(row['intime']) & pd.notnull(row['dob'])):
        age_val = len(pd.date_range(end=row['intime'], start=row['dob'], freq='A'))
    else:
        age_val = np.nan
    return age_val

def calculate_icu_stay_length(row):

    if (pd.notnull(row['intime']) & pd.notnull(row['outtime'])):
        icu_stay_val = len(pd.date_range(end=row['outtime'], start=row['intime'], freq='H'))
    else:
        icu_stay_val = np.nan
    return icu_stay_val

def calculate_hospital_stay(row):

    if (pd.notnull(row['admittime']) & pd.notnull(row['dischtime'])):
        hosp_stay_val = len(pd.date_range(end=row['dischtime'], start=row['admittime'], freq='H'))
    else:
        hosp_stay_val = np.nan
    return hosp_stay_val

def reconfigure_patient_demographics_columns(ptnt_demog_df):

    print("Reconfiguring columns")
    cols = list(ptnt_demog_df.columns)
    cols.pop(cols.index('icd9_code'))
    cols.pop(cols.index('icd9_code.1'))
    cols.pop(cols.index('short_title'))
    cols.pop(cols.index('intime'))
    cols.pop(cols.index('outtime'))
    cols.pop(cols.index('admittime'))
    cols.pop(cols.index('dischtime'))
    cols.pop(cols.index('seq_num'))
    cols.pop(cols.index('dob'))

    # cols.insert(0, cols.pop(cols.index('icustay_id')))
    cols.insert(0, cols.pop(cols.index('hadm_id')))
    cols.insert(1, cols.pop(cols.index('age')))
    cols.insert(2, cols.pop(cols.index('icu_stay')))
    cols.insert(3, cols.pop(cols.index('hosp_stay')))
    cols.insert(len(cols), cols.pop(cols.index('hospital_expire_flag')))
    ptnt_demog_df = ptnt_demog_df[cols].copy()
    return ptnt_demog_df

def truncate_age_values(ptnt_demog_df):

    age_replace_vals = list(ptnt_demog_df[ptnt_demog_df['age'] > 110]['age'].unique())
    ptnt_demog_df['age'].replace(age_replace_vals, np.nan, inplace=True)

    return ptnt_demog_df

def calculate_age_icu_and_hospital_stay_durations(ptnt_demog_df):

    '''
    Calculates age and duration of stays in ICU and hospital
    Complex function but leverages the single call to iterrows for efficiency
    :param ptnt_demog_df:
    :return ptnt_demog_df:
    '''
    print("Calculating ages, duration of stays")
    # len(pd.date_range()) APPEARS TO TAKE A VERY LONG TIME
    for index, row in ptnt_demog_df.iterrows():
        age_val = calculate_age(row)
        icu_stay_val = calculate_icu_stay_length(row)
        hosp_stay_val = calculate_hospital_stay(row)
        ptnt_demog_df.at[index, 'age'] = age_val
        ptnt_demog_df.at[index, 'icu_stay'] = icu_stay_val
        ptnt_demog_df.at[index, 'hosp_stay'] = hosp_stay_val

    return ptnt_demog_df
    

def load_diagnoses_definitions():
    """

    Returns
    -------
    definitions: dictionary
                    contains ICD9 codes for diagnoses definitions from HCUP CCS 2015
    """
    print("creating diagnoses definitions")
    definitions_path = os.path.join(DATA_DIR, 'external/hcup_ccs_2015_definitions.yaml')
    definitions = yaml.load(open(definitions_path, 'r'), Loader=yaml.FullLoader)
    return definitions


def create_diagnoses_defs(ptnt_demog_df):

    """


    Parameters
    ----------
    ptnt_demog_df: pandas DataFrame
                    patient demographic data
    """

    print("creating diagnoses definitions")
    definitions = load_diagnoses_definitions()

    diagnoses = ptnt_demog_df[['hadm_id', 'icd9_code', 'short_title']].copy()

    """
    create mapping of hcup_ccs_2015_definitions to diagnoses icd9 codes
    resulting dictionary has codes as keys, the corresponding diagnosis and whether that diagnosis was used in 
    benchmarking as values
    """
    def_map = {}
    for dx in definitions:
        for code in definitions[dx]['codes']:
            def_map[code] = (dx, definitions[dx]['use_in_benchmark'])

    print("map created")
    # map hcup_ccs_2015 definitions to icd9 diagnoses codes
    # map diagnosis name to 'HCUP_CCS_2015' and whether it is used in the benchmark exercise in 'USE_IN_BENCHMARK'
    diagnoses['HCUP_CCS_2015'] = diagnoses.icd9_code.apply(lambda c: def_map[c][0] if c in def_map else None)
    diagnoses['USE_IN_BENCHMARK'] = diagnoses.icd9_code.apply(lambda c: int(def_map[c][1]) if c in def_map else None)

    # create dataframe from the def_map dict so that we can isolate the 
    # definitions that are used in benchmarking
    def_map_df = pd.DataFrame.from_dict(def_map, orient='index')
    def_map_df.columns = ['Diagnoses', 'Benchmark']

    diagnoses_bm_list = list(def_map_df[def_map_df.Benchmark==True].drop_duplicates('Diagnoses').Diagnoses)
    
    return diagnoses_bm_list, diagnoses
    
    
def create_diagnoses_df(ptnt_demog_df, diagnoses_bm_list, diagnoses):

    """

    Parameters
    ----------
    ptnt_demog_df: pandas DataFrame
                    patient demographic data
    diagnoses_bm_list: list
                    benchmarked diagnoses (diagnoses used in a model performance
                                                benchmarking exercise)
    diagnoses: pandas DataFrame
                    diagnoses from patient demographics table

    Returns
    -------
    diagnoses2: pandas DataFrame
                    diagnoses index by icustays

    """

    icustays = list(ptnt_demog_df.index)
    """
    prior work was done to create benchmarks for model performance. Diagnoses that were used in that
    excercise are included here
    """
    # create dataframe with hcup_ccp diagnoses benchmark categories as columns and
    # icustay_id information as indices. if the diagnosis is present for a given icustay the 
    # value is 1, otherwise 0. 

    diagnoses2 = pd.DataFrame(columns = diagnoses_bm_list, index = icustays)
    diagnoses2.fillna(0, inplace = True)
    #print "created empty diagnoses dataframe"
    for row in diagnoses.iterrows():
        if row[1]['USE_IN_BENCHMARK'] == 1:
            diagnoses2.loc[row[0]][row[1]['HCUP_CCS_2015']] = 1

    return diagnoses2
    

        



def write_best_features(dummies):
    
    frame = dummies
    X = frame[frame.columns[1:]]
    y = frame['hospital_expire_flag']

        
    # SELECT K BEST FEATURES BASED ON CHI2 SCORES
    selector = SelectKBest(score_func = chi2, k = 'all')
    selector.fit(X, y)
    p_vals = pd.Series(selector.pvalues_, name = 'p_values', index = X.columns)
    scores = pd.Series(selector.scores_, name = 'scores', index = X.columns)
    features_df = pd.concat([p_vals, scores], axis = 1)
    features_df.sort_values(by ='scores', ascending = False, inplace = True)
    print("Feature scores/p_values in descending/ascending order")
    print(features_df.head(20))

    best_features = frame[features_df[features_df.p_values < .001].index]

    frame = pd.DataFrame(y).merge(best_features, left_index = True, right_index = True, 
                    how = 'left', sort = True)


    print("head of selected feature frame ")
    print(frame.head())
    #code for writing features to file    
    root = '../data/features/'
    name = 'Ptnt_Demog_Features.csv'
    name2 = 'Ptnt_Demog_FeaturesScores.csv'
    frame.to_csv(root + name)
    features_df[features_df.p_values < .001].to_csv(root + name2)
    y = pd.DataFrame(y)
    y.to_csv(root + 'outcomes.csv')




