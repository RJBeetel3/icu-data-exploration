
import pandas as pd



def group_column_names_by_data_type(data_df):
    """
     Groups columns by data type and returns each type in a list


        Parameters
        ----------
        jdata_df: set
                    set of unique id's

        Returns
        -------
        columns_by_data_type_dict: dictionary
                    dictionary with data type as key and list of corresponding column names as values
                    dictionary with data type as key and list of corresponding column names as values
    """
    columns_by_data_type_dict = {}
    # collect feature column names by type
    columns_by_data_type_dict["float64_cols"] = [index for index, val in
                                                 data_df.dtypes.iteritems() if val == 'float64']
    # uint_cols = [index for index, val in labels_and_features_dropped_null_cols_df.dtypes.iteritems() if val=='uint8']
    columns_by_data_type_dict["categorical_cols"] = [index for index, val in
                                                     data_df.dtypes.iteritems() if val == 'object']
    columns_by_data_type_dict["bool_cols"] = [index for index, val in
                                              data_df.dtypes.iteritems() if val == 'bool']
    columns_by_data_type_dict["int64_cols"] = [index for index, val in
                                                data_df.dtypes.iteritems() if val == 'int64']
    columns_by_data_type_dict["datetime64[ns]"] = [index for index, val in
                                                data_df.dtypes.iteritems() if val == 'datetime64[ns]']

    return columns_by_data_type_dict

def continuous_to_categorical(ptnt_demog_data2_df, columns_by_data_type_dict):
    demog_stats = ptnt_demog_data2_df[columns_by_data_type_dict['float64_cols']].dropna().describe()
    print(demog_stats)
    for col in ptnt_demog_data2_df[columns_by_data_type_dict['float64_cols']]:
        Q1 = demog_stats[col].loc['25%']
        Q2 = demog_stats[col].loc['50%']
        Q3 = demog_stats[col].loc['75%']
        ptnt_demog_data2_df[col] = ptnt_demog_data2_df[col].apply(lambda x: quant_cats(x, Q1, Q2, Q3))

    return ptnt_demog_data2_df


def one_hot_encode_categorical_features(data_df, feature_columns):
    data_df = pd.get_dummies(data_df, columns=feature_columns)

    return data_df


def cast_dates_and_times(data_df, date_time_cols):

    for col in date_time_cols:
        print("converting {}".format(col))
        new_series = pd.to_datetime(data_df.loc[:, col])
        data_df.loc[:, col] = new_series

    return data_df


def quant_cats(feature, Q1, Q2, Q3):
    if feature <= Q1:
        return 'Q0'
    elif (feature > Q1 and feature <= Q2):
        return 'Q1'
    elif (feature > Q2 and feature <= Q3):
        return 'Q2'
    elif feature > Q3:
        return 'Q3'


def select_best_features(data_df, feature_columns, outcome_column):
    frame = data_df
    X = frame[feature_columns]
    y = frame[outcome_column]

    # SELECT K BEST FEATURES BASED ON CHI2 SCORES
    selector = SelectKBest(score_func=chi2, k='all')
    selector.fit(X, y)
    p_vals = pd.Series(selector.pvalues_, name='p_values', index=X.columns)
    scores = pd.Series(selector.scores_, name='scores', index=X.columns)
    features_df = pd.concat([p_vals, scores], axis=1)
    features_df.sort_values(by='scores', ascending=False, inplace=True)
    print("Feature scores/p_values in descending/ascending order")
    print(features_df.head(20))

    best_features = frame[features_df[features_df.p_values < .001].index]

    frame = pd.DataFrame(y).merge(best_features, left_index=True, right_index=True,
                                  how='left', sort=True)

    print("head of selected feature frame ")
    print(frame.head())
    # code for writing features to file
    root = '../data/features/'
    name = 'Ptnt_Demog_Features.csv'
    name2 = 'Ptnt_Demog_FeaturesScores.csv'
    frame.to_csv(root + name)
    features_df[features_df.p_values < .001].to_csv(root + name2)
    y = pd.DataFrame(y)
    y.to_csv(root + 'outcomes.csv')
