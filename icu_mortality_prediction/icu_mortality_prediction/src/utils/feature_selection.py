"""
Selects features for supervised and unsupervised learning

TODOs:
All of it.
"""

__author__ = 'rjb'


import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

def calculate_mutual_info_between_features_and_labels(features_df, labels_df):
    """
   Calculates mutual information between features and targets
    Parameters
    ----------
    features_df: pandas dataframe
                features
    labels_df: pandas dataframe
              targets

    Returns
    ----------
    mutual_info_series: pandas series
            series with feature names as index and with mutual info
            between feature and targets as values
    TODO: add exception handling
    """

    mutual_info_values = mutual_info_classif(features_df, labels_df)
    mutual_info_series = pd.Series(mutual_info_values, features_df.columns)
    return mutual_info_series

def filter_features_by_mutual_info_score(mutual_info_series, mutual_info_threshold):
    """
   Filters features by mutual information score
    Parameters
    ----------
    mutual_info_series: pandas series
            series with feature names as index and with mutual info
            between feature and targets as values
    mutual_info_threshold: float64
              threshold below which features will be discarded

    Returns
    ----------
    filtered_mutual_info_series: pandas series
            series with feature names as index and with mutual info
            between feature and targets as values. includes only
            features with mutual info >= threshold

    """

    filtered_mutual_info_series = mutual_info_series[mutual_info_series >= mutual_info_threshold]

    return filtered_mutual_info_series

def select_top_features_by_mutual_info_threshold(mutual_info_feature_scores, mutual_info_threshold):
    """
   Selects top features by mutual information score
    Parameters
    ----------
    mutual_info_series: pandas series
            series with feature names as index and with mutual info
            between feature and targets as values
    number_of_top_features: int
             number of top features to select

    Returns
    ----------
    top_mutual_info_feature_scores: pandas series
            series with feature names as index and with mutual info
            between feature and targets as values. includes top n number of features

    """

    top_mutual_info_feature_scores = mutual_info_feature_scores[
                                            mutual_info_feature_scores > mutual_info_threshold]

    return top_mutual_info_feature_scores

def select_top_n_features_by_mutual_info_score(mutual_info_feature_scores, number_of_top_features):
    """
   Selects top features by mutual information score
    Parameters
    ----------
    mutual_info_series: pandas series
            series with feature names as index and with mutual info
            between feature and targets as values
    number_of_top_features: int
             number of top features to select

    Returns
    ----------
    top_mutual_info_feature_scores: pandas series
            series with feature names as index and with mutual info
            between feature and targets as values. includes top n number of features

    """

    top_mutual_info_feature_scores = mutual_info_feature_scores.sort_values(
                                    ascending=False)[:number_of_top_features]

    return top_mutual_info_feature_scores

def select_k_best_features(data_df, feature_selection_parameters_dict):
    """
    Selects best features given features, labels and methods
    Selection can be for a given number of features or for features with
    A threshold p-value
    Assumes labels in column 0 and features in remaining columns
    Parameters
    ----------
    data_df: pandas dataframe
            data labels in first columm and features in remaining cols
    selection_parameters_dict: dict
            parameters include
            'k': 'all',
            'p_val': 0.1,
            'scorer': chi2

    Returns
    ----------
    best_features_data_df: pandas dataframe
            data labels in first columm and features in remaining cols
    """
    #TODO: add scorer options
    feature_columns = data_df.columns[1:]
    label_columns = data_df.columns[:1]


    if feature_selection_parameters_dict['scorer']=='chi2':
        scorer=chi2
    else:
        print("please indicate 'chi2' as scorer")
        return

    selector = SelectKBest(scorer,
                           feature_selection_parameters_dict['k'])
    selector.fit(data_df[feature_columns], data_df[label_columns])
    k_best_features_series = create_feature_selection_pvalues_series(selector, feature_columns)
    k_best_features_series = k_best_features_series.sort_values()

    print("Selecting {} best features \n".format(
        feature_selection_parameters_dict['k'],))

    return k_best_features_series


def select_best_features_by_p_value(data_df, feature_selection_parameters_dict):
    """
    Selects best features given features, labels and methods
    Selection is for features with p-values below a specified limit

    Assumes labels in column 0 and features in remaining columns
    Parameters
    ----------
    data_df: pandas dataframe
            data labels in first columm and features in remaining cols
    selection_parameters_dict: dict
            parameters include
            'k': 'all',
            'p_val': 0.1,
            'scorer': chi2

    Returns
    ----------
    best_features_data_df: pandas dataframe
            data labels in first columm and features in remaining cols
    """
    #TODO: add scorer options
    feature_columns = data_df.columns[1:]
    label_columns = data_df.columns[:1]


    if feature_selection_parameters_dict['scorer']=='chi2':
        scorer=chi2
    else:
        print("please indicate 'chi2' as scorer")
        return

    selector = SelectKBest(scorer,
                           'all')
    selector.fit(data_df[feature_columns], data_df[label_columns])
    best_features_by_p_val_series = create_feature_selection_pvalues_series(selector, feature_columns)
    best_features_by_p_val_series = best_features_by_p_val_series.sort_values()

    print("Selecting features with p-values < {}\n".format(
        feature_selection_parameters_dict['p_val']))
    best_features_by_p_val_series = filter_best_features_by_pvals(best_features_by_p_val_series,
                                                                  feature_selection_parameters_dict)

    return best_features_by_p_val_series



# def select_k_best_features_2(cohort_data, feature_selection_parameters_dict):
#     """
#     Selects best features given features, labels and methods
#     Selection can be for a given number of features or for features with
#     A threshold p-value
#     Parameters
#     ----------
#     data_df: pandas dataframe
#             data labels in first columm and features in remaining cols
#     selection_parameters_dict: dict
#             parameters include
#             'k': 'all',
#             'p_val': 0.1,
#             'scorer': chi2
#
#     Returns
#     ----------
#     best_features_data_df: pandas dataframe
#             data labels in first columm and features in remaining cols
#     """
#     #TODO: refactor with data object
#     feature_columns = cohort_data.get_feature_columns()
#     if feature_selection_parameters_dict['k']=='all':
#         print("Selecting features with p-values < {}\n".format(
#             feature_selection_parameters_dict['p_val']))
#     else:
#         print("Selecting {} best features \n".format(
#             feature_selection_parameters_dict['k'],))
#     selector = SelectKBest(feature_selection_parameters_dict['scorer'],
#                            feature_selection_parameters_dict['k'])
#     selector.fit(cohort_data.get_features(), cohort_data.get_classification_labels())
#     k_best_features_series = create_feature_selection_pvalues_series(selector, feature_columns)
#     k_best_features_series = k_best_features_series.sort_values()
#
#     return k_best_features_series


def filter_best_features_by_pvals(k_best_features_series, feature_selection_parameters_dict):


        print("Filtering features to include only those with p-val < {}.".format(
            feature_selection_parameters_dict['p_val']))
        best_features_by_pval_series = k_best_features_series[
                                    k_best_features_series<feature_selection_parameters_dict['p_val']]
        print("resulting in {} features".format(best_features_by_pval_series.shape[0]))
        return best_features_by_pval_series


def create_feature_selection_pvalues_series(selector, feature_columns):

    mask = selector.get_support()
    # creates sorted series of best feature names and p-values
    #TODO: add scorer to name
    best_pvals_series = pd.Series(selector.pvalues_[mask], feature_columns[mask], name="select_best p-values")

    return best_pvals_series

# def select_k_best_features(X, y, select_split_dict_params):  # p_val = 0.05, k = 'all'):
#
#     # selects features that are significantly correlated with targets
#     # as determined by chi2 test
#
#     # calculates n best chi2 scores and p-values for each feature X
#     p_val = select_split_dict_params['p_val']
#     k = select_split_dict_params['k']
#     scorer = select_split_dict_params['scorer']
#     if k != 'all':
#         selector = SelectKBest(scorer, k).fit(X, y)
#     else:
#         selector = SelectKBest(scorer).fit(X, y)
#     # gets column names of n best features and creates dataframe
#     mask = selector.get_support()
#     best_features = X.columns[mask]
#
#     # creates sorted series of best feature names and p-values
#     X_best_pvals = pd.Series(selector.pvalues_[mask], X.columns[mask], name="chi2 p-values")
#     X_best_pvals = X_best_pvals[X_best_pvals < p_val].sort_values()
#
#     # select features with significance p <= 0.05
#     best_features = list(X_best_pvals.index)
#     X_best = X[best_features]
#     # X_best = selector.transform(X)
#
#     # print(X_best.shape)
#     #     print("Features with chi2 p-values <= 0.05:\n")
#     #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     #         display(X_best_pvals[best_features].sort_values())
#     return X_best, X_best_pvals


def main():
    #do some stuff
    # # Get input file
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data-source", help="data source (apollo, ssathi)", type=str)
    # parser.add_argument("--csv", help="output directory name", type=str)
    # parser.add_argument("-s", "--save-to-db", help="save population analysis to MongoDB", action="store_true")
    # parser.add_argument("-p", "--profile", help="obtain rule-counts", action="store_true")
    # parser.add_argument("-r", "--generate-report", help="generate PDF report", action="store_true")

    # parser.add_argument("--inspect-data", help="save outputs of data inspection", action="store_true")
    # args = parser.parse_args()
    print("Coming Soon......")

if __name__ == "__main__":
    main()