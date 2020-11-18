def group_column_names_by_data_type(self):
    self.columns_by_data_type_dict = {}
    # collect feature column names by type
    self.columns_by_data_type_dict["float64_cols"] = [index for index, val in
                                                      self.data.dtypes.iteritems() if val == 'float64']
    # uint_cols = [index for index, val in labels_and_features_dropped_null_cols_df.dtypes.iteritems() if val=='uint8']
    self.columns_by_data_type_dict["categorical_cols"] = [index for index, val in
                                                          self.data.dtypes.iteritems() if val == 'object']
    self.columns_by_data_type_dict["bool_cols"] = [index for index, val in
                                                   self.data.dtypes.iteritems() if val == 'bool']
    self.columns_by_data_type_dict["int64_cols"] = [index for index, val in
                                                    self.data.dtypes.iteritems() if val == 'int64']
