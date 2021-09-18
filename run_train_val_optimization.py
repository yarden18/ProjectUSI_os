import pandas as pd
import ml_algorithms_optimization


if __name__ == "__main__":
    """
    this script read all the feature data (train and validation only), and run the optimization of the hyper parameters (ml_algorithms_optimization)
    """

    text_type = 'original_summary_description_acceptance_sprint'

    dict_labels = {'is_change_text_num_words_5': 'num_unusable_issues_cretor_prev_text_word_5_ratio',
                   'is_change_text_num_words_10': 'num_unusable_issues_cretor_prev_text_word_10_ratio',
                   'is_change_text_num_words_15': 'num_unusable_issues_cretor_prev_text_word_15_ratio',
                   'is_change_text_num_words_20': 'num_unusable_issues_cretor_prev_text_word_20_ratio'
                   }
    path = ''
    projects_key = ['DEVELOPER', 'REPO', 'XD', 'DM']
    # extract the data for each project
    for project_key in projects_key:
        for label_name in dict_labels.items():
            print("data: {}, \n label_name.key: {}, \n".format(project_key, label_name[0]))
            all_but_one_group = True
            # with all but one:
            if all_but_one_group:
                features_data_train = pd.read_csv(
                    '{}/train_val_after_all_but/features_data_train_{}_{}.csv'.format(
                        path,project_key, label_name[0]), low_memory=False)
                features_data_valid = pd.read_csv(
                    '{}/train_val_after_all_but/features_data_valid_{}_{}.csv'.format(
                        path,project_key, label_name[0]), low_memory=False)
            else:
                features_data_train = pd.read_csv(
                    '{}/train_val_after_chi/features_data_train_{}_{}.csv'.format(
                        path,project_key, label_name[0]), low_memory=False)
                features_data_valid = pd.read_csv(
                    '{}/train_val_after_chi/features_data_valid_{}_{}.csv'.format(
                        path,project_key, label_name[0]), low_memory=False)

            labels_train = pd.read_csv(
                '{}/train_val/labels_train_{}_{}.csv'.format(
                    path,project_key, label_name[0]), low_memory=False)
            labels_valid = pd.read_csv(
                '{}/train_val/labels_valid_{}_{}.csv'.format(
                    path,project_key, label_name[0]), low_memory=False)

            names = list(features_data_train.columns.values)
            if 'dominant_topic' in names:
                features_data_train = pd.get_dummies(features_data_train, columns=['dominant_topic'],
                                                     drop_first=True)
                features_data_valid = pd.get_dummies(features_data_valid, columns=['dominant_topic'],
                                                     drop_first=True)
                # Get missing columns in the training test
                missing_cols = set(features_data_train.columns) - set(features_data_valid.columns)
                # Add a missing column in test set with default value equal to 0
                for c in missing_cols:
                    features_data_valid[c] = 0
                # Ensure the order of column in the test set is in the same order than in train set
                features_data_valid = features_data_valid[features_data_train.columns]

            # get only train of the test set
            if all_but_one_group:
                features_data_train_test = pd.read_csv(
                    '{}/train_test_after_all_but/features_data_train_{}_{}.csv'.format(
                        path,project_key, label_name[0]), low_memory=False)
            else:
                features_data_train_test = pd.read_csv(
                    '{}/train_test_after_chi/features_data_train_{}_{}.csv'.format(
                        path,project_key, label_name[0]), low_memory=False)
            labels_train_test = pd.read_csv(
                '{}/train_test/labels_train_{}_{}.csv'.format(
                    path,project_key, label_name[0]), low_memory=False)

            names2 = list(features_data_train_test.columns.values)
            if 'dominant_topic' in names2:
                features_data_train_test = pd.get_dummies(features_data_train_test, columns=['dominant_topic'],
                                                          drop_first=True)

            # run optimization:
            full_optimization = False
            if full_optimization:
                ml_algorithms_optimization.run_model_optimization(features_data_train, features_data_valid,
                                                                  labels_train['usability_label'],
                                                                  labels_valid['usability_label'], project_key,
                                                                  label_name[0], all_but_one_group,run_model_grid)
            else:
                ml_algorithms_optimization.run_model_grid(features_data_train_test,
                                                          labels_train_test['usability_label'], project_key,
                                                          label_name[0], all_but_one_group,run_model_grid)
