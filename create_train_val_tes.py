import mysql
import pandas as pd
import mysql.connector
import clean_text_create_features
import create_topic_model
import create_doc_vec
import create_issue_link


def choose_data_base(db_name):
    """
    connect to SQL
    """
    mysql_con = mysql.connector.connect(user='root', password='', host='localhost',
                                        database='{}'.format(db_name), auth_plugin='mysql_native_password',
                                        use_unicode=True)
    return mysql_con


def split_train_valid_test(data_to_split):
    """
    function who get the data, split it to train, validation and test set (0.6,0.2,0.2), and return it
    """
    data_to_split = data_to_split.sort_values(by=['time_add_to_sprint'])
    data_to_split = data_to_split.reset_index(drop=True)
    # with validation
    num_rows_train = round(0.6*len(data_to_split))
    num_rows_valid = round(0.8*len(data_to_split))

    train = data_to_split.loc[0:num_rows_train-1, :].reset_index(drop=True)
    valid = data_to_split.loc[num_rows_train:num_rows_valid-1, :].reset_index(drop=True)
    test = data_to_split.loc[num_rows_valid:, :].reset_index(drop=True)
    return train, valid, test


if __name__ == "__main__":

    # connect to SQL
    db_name_os = 'data_base_os'
    mysql_con_os = choose_data_base(db_name_os)
    path = ''

    ########################################################################################################
    # ##########################################   read data:    ###########################################
    ########################################################################################################

    data_developer = pd.read_sql("SELECT * FROM features_labels_table_os2 WHERE project_key='DEVELOPER'",
                                 con=mysql_con_os)
    data_repo = pd.read_sql("SELECT * FROM features_labels_table_os2 WHERE project_key='REPO'", con=mysql_con_os)
    data_dm = pd.read_sql("SELECT * FROM features_labels_table_os2 WHERE project_key='DM'", con=mysql_con_os)
    data_xd = pd.read_sql("SELECT * FROM features_labels_table_os2 WHERE project_key='XD'", con=mysql_con_os)

    data_all = [data_developer, data_repo, data_xd, data_dm]

    ########################################################################################################
    # ######################################## data to issue links: ########################################
    ########################################################################################################

    # add the issue link data to each project by the script create_issue_link
    data_developer, data_repo, data_xd, data_dm = create_issue_link.create_issue_links_all(data_developer, data_repo,
                                                                                           data_xd, data_dm)

    text_type = 'original_summary_description_acceptance_sprint'

    dict_labels = {'is_change_text_num_words_5': 'num_unusable_issues_cretor_prev_text_word_5_ratio',
                   'is_change_text_num_words_10': 'num_unusable_issues_cretor_prev_text_word_10_ratio',
                   'is_change_text_num_words_15': 'num_unusable_issues_cretor_prev_text_word_15_ratio',
                   'is_change_text_num_words_20': 'num_unusable_issues_cretor_prev_text_word_20_ratio'}

    for data in data_all:
        # ############# create label data: ################
        project_key = data['project_key'][0]

        if project_key == 'DEVELOPER':
            num_topics = 4
            size_vec = 10
        elif project_key == 'REPO':
            num_topics = 3
            size_vec = 15
        elif project_key == 'XD':
            num_topics = 3
            size_vec = 5
        elif project_key == 'DM':
            num_topics = 4
            size_vec = 10

        # split to train validation and test set     
        train, valid, test = split_train_valid_test(data)

        # ############################ clean text and create all features ######################
        features_data_train_val = clean_text_create_features.create_feature_data(train, text_type, project_key)
        features_data_valid = clean_text_create_features.create_feature_data(valid, text_type, project_key)
        features_data_train_test = features_data_train_val.append(features_data_valid, ignore_index=True)
        features_data_test = clean_text_create_features.create_feature_data(test, text_type, project_key)
        # ########### create doc vec with the script create_doc_vec ################
        train_vec, valid_vec = create_doc_vec.create_doc_to_vec(train, valid, True, size_vec, project_key)
        features_data_train_val = pd.concat([features_data_train_val, train_vec], axis=1)
        features_data_valid = pd.concat([features_data_valid, valid_vec], axis=1)

        train_val = train.append(valid, ignore_index=True)
        train_test_vec, test_vec = create_doc_vec.create_doc_to_vec(train_val, test, True, size_vec, project_key)
        features_data_train_test = pd.concat([features_data_train_test, train_test_vec], axis=1)
        features_data_test = pd.concat([features_data_test, test_vec], axis=1)
        # ########### add topic model with the script create_topic_model ################
        # train val
        dominant_topic_train_val, dominant_topic_valid = create_topic_model.create_topic_model(train, valid, num_topics, project_key)
        dominant_topic_train_val = dominant_topic_train_val.reset_index(drop=True)
        dominant_topic_valid = dominant_topic_valid.reset_index(drop=True)
        features_data_train_val['dominant_topic'] = dominant_topic_train_val['Dominant_Topic']
        features_data_valid['dominant_topic'] = dominant_topic_valid['Dominant_Topic']

        # train test
        dominant_topic_train_test, dominant_topic_test = create_topic_model.create_topic_model(train_val, test,
                                                                                               num_topics, project_key)
        dominant_topic_train_test = dominant_topic_train_test.reset_index(drop=True)
        dominant_topic_test = dominant_topic_test.reset_index(drop=True)
        features_data_train_test['dominant_topic'] = dominant_topic_train_test['Dominant_Topic']
        features_data_test['dominant_topic'] = dominant_topic_test['Dominant_Topic']

        for label_name in dict_labels.items():
            # save label date to every set
            print("data {}: \n, \n label_name.key: {}, \n".format(project_key, label_name[0]))
            labels_train_val = pd.DataFrame()
            labels_valid = pd.DataFrame()
            labels_train_test = pd.DataFrame()
            labels_test = pd.DataFrame()
            labels_train_val['usability_label'] = train['{}'.format(label_name[0])]
            labels_valid['usability_label'] = valid['{}'.format(label_name[0])]
            labels_train_test['usability_label'] = train_val['{}'.format(label_name[0])]
            labels_test['usability_label'] = test['{}'.format(label_name[0])]
            labels_train_val['issue_key'] = train['issue_key']
            labels_valid['issue_key'] = valid['issue_key']
            labels_train_test['issue_key'] = train_val['issue_key']
            labels_test['issue_key'] = test['issue_key']
            features_data_train_val['ratio_unusable_issues_text_by_previous'] = train['{}'.format(label_name[1])]
            features_data_valid['ratio_unusable_issues_text_by_previous'] = valid['{}'.format(label_name[1])]
            features_data_train_test['ratio_unusable_issues_text_by_previous'] = train_val['{}'.format(label_name[1])]
            features_data_test['ratio_unusable_issues_text_by_previous'] = test['{}'.format(label_name[1])]

            # ################################### save the feature table  ####################################
            # train val
            features_data_train_val.to_csv(
                '{}/train_val/features_data_train_{}_{}.csv'.format(path, project_key, label_name[0]), index=False)
            features_data_valid.to_csv(
                '{}/train_val/features_data_valid_{}_{}.csv'.format(path,project_key, label_name[0]), index=False)

            labels_train_val.to_csv(
                '{}/train_val/labels_train_{}_{}.csv'.format(path,project_key, label_name[0]), index=False)
            labels_valid.to_csv(
                '{}/train_val/labels_valid_{}_{}.csv'.format(path,project_key, label_name[0]), index=False)

            # train test
            features_data_train_test.to_csv(
                '{}/train_test/features_data_train_{}_{}.csv'.format(path,project_key, label_name[0]), index=False)
            features_data_test.to_csv(
                '{}/train_test/features_data_test_{}_{}.csv'.format(path,project_key, label_name[0]), index=False)

            labels_train_test.to_csv(
                '{}/train_test/labels_train_{}_{}.csv'.format(path,project_key, label_name[0]), index=False)
            labels_test.to_csv(
                '{}/train_test/labels_test_{}_{}.csv'.format(path,project_key, label_name[0]), index=False)
