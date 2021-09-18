import pandas as pd
import sklearn
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":

    results = pd.DataFrame(columns=['project_key', 'usability_label', 'feature_name', 'chi_square', 'p_value',
                                    'rfe_support', 'rfe_ranking'])
    path = ''
    model = RandomForestClassifier()
    text_type = 'original_summary_description_acceptance_sprint'

    dict_labels = {'is_change_text_num_words_5': 'num_unusable_issues_cretor_prev_text_word_5_ratio',
                   'is_change_text_num_words_10': 'num_unusable_issues_cretor_prev_text_word_10_ratio',
                   'is_change_text_num_words_15': 'num_unusable_issues_cretor_prev_text_word_15_ratio',
                   'is_change_text_num_words_20': 'num_unusable_issues_cretor_prev_text_word_20_ratio'}

    projects_key = ['DEVELOPER', 'REPO', 'XD', 'DM']

    for project_key in projects_key:
        # run for all the 4 projects 
        for label_name in dict_labels.items():
            # for each label type (5,10,15,20)
            print("data: {}, \n label_name.key: {}, \n".format(project_key, label_name[0]))
            # extract features
            features_data_train = pd.read_csv(
                '{}/train_test/features_data_train_{}_{}.csv'.format(path,project_key, label_name[0]), low_memory=False)
            labels_train = pd.read_csv(
                '{}/train_test/labels_train_{}_{}.csv'.format(path, project_key, label_name[0]), low_memory=False)
            # delete unrelevant features
            del features_data_train['issue_key']
            del features_data_train['created']
            del features_data_train['original_story_points_sprint']
            del features_data_train['num_headlines']
            del features_data_train['num_question_marks']
            del features_data_train['num_sentences']
            del features_data_train['len_sum_desc']
            del features_data_train['num_words']
            del features_data_train['avg_word_len']
            del features_data_train['avg_num_word_in_sentence']
            del features_data_train['len_description']
            if project_key == 'XD':
                del features_data_train['len_acceptance']
            del features_data_train['num_stopwords']
            del features_data_train['num_issues_cretor_prev']
            del features_data_train['num_changes_text_before_sprint']
            del features_data_train['ratio_unusable_issues_text_by_previous']
            del features_data_train['num_comments_before_sprint']
            del features_data_train['num_changes_story_point_before_sprint']
            del features_data_train['time_until_add_sprint']
            del features_data_train['noun_count']
            del features_data_train['verb_count']
            del features_data_train['adj_count']
            del features_data_train['adv_count']
            del features_data_train['pron_count']
            del features_data_train['block']
            del features_data_train['block_by']
            del features_data_train['duplicate']
            del features_data_train['relates']
            del features_data_train['duplicate_by']
            if project_key == 'DEVELOPER' or project_key == 'DM':
                features_data_train.drop(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], axis=1, inplace=True)
            if project_key == 'REPO':
                features_data_train.drop(
                    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'], axis=1,
                    inplace=True)
            if project_key == 'XD':
                features_data_train.drop(['0', '1', '2', '3', '4'], axis=1, inplace=True)

            print("data {}: \n, \n label_name.key: {}, \n".format(project_key, label_name[0]))
            names = list(features_data_train.columns.values)
            print(names)
            # calculate the chi-square    
            rfe = RFE(model, 5)
            rfe = rfe.fit(features_data_train, labels_train['usability_label'])
            chi_square = sklearn.feature_selection.chi2(features_data_train, labels_train['usability_label'])
            for i in range(0, len(names)):
                print(names[i])
                print("chi_square: {}".format(chi_square[0][i]))
                print("chi_square p value: {}".format(chi_square[1][i]))
                print("rfe.support_: {}".format(rfe.support_[i]))
                print("rfe.ranking: {}".format(rfe.ranking_[i]))

                d = {'project_key': project_key, 'usability_label': label_name[0], 'feature_name': names[i],
                     'chi_square': chi_square[0][i], 'p_value': chi_square[1][i],
                     'rfe_support': rfe.support_[i], 'rfe_ranking': rfe.ranking_[i]}

                results = results.append(d, ignore_index=True)

        # write the results to excel
        results.to_csv('{}/chi_square/chi_square_{}.csv'.format(path,project_key),index=False)
        print("project key done: {}".format(project_key))

        results = pd.DataFrame(columns=['project_key', 'usability_label', 'feature_name', 'chi_square', 'p_value',
                                        'rfe_support', 'rfe_ranking'])
