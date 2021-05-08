import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from gensim.models import LsiModel
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from numpy import random
import mysql
import mysql.connector
import clean_text


def choose_data_base(db_name):
    mysql_con = mysql.connector.connect(user='root', password='', host='localhost',
                                        database='{}'.format(db_name), auth_plugin='mysql_native_password',
                                        use_unicode=True)
    return mysql_con

def split_train_valid_test(data_to_split):
    data_to_split = data_to_split.sort_values(by=['time_add_to_sprint'])
    data_to_split = data_to_split.reset_index(drop=True)
    # with validation
    num_rows_train = round(0.6*len(data_to_split))
    num_rows_valid = round(0.8*len(data_to_split))

    train = data_to_split.loc[0:num_rows_train-1, :].reset_index(drop=True)
    valid = data_to_split.loc[num_rows_train:num_rows_valid-1, :].reset_index(drop=True)
    test = data_to_split.loc[num_rows_valid:, :].reset_index(drop=True)
    return train, valid, test


def run_random_forest(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=1000, max_features='sqrt', random_state=7)
    # Train the model
    clf.fit(x_train, y_train)
    feature_imp = pd.Series(clf.feature_importances_, index=list(x_train.columns.values)).sort_values(ascending=False)
    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    # Model Accuracy
    print("Accuracy RF:", accuracy)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("confusion_matrix RF: \n {}".format(confusion_matrix))
    classification_report = metrics.classification_report(y_test, y_pred)
    print("classification_report RF: \n {}".format(classification_report))
    # Create precision, recall curve
    average_precision = metrics.average_precision_score(y_test, y_score[:, 1])
    print('Average precision-recall score RF: {0:0.2f}'.format(average_precision))
    auc = metrics.roc_auc_score(y_test, y_score[:, 1])
    print('AUC roc RF: {}'.format(auc))
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score[:, 1])
    area_under_pre_recall_curve = metrics.auc(recall, precision)
    print('area_under_pre_recall_curve RF: {}'.format(area_under_pre_recall_curve))

    return [accuracy, confusion_matrix, classification_report, area_under_pre_recall_curve, average_precision, auc,
            y_pred, feature_imp, precision, recall, thresholds]


def create_gensim_lda_model(doc_clean, number_of_topics, words, dictionary, corpus):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    # generate LSA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=number_of_topics,
                                                random_state=100, update_every=1, chunksize=100, passes=10,
                                                alpha='auto', per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    return lda_model


# Finding the dominant topic in each sentence
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4),
                                                                  topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return sent_topics_df


def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, model_lsa, stop, start=2, step=3):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for number_of_topics in range(start, stop, step):
        # generate LSA model
        if model_lsa:
            model = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word=dictionary)  # train model
        else:
            model = gensim.models.ldamodel.LdaModel(corpus=doc_term_matrix, id2word=dictionary,
                                                    num_topics=number_of_topics, random_state=100, update_every=1,
                                                    chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


def plot_graph(doc_clean, dictionary, doc_term_matrix, model_lsa, start, stop, step, project_key, path):
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix, doc_clean, model_lsa,
                                                            stop, start, step)
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.title("Select Number of Topics {}".format(project_key))
    plt.legend(("coherence_values"), loc='best')
    plt.savefig('{}/topic_model/coherence_{}.png'.format(path, project_key))
    plt.ylim(0)
    plt.savefig('{}/topic_model/coherence_{}_2.png'.format(path, project_key))
    plt.close()

def create_topic_model2(data_train, project_key, path):

    start, stop, step = 2, 11, 1
    text_train_list = []
    for row in data_train['clean_text_new']:
        text_train_list.append(row)
    # Creating the term dictionary of our corpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(text_train_list)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    corpus = [dictionary.doc2bow(doc) for doc in text_train_list]

    plot_graph(text_train_list, dictionary, corpus, False, start, stop, step, project_key, path)


def create_topic_model(data_train, project_key, data_test, labels_train, labels_test,path):
    results = pd.DataFrame(columns=['project_key', 'usability_label', 'num_topics', 'feature_importance', 'accuracy_rf',
                                    'confusion_matrix_rf', 'classification_report_rf', 'area_under_pre_recall_curve_rf',
                                    'avg_precision_rf', 'area_under_roc_curve_rf', 'y_pred_rf', 'precision_rf',
                                    'recall_rf', 'thresholds_rf', 'y_test', 'features'])
    # select by result prediction:
    path = ''
    num_topic_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    words = 10
    # train
    text_train_list = []
    for row in data_train['clean_text_new']:
        text_train_list.append(row)
    # Creating the term dictionary of our corpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(text_train_list)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    corpus = [dictionary.doc2bow(doc) for doc in text_train_list]
    # test
    text_test_list = []
    for row in data_test['clean_text_new']:
        text_test_list.append(row)
    corpus_test = [dictionary.doc2bow(doc) for doc in text_test_list]
    for num in num_topic_list:
        lda_model = create_gensim_lda_model(text_train_list, num, words, dictionary, corpus)
        df_topic_sents_keywords = format_topics_sentences(lda_model, corpus, text_train_list)
        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Text']
        df_topic_sents_keywords_test = format_topics_sentences(lda_model, corpus_test, text_test_list)
        # Format
        df_dominant_topic_test = df_topic_sents_keywords_test.reset_index()
        df_dominant_topic_test.columns = ['Document_No', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords',
                                          'Text']
        df_dominant_topic = df_dominant_topic.reset_index(drop=True)
        df_dominant_topic_test = df_dominant_topic_test.reset_index(drop=True)

        # results only dominant topic

        x_train = pd.DataFrame(df_dominant_topic['Dominant_Topic'])
        x_test = pd.DataFrame(df_dominant_topic_test['Dominant_Topic'])
        # run results:
        accuracy_rf, confusion_matrix_rf, classification_report_rf, area_under_pre_recall_curve_rf, avg_pre_rf, \
            avg_auc_rf, y_pred_rf, feature_importance, precision_rf, recall_rf, \
            thresholds_rf = run_random_forest(x_train, x_test, labels_train['usability_label'],
                                              labels_test['usability_label'])

        df = {'project_key': project_key, 'usability_label': 'is_change_text_num_words_5', 'num_topics': num,
              'feature_importance': feature_importance, 'accuracy_rf': accuracy_rf,
              'confusion_matrix_rf': confusion_matrix_rf, 'classification_report_rf': classification_report_rf,
              'area_under_pre_recall_curve_rf': area_under_pre_recall_curve_rf, 'avg_precision_rf': avg_pre_rf,
              'area_under_roc_curve_rf': avg_auc_rf, 'y_pred_rf': y_pred_rf,
              'precision_rf': precision_rf, 'recall_rf': recall_rf, 'thresholds_rf': thresholds_rf,
              'y_test': labels_test['usability_label'], 'features': 'only topic model one hot'}

        results = results.append(df, ignore_index=True)

        # results dominant topic dummies

        x_train = pd.get_dummies(x_train, columns=['Dominant_Topic'], drop_first=True)
        x_test = pd.get_dummies(x_test, columns=['Dominant_Topic'], drop_first=True)
        # Get missing columns in the training test
        missing_cols = set(x_train.columns) - set(x_test.columns)
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            x_test[c] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        x_test = x_test[x_train.columns]

        # run results:
        accuracy_rf, confusion_matrix_rf, classification_report_rf, area_under_pre_recall_curve_rf, avg_pre_rf, \
            avg_auc_rf, y_pred_rf, feature_importance, precision_rf, recall_rf, \
            thresholds_rf = run_random_forest(x_train, x_test, labels_train['usability_label'],
                                              labels_test['usability_label'])

        d = {'project_key': project_key, 'usability_label': 'is_change_text_num_words_5', 'num_topics': num,
             'feature_importance': feature_importance, 'accuracy_rf': accuracy_rf,
             'confusion_matrix_rf': confusion_matrix_rf, 'classification_report_rf': classification_report_rf,
             'area_under_pre_recall_curve_rf': area_under_pre_recall_curve_rf, 'avg_precision_rf': avg_pre_rf,
             'area_under_roc_curve_rf': avg_auc_rf, 'y_pred_rf': y_pred_rf,
             'precision_rf': precision_rf, 'recall_rf': recall_rf, 'thresholds_rf': thresholds_rf,
             'y_test': labels_test['usability_label'], 'features': 'only topic model dummy'}

        results = results.append(d, ignore_index=True)

        results.to_csv(
            '{}/topic_model/results_{}_label_is_change_text_num_words_5.csv'.format(
                path,project_key),
            index=False)


if __name__ == "__main__":

    db_name_os = 'data_base_os'
    mysql_con_os = choose_data_base(db_name_os)
    path = ''

    data_developer = pd.read_sql("SELECT * FROM features_labels_table_os2 WHERE project_key='DEVELOPER'",
                                 con=mysql_con_os)
    data_repo = pd.read_sql("SELECT * FROM features_labels_table_os2 WHERE project_key='REPO'", con=mysql_con_os)
    data_dm = pd.read_sql("SELECT * FROM features_labels_table_os2 WHERE project_key='DM'", con=mysql_con_os)
    data_xd = pd.read_sql("SELECT * FROM features_labels_table_os2 WHERE project_key='XD'", con=mysql_con_os)

    data_all = [data_developer, data_repo, data_xd, data_dm]

    text_type = 'original_summary_description_acceptance_sprint'

    for data in data_all:

        project_key = data['project_key'][0]
        train, valid, test = split_train_valid_test(data)

        # ############################ clean text ######################
        clean_text.create_clean_text(train, text_type)
        clean_text.create_clean_text(valid, text_type)
        clean_text.create_clean_text(test, text_type)
        train_test = train.append(valid, ignore_index=True)

        labels_train = pd.DataFrame()
        labels_valid = pd.DataFrame()
        labels_train['usability_label'] = train['is_change_text_num_words_5']
        labels_valid['usability_label'] = valid['is_change_text_num_words_5']
        labels_train['issue_key'] = train['issue_key']
        labels_valid['issue_key'] = valid['issue_key']

        # select number of topic to each project by two options:
        create_topic_model2(train_test, project_key,path)
        create_topic_model(train, project_key, valid, labels_train,
                           labels_valid,path)
