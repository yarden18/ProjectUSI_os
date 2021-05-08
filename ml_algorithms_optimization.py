import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import signature
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import sklearn
import xgboost as xgb
from xgboost import XGBClassifier


def down_sampling1(x, y, is_down):
    # down_sampling:
    # Indicies of each class' observations

    y0_indices = [i for i in range(len(y)) if y[i] == 0]
    y1_indices = [i for i in range(len(y)) if y[i] == 1]
    # Number of observations in each class
    num_class0 = len(y0_indices)
    num_class1 = len(y1_indices)
    if is_down:
        # For every observation of class 0, randomly sample from class 1 without replacement
        np.random.seed(77)
        i_class0_downsampled = np.random.choice(y0_indices, size=num_class1, replace=False)
        # Join together class 0's target vector with the downsampled class 1's target vector
        y = y[i_class0_downsampled].append(y[y1_indices], ignore_index=True)
        x = pd.DataFrame(x)
        x = x.iloc[i_class0_downsampled].append(x.iloc[y1_indices], ignore_index=True)
    else:
        #  ######## up sampling: ##################
        # For every observation in class 0, randomly sample from class 1 with replacement
        np.random.seed(77)
        i_class1_upsampled = np.random.choice(y1_indices, size=int(num_class0), replace=True)
        # Join together class 1's upsampled target vector with class 0's target vector
        y = y[i_class1_upsampled].append(y[y0_indices], ignore_index=True)
        x = pd.DataFrame(x)
        x = x.iloc[i_class1_upsampled].append(x.iloc[y0_indices], ignore_index=True)

    return x, y


def run_random_forest(x_train, x_test, y_train, y_test, num_trees, max_feature, max_depths, min_sample_split,
                      min_sample_leaf, bootstraps):
    clf = RandomForestClassifier(n_estimators=num_trees, max_features=max_feature, max_depth=max_depths,
                                 min_samples_leaf=min_sample_leaf, min_samples_split=min_sample_split,
                                 bootstrap=bootstraps, random_state=7)
    # Train the model
    clf.fit(x_train, y_train)
    feature_imp = pd.Series(clf.feature_importances_, index=list(x_train.columns.values)).sort_values(ascending=False)
    print("feature importance RF{}".format(feature_imp))
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
    auc = metrics.roc_auc_score(y_test, y_score[:, 1], average='macro', sample_weight=None, max_fpr=None)
    print('AUC roc RF: {}'.format(auc))
    area_under_pre_recall_curve = create_pre_rec_curve(y_test, y_score, average_precision)
    print('area_under_pre_recall_curve RF: {}'.format(area_under_pre_recall_curve))

    return [accuracy, confusion_matrix, classification_report, area_under_pre_recall_curve, average_precision, auc,
            y_pred, feature_imp]


def run_neural_net(x_train, x_test, y_train, y_test, num_unit_hidden_layer, max_iteration, solvers, num_batch, activations):
    clf = MLPClassifier(solver=solvers, hidden_layer_sizes=num_unit_hidden_layer, random_state=1,
                        max_iter=max_iteration, batch_size=num_batch, activation=activations)
    # Train the model
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    # Model Accuracy
    print("Accuracy NN:", accuracy)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("confusion_matrix NN: \n {}".format(confusion_matrix))
    classification_report = metrics.classification_report(y_test, y_pred)
    print("classification_report NN: \n {}".format(classification_report))
    # Create precision, recall curve
    average_precision = metrics.average_precision_score(y_test, y_score[:, 1])
    print('Average precision-recall score NN: {0:0.2f}'.format(average_precision))
    auc = metrics.roc_auc_score(y_test, y_score[:, 1], average='macro', sample_weight=None, max_fpr=None)
    print('AUC roc NN: {}'.format(auc))
    print(8)
    area_under_pre_recall_curve = create_pre_rec_curve(y_test, y_score, average_precision)
    print('area_under_pre_recall_curve NN: {}'.format(area_under_pre_recall_curve))

    return [accuracy, confusion_matrix, classification_report, area_under_pre_recall_curve, average_precision, auc,
            y_pred]


def run_xgboost(x_train, x_test, y_train, y_test, num_trees, max_feature, max_depths, alpha_t, scale_pos,
                num_child_weight_t):
    clf = GradientBoostingClassifier(n_estimators=num_trees, max_features=max_feature, max_depth=max_depths,
                                     random_state=7)
    # Train the model
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    # Model Accuracy
    print("Accuracy xgboost:", accuracy)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("confusion_matrix xgboost: \n {}".format(confusion_matrix))
    classification_report = metrics.classification_report(y_test, y_pred)
    print("classification_report: \n {}".format(classification_report))
    # Create precision, recall curve
    average_precision = metrics.average_precision_score(y_test, y_score[:, 1])
    print('Average precision-recall score xgboost: {0:0.2f}'.format(average_precision))
    auc = metrics.roc_auc_score(y_test, y_score[:, 1], average='macro', sample_weight=None, max_fpr=None)
    print('AUC roc xgboost: {}'.format(auc))
    area_under_pre_recall_curve = create_pre_rec_curve(y_test, y_score, average_precision)
    print('area_under_pre_recall_curve xgboost: {}'.format(area_under_pre_recall_curve))

    return [accuracy, confusion_matrix, classification_report, area_under_pre_recall_curve, average_precision, auc,
            y_pred]


def run_is_empty(x_train, x_test, y_train, y_test):
    # Train the model
    y_score = pd.DataFrame()
    y_score['0'] = x_test['label_is_empty'].apply(lambda x: 1.0 if x == 0 else 0.0)
    y_score['1'] = x_test['label_is_empty'].apply(lambda x: 1.0 if x == 1 else 0.0)
    accuracy = metrics.accuracy_score(y_test, x_test['label_is_empty'])
    # Model Accuracy
    print("Accuracy xgboost:", accuracy)
    confusion_matrix = metrics.confusion_matrix(y_test, x_test['label_is_empty'])
    print("confusion_matrix xgboost: \n {}".format(confusion_matrix))
    classification_report = metrics.classification_report(y_test, x_test['label_is_empty'])
    print("classification_report: \n {}".format(classification_report))
    # Create precision, recall curve
    average_precision = metrics.average_precision_score(y_test, y_score['1'])
    print('Average precision-recall score xgboost: {0:0.2f}'.format(average_precision))
    auc = metrics.roc_auc_score(y_test, y_score['1'], average='macro', sample_weight=None, max_fpr=None)
    print('AUC roc xgboost: {}'.format(auc))
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score['1'], pos_label=1)
    area_under_pre_recall_curve = metrics.auc(recall, precision)
    print('area_under_pre_recall_curve xgboost: {}'.format(area_under_pre_recall_curve))

    return [accuracy, confusion_matrix, classification_report, area_under_pre_recall_curve, average_precision, auc,
            x_test['label_is_empty']]


def create_pre_rec_curve(y_test, y_score, average_precision):
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score[:, 1], pos_label=1)
    area = metrics.auc(recall, precision)
    print('Area Under Curve: {0:0.2f}'.format(area))
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    p;t.show()
    return area


def tuning_random_forest(x_train, y_train, project_key, label_name, all_but_one_group):
    ''''
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=50)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 4, 5, 7, 10, 15]
    # Maximum number of levels in tree
    max_depth = [int(j) for j in np.linspace(5, 50, num=10)]
    max_depth.append(None)
    # Create the random grid
    random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=4, verbose=2,
                                   random_state=12)
    # Fit the random search model
    rf_random.fit(x_train, y_train)
    best_para = rf_random.best_params_
    print("best parameters rf project {}: {}".format(project_key, best_para))'''

    if all_but_one_group:
        parameter_space = {
            'n_estimators': [int(x) for x in np.linspace(start=50, stop=2000, num=40)],
            'max_features': ['auto', 'sqrt', 4, 5, 7, 10],
            'max_depth': [int(j) for j in np.linspace(5, 50, num=10)],
        }
    else:
        parameter_space = {
            'n_estimators': [int(x) for x in np.linspace(start=50, stop=2000, num=40)],
            'max_features': ['auto', 'sqrt', 4, 5, 7, 10, 15],
            'max_depth': [int(j) for j in np.linspace(5, 50, num=10)]}

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(rf, parameter_space, cv=4, scoring='average_precision', random_state=7)
    # Fit the random search model
    rf_random.fit(x_train, y_train)
    best_para = rf_random.best_params_
    print("best parameters rf  project {} label name {} : {}".format(project_key, label_name, best_para))

    return best_para


def tuning_xgboost(x_train, y_train, project_key, label_name):
    parameter_space = {
        'n_estimators': [int(x) for x in np.linspace(start=50, stop=2000, num=40)],
        'max_depth': [int(j) for j in np.linspace(5, 50, num=10)],
        'alpha': [0, 0.0001, 0.005, 0.01, 0.05],
        'scale_pos_weight ': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                              y_train[y_train == 0].count()/float(y_train[y_train == 1].count())],
        'min_child_weight': range(1, 6, 2),
    }

    xgboost = XGBClassifier()

    xgboost_random = RandomizedSearchCV(xgboost, parameter_space, cv=4, scoring='average_precision', random_state=7)
    # Fit the random search model
    xgboost_random.fit(x_train, y_train)
    best_para = xgboost_random.best_params_
    print("best parameters xgboost  project {} label name {} : {}".format(project_key, label_name, best_para))

    return best_para


def tuning_neural_network(x_train, y_train, project_key, label_name):
    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (20,), (30,), (50,), (5, 10,), (20, 30,), (30, 50,),
                               (50, 50,), (80, 50,), (50, 50,), (20, 20, 20), (50, 30, 20), (15, 20, 30)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.005, 0.01, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    mlp = MLPClassifier(max_iter=7000)
    clf = RandomizedSearchCV(mlp, parameter_space, cv=4, scoring='average_precision', random_state=7)
    # Fit the random search model
    clf.fit(x_train, y_train)
    best_para = clf.best_params_
    print("best parameters nn  project {} label name {}: {}".format(project_key, label_name, best_para))

    return best_para


def run_model_grid(x_train, y_train, project_key, label, all_but_one_group, path):
    print(list(x_train))
    x_train = x_train.drop(columns=['created'])
    x_train = x_train.drop(columns=['issue_key'])
    features = list(x_train)
    rf_results = pd.DataFrame(columns=['project_key', 'usability_label', 'features', 'n_estimators_rf',
                                       'max_features_rf', 'max_depth_rf'])
    # tuning grid search:
    best_para_rf = tuning_random_forest(x_train, y_train, project_key, label, all_but_one_group)

    d = {'project_key': project_key, 'usability_label': label, 'features': features,
         'n_estimators_rf': best_para_rf['n_estimators'], 'max_features_rf': best_para_rf['max_features'],
         'max_depth_rf': best_para_rf['max_depth']}

    rf_results = rf_results.append(d, ignore_index=True)

    if all_but_one_group:
        rf_results.to_csv(
            '{}/optimization_results/grid_results_groups_{}_label_{}_RF.csv'.format(path,
                project_key, label), index=False)
    else:
        rf_results.to_csv(
            '{}/optimization_results/grid_results_{}_label_{}_RF.csv'.format(path,
                project_key, label), index=False)

    # XGboost:
    xgboost_results = pd.DataFrame(columns=['project_key', 'usability_label', 'features', 'n_estimators_xgboost',
                                            'max_depth_xgboost', 'alpha_xgboost',
                                            'scale_pos_weight_xgboost', 'min_child_weight_xgboost',
                                            ])
    # tuning grid search:
    best_para_xgboost = tuning_xgboost(x_train, y_train, project_key, label)

    d = {'project_key': project_key, 'usability_label': label, 'features': features,
         'n_estimators_xgboost': best_para_xgboost['n_estimators'],
         'max_depth_xgboost': best_para_xgboost['max_depth'], 'alpha_xgboost': best_para_xgboost['alpha'],
         'scale_pos_weight_xgboost': best_para_xgboost['scale_pos_weight '],
         'min_child_weight_xgboost':  best_para_xgboost['min_child_weight']}

    xgboost_results = xgboost_results.append(d, ignore_index=True)

    if all_but_one_group:
        xgboost_results.to_csv(
            '{}/optimization_results/grid_results_groups_{}_label_{}_XGboost.csv'.format(path,
                project_key, label), index=False)
    else:
        xgboost_results.to_csv(
            '{}/optimization_results/grid_results_{}_label_{}_XGboost.csv'.format(path,
                project_key, label), index=False)

    # NN:
    nn_results = pd.DataFrame(columns=['project_key', 'usability_label', 'features', 'hidden_layer_sizes_nn',
                                       'activation_nn', 'solver_nn', 'alpha_nn', 'learning_rate_nn'])

    x_train_nn = x_train
    names = list(x_train_nn.columns.values)
    x_train_nn[names] = x_train_nn[names].astype(float)
    scaler = StandardScaler()
    scaler.fit(x_train_nn)
    x_train_nn[names] = scaler.transform(x_train_nn[names])
    print("features:")
    print(x_train_nn.columns.values)
    best_para_nn = tuning_neural_network(x_train, y_train, project_key, label)

    d = {'project_key': project_key, 'usability_label': label, 'features': features,
         'hidden_layer_sizes_nn': best_para_nn['hidden_layer_sizes'],
         'activation_nn': best_para_nn['activation'], 'solver_nn': best_para_nn['solver'],
         'alpha_nn': best_para_nn['alpha'], 'learning_rate_nn': best_para_nn['learning_rate']}

    nn_results = nn_results.append(d, ignore_index=True)
    if all_but_one_group:
        nn_results.to_csv(
            '{}/optimization_results/grid_results_groups_{}_label_{}_NN.csv'.format(path,
                project_key, label), index=False)
    else:
        nn_results.to_csv(
            '{}/optimization_results/grid_results_{}_label_{}_NN.csv'.format(path,
                project_key, label), index=False)


def run_model_optimization(x_train, x_test, y_train, y_test, project_key, label, all_but_one_group, path):

    print(list(x_train))
    x_train = x_train.drop(columns=['created'])
    x_test = x_test.drop(columns=['created'])
    x_train = x_train.drop(columns=['issue_key'])
    x_test = x_test.drop(columns=['issue_key'])
    features = list(x_train)
    # down/up_sampling
    # x_train, y_train = down_sampling1(x_train, y_train, True)

    # run model:
    # RF:
    rf_results = pd.DataFrame(columns=['project_key', 'usability_label', 'features', 'feature_importance',
                                       'accuracy_rf', 'confusion_matrix_rf', 'classification_report_rf',
                                       'area_under_pre_recall_curve_rf', 'avg_precision_rf',
                                       'area_under_roc_curve_rf', 'y_pred_rf', 'num_trees', 'max_features',
                                       'max_depth', 'min_samples_split', 'min_samples_leaf', 'bootstrap'])

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depths = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depths.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstraps = [True, False]
    for num_trees in n_estimators:
        for max_feature in max_features:
            for max_depth in max_depths:
                for min_sample_split in min_samples_split:
                    for min_sample_leaf in min_samples_leaf:
                        for bootstrap in bootstraps:
                            accuracy_rf, confusion_matrix_rf, classification_report_rf, \
                                area_under_pre_recall_curve_rf, avg_pre_rf, avg_auc_rf, y_pred_rf,\
                                feature_importance = run_random_forest(x_train, x_test, y_train[
                                'usability_label'], y_test['usability_label'], num_trees, max_feature, max_depth,
                                                                   min_sample_split, min_sample_leaf, bootstrap)
                            d = {'project_key': project_key, 'usability_label': label, 'features': features,
                                 'feature_importance': feature_importance, 'accuracy_rf': accuracy_rf,
                                 'confusion_matrix_rf': confusion_matrix_rf,
                                 'classification_report_rf': classification_report_rf,
                                 'area_under_pre_recall_curve_rf': area_under_pre_recall_curve_rf,
                                 'avg_precision_rf': avg_pre_rf, 'area_under_roc_curve_rf': avg_auc_rf,
                                 'y_pred_rf': y_pred_rf, 'num_trees': num_trees, 'max_features': max_feature,
                                 'max_depth': max_depth, 'min_samples_split': min_sample_split,
                                 'min_samples_leaf': min_sample_leaf, 'bootstrap': bootstrap}

                            rf_results = rf_results.append(d, ignore_index=True)

    if all_but_one_group:
        rf_results.to_csv(
            '{}/optimization_results/results_groups_{}_label_{}_RF.csv'.format(path, project_key, label), index=False)
    else:
        rf_results.to_csv(
            '{}/optimization_results/results_{}_label_{}_RF.csv'.format(path, project_key, label), index=False)

    # XGboost:
    xgboost_results = pd.DataFrame(columns=['project_key', 'usability_label', 'accuracy_xgboost',
                                            'confusion_matrix_xgboost', 'classification_report_xgboost',
                                            'area_under_pre_recall_curve_xgboost', 'avg_precision_xgboost',
                                            'area_under_roc_curve_xgboost', 'y_pred_xgboost', 'num_trees',
                                            'max_features', 'max_depth', 'min_samples_split', 'min_samples_leaf'
                                            ])
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depths = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depths.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    for num_trees in n_estimators:
        for max_feature in max_features:
            for max_depth in max_depths:
                for min_sample_split in min_samples_split:
                    for min_sample_leaf in min_samples_leaf:
                        accuracy_xgboost, confusion_matrix_xgboost, classification_report_xgboost, \
                            area_under_pre_recall_curve_xgboost, avg_pre_xgboost, avg_auc_xgboost, \
                            y_pred_xgboost, = run_xgboost(x_train, x_test, y_train[
                                'usability_label'], y_test['usability_label'], num_trees, max_feature, max_depth,
                                                               min_sample_split, min_sample_leaf)
                        d = {'project_key': project_key, 'usability_label': label,
                             'accuracy_xgboost': accuracy_xgboost,
                             'confusion_matrix_xgboost': confusion_matrix_xgboost,
                             'classification_report_xgboost': classification_report_xgboost,
                             'area_under_pre_recall_curve_xgboost': area_under_pre_recall_curve_xgboost,
                             'avg_precision_xgboost': avg_pre_xgboost, 'area_under_roc_curve_xgboost': avg_auc_xgboost,
                             'y_pred_xgboost': y_pred_xgboost, 'num_trees': num_trees, 'max_features': max_feature,
                             'max_depth': max_depth, 'min_samples_split': min_sample_split,
                             'min_samples_leaf': min_sample_leaf}

                        xgboost_results = xgboost_results.append(d, ignore_index=True)

    if all_but_one_group:
        xgboost_results.to_csv(
            '{}/optimization_results/results_groups_{}_label_{}_XGboost.csv'.format(path, project_key, label), index=False)
    else:
        xgboost_results.to_csv(
            '{}/optimization_results/results_{}_label_{}_XGboost.csv'.format(path, project_key, label), index=False)

    # NN:
    nn_results = pd.DataFrame(columns=['project_key', 'usability_label', 'accuracy_nn',
                                       'confusion_matrix_nn', 'classification_report_nn',
                                       'area_under_pre_recall_curve_nn', 'avg_precision_nn',
                                       'area_under_roc_curve_nn', 'y_pred_nn', 'num_units_hidden_layer',
                                       'max_iterations', 'solver', 'num_batches_size', 'activation'])
    x_train_nn = x_train
    x_test_nn = x_test
    names = list(x_train_nn.columns.values)
    x_train_nn[names] = x_train_nn[names].astype(float)
    x_test_nn[names] = x_test_nn[names].astype(float)
    scaler = StandardScaler()
    scaler.fit(x_train_nn)
    x_train_nn[names] = scaler.transform(x_train_nn[names])
    x_test_nn[names] = scaler.transform(x_test_nn[names])
    print("features:")
    print(x_train_nn.columns.values)
    # hidden
    num_units_hidden_layer = [(10,), (20,), (30,), (50,), (70,), (100,), (50, 50, 50), (5, 20, 30), (30, 70, 30),
                              (100, 50, 100)]
    # Number of max iterations
    max_iterations = [200, 400, 600, 800, 1000, 1500, 2000]
    # solvers
    solvers = ['adam', 'sgd', 'lbfgs']
    num_batches_size = ['auto', 20, 30, 40, 50, 100]
    activations = ['logistic', 'tanh', 'relu']
    for num_unit_hidden_layer in num_units_hidden_layer:
        for max_iteration in max_iterations:
            for solver in solvers:
                for num_batch in num_batches_size:
                    for activation in activations:
                        accuracy_nn, confusion_matrix_nn, classification_report_nn, \
                            area_under_pre_recall_curve_nn, avg_pre_nn, avg_auc_nn, \
                            y_pred_nn = run_neural_net(x_train_nn, x_test_nn, y_train['usability_label'],
                                                       y_test['usability_label'], num_unit_hidden_layer, max_iteration,
                                                       solver, num_batch, activation)
                        d = {'project_key': project_key, 'usability_label': label,
                             'accuracy_nn': accuracy_nn, 'confusion_matrix_nn': confusion_matrix_nn,
                             'classification_report_nn': classification_report_nn,
                             'area_under_pre_recall_curve_nn': area_under_pre_recall_curve_nn,
                             'avg_precision_nn': avg_pre_nn, 'area_under_roc_curve_nn': avg_auc_nn,
                             'y_pred_nn': y_pred_nn, 'num_units_hidden_layer': num_unit_hidden_layer,
                             'max_iterations': max_iteration, 'solver': solver, 'num_batches_size': num_batch,
                             'activation': activation}

                        nn_results = nn_results.append(d, ignore_index=True)

    if all_but_one_group:
        nn_results.to_csv(
            '{}/optimization_results/results_groups_{}_label_{}_NN.csv'.format(path,
                project_key, label), index=False)
    else:
        nn_results.to_csv(
            '{}/optimization_results/results_{}_label_{}_NN.csv'.format(path,
                project_key, label), index=False)
