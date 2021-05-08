# ProjectUSI_os
The project is running in python and all the data saved in database in mySQL, so there are python code and SQL code.
SQL:
1.	create_DB - Create the DataBases
2.	create_combine_columns_summary_description - create text field of the summary and description together.
3.	Create_feature_label_table – create table with all the relevant features and some calculation of more relevant features.
4.	calculate_features_all_num_bad_issue – another calculated feature.
Python codes:
The first part is the data preparation, which includes the data extraction and the calculation of fields:
1.	Create_DB_sql – extract all the data from the open sources to the sql DB.
2.	calculate_time_add_sprint – calculate for each USI, when it was added to sprint
3.	Prepare_data_sql – some calculation and organization regarding the text fields.
4.	Add_body_clean_comments – clean the field of the comments from all kind of marks.
5.	calculate_ratio_nltk – calculate the change in the USI, count the number of words that changed, and ratio.
6.	Add_columns_empty_label – another calculated feature.

The second part is all the experiments process:
7.	Select_num_topic_model – calculation to check the best number of topic model and saving the running results.
8.	Select_length_doc_vector - calculation to check the best length of the vector and save the running results.
9.	Create_train_val_test –  calculate all the features, preparing the data and split the data to train, validation and test set (call to create_topic_model, create_doc_vec).
10.	chi_square – calculate the chi-square measure for the relevant features and save the running results.
11.	feature_selection_groups – calculate the results by all the combinations of features groups to find the best one by the train and validation set.
12.	remove_features – remove the unwanted features by the chi square and feature selection results.
13.	run_train_val_optimization – find the best parameters to each parameter by performing hyper parameter methods (call to ml_algorithms_optimization).
14.	run_train_test_best_parameters – run the results with the best parameters (call to ml_algorithms_run_best_parameters).
