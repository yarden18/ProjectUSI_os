# ProjectUSI_os
The project is running in python and all the data saved in database in mySQL, so there are python code and SQL code.
SQL:

These codes preparing the DB and all the tables and columns that we need.
For the SQL codes we don’t need any inputs.

1.	create_DB - Create the DataBases. This is an SQL which creates all the empty tables that we need with all the wanted columns, names and types. The output of the code is the empty tables of the DB.
2.	create_combine_columns_summary_description - create text field of the summary and description together.
3.	Add_columns_main_changes – add more columns to the main table.
4.	Delete_no_sprint_no_done – delete USI that don’t relate to spirt and haven’t done yet.
5.	Create_feature_label_table – create table with all the relevant features and some calculation of more relevant features.
6.	calculate_features_all_num_bad_issue – another calculated feature.

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





# The order for running all the experiments, data creation and preparation

First, we need to create the database and the empty tables by running this script in SQL:
create_DB.sql.
Later, we need to extract all the data from the open sources to the tables in the data base by running this script in python: Create_DB_sql.py
Now we need to prepare the data for the model, clean the data and create calculation fields, where part of it we are doing in python and part of it in SQL. 

First, for each issue we calculate when it was added to the sprint in the script calculate_time_add_sprint.py. 
Then, we prepare some of the calculations of the text fields, by regular change, change after sprint and change after status. This is calculated in python in the script Prepare_data_sql.py, and then clean the comment text field in the script Add_body_clean_comments.py.
Next, we run the scripts in SQL Add_columns_main_changes.sql and Delete_no_sprint_no_done.sql to clean the data from open issues and etc, and add some fields to the main table.
Next, we prepare the text fields combinations and the calculation of number of changes in the SQL script create_combine_columns_summary_description.sql. 
Next, we want to calculate the changes in the USI, to count how many words were changed and the ration of the change. We calculate it in the python script calculate_ratio_nltk.py.
Next, we want to create the features table which include all the relevant features to the model and to add more important calculated features. To do so, we run the script Create_feature_label_table.sql first to create the table, and then the scripts calculate_features_all_num_bad_issue.sql and Add_columns_empty_label.py to add some calculated features.

Now is the second part of the project which is the experiments process.
The first stage is to run the script Select_num_topic_model.py in order to check the best number of topic model and saving the running results. Then, we run the script Select_length_doc_vector.py to check the best length of the document vector and save the results.  
Next, we want to calculate all the needed features, prepare the data and split to train, validation and test sets. For this we run the script Create_train_val_test.py, and in this script there is also running of the scripts to create_topic_model.py and create_doc_vec.py which create to each project the topic model and the document vector. 
The next stage is to find the best features to each project. To do so, we need to run the python scripts 
chi_square.py, feature_selection_groups.py and remove_features.py. in the chi_square script we calculate the chi-square measure to the relevant features, and save the results  in excel file. To find the best combination of feature groups we run feature_selection_groups, which is also writes the results to excel file. To do so we are using only the train and validation sets. By the results in the excel we delete the unwanted features in the script remove_features.
Next, we choose the parameters of the models by optimization on the train and validation sets, to find the best parameters by performing hyper parameter methods in the script run_train_val_optimization.py. this script also run the script ml_algorithms_optimization.py and write the results in excel file.
And last, we want to run the experiment and test the results on the test set, we can do it or with all groups after removing the features by the chi-square test, or to run with the best feature groups. If the variable:  all_but_one_group = True then it’s by the best feature groups, and in it False it with all the feature groups. To run the experiment, we use the script run_train_test_best_parameters.py which also run inside the script ml_algorithms_run_best_parameters.py, and save the results in excel file.

For more information about each script, look at the comments in the script files. 

