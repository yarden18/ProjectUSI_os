# open sources


#################################################################################################
/* create new table in the database which include all the relevant features for the model */
#################################################################################################


CREATE TABLE features_labels_table_os2 AS SELECT issue_key, issue_type, project_key, created, epic_link, 
has_change_story_point_sprint, summary, features_labels_table_os.description, acceptance_criteria, summary_description_acceptance,
original_story_points_sprint, creator, reporter, priority, 
num_all_changes, story_point, num_bugs_issue_link, num_comments, num_issue_links, num_sprints, 
num_changes_story_points_new, num_changes_summary_description_acceptance, 
num_sub_tasks, num_changes_sprint, num_changes_story_points_new_sprint, 
num_comments_before_sprint, num_comments_after_sprint, num_issues_cretor_prev,
num_changes_text_before_sprint, num_changes_story_point_before_sprint, time_add_to_sprint
FROM features_labels_table_os;


# columns to add and calculate in python: 

Alter table features_labels_table_os2 add column original_summary_sprint MEDIUMTEXT;       
Alter table features_labels_table_os2 add column original_description_sprint MEDIUMTEXT;       
Alter table features_labels_table_os2 add column original_acceptance_criteria_sprint MEDIUMTEXT;  

Alter table features_labels_table_os2 add column num_changes_summary_sprint INT(10);
Alter table features_labels_table_os2 add column num_changes_description_sprint INT(10);
Alter table features_labels_table_os2 add column num_changes_acceptance_criteria_sprint INT(10); 

Alter table features_labels_table_os2 add column num_different_words_all_text_sprint INT(11);       
Alter table features_labels_table_os2 add column num_ratio_words_all_text_sprint float; 

############################ after the python code:############################################

########################################################################################################
/* calculted fields in the feature table (combination of the text fields, number of changes and more) */
########################################################################################################


Alter table features_labels_table_os2 add column original_summary_description_acceptance_sprint MEDIUMTEXT;       

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os2 SET original_summary_description_acceptance_sprint= 
CASE
	WHEN original_description_sprint is null then CONCAT(original_summary_sprint, " ", "$end$") 
	ELSE CONCAT(original_summary_sprint, " ", "$end$", " ", original_description_sprint) 
END;

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os2 SET original_summary_description_acceptance_sprint= 
CASE
	WHEN original_acceptance_criteria_sprint is null then CONCAT(original_summary_description_acceptance_sprint, " ", "$acceptance criteria:$")
	ELSE CONCAT(original_summary_description_acceptance_sprint, " ", "$acceptance criteria:$", " ", original_acceptance_criteria_sprint) 
END;


Alter table features_labels_table_os2 add column num_changes_summary_description_acceptance_sprint INT(11);       

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os2 SET num_changes_summary_description_acceptance_sprint= 
CASE
	WHEN num_changes_summary_sprint > 0 or num_changes_description_sprint > 0  or num_changes_acceptance_criteria_sprint > 0 
    then num_changes_summary_sprint + num_changes_description_sprint + num_changes_acceptance_criteria_sprint
	ELSE 0
END;



# change more than 1
Alter table features_labels_table_os2 add column is_change_text_num_words_1 int(11);       

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os2 SET is_change_text_num_words_1= 
CASE
	WHEN num_different_words_all_text_sprint >0 and num_changes_summary_description_acceptance_sprint>0 then 1
	ELSE 0
END;


# change more than 5
Alter table features_labels_table_os2 add column is_change_text_num_words_5 int(11);       

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os2 SET is_change_text_num_words_5= 
CASE
	WHEN num_different_words_all_text_sprint >= 5 and num_changes_summary_description_acceptance_sprint>0 then 1
	ELSE 0
END;

# change more than 10
Alter table features_labels_table_os2 add column is_change_text_num_words_10 int(11);       

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os2 SET is_change_text_num_words_10= 
CASE
	WHEN num_different_words_all_text_sprint >= 10 and num_changes_summary_description_acceptance_sprint>0 then 1
	ELSE 0
END;

# change more than 15
Alter table features_labels_table_os2 add column is_change_text_num_words_15 int(11);       

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os2 SET is_change_text_num_words_15= 
CASE
	WHEN num_different_words_all_text_sprint >= 15 and num_changes_summary_description_acceptance_sprint>0 then 1
	ELSE 0
END;

# change more than 20
Alter table features_labels_table_os2 add column is_change_text_num_words_20 int(11);       

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os2 SET is_change_text_num_words_20= 
CASE
	WHEN num_different_words_all_text_sprint >= 20 and num_changes_summary_description_acceptance_sprint>0 then 1
	ELSE 0
END;


# combined label - change in story point or change in text or more than one sprint:
Alter table features_labels_table_os2 add column is_change_text_sp_sprint int(11);       

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os2 SET is_change_text_sp_sprint= 
CASE
	WHEN is_change_text_num_words_5 > 0 or num_changes_story_point_after_sprint > 0 or num_sprints > 1 then 1
	ELSE 0
END;




# time until add to sprint
Alter table features_labels_table_os2 add column time_until_add_to_sprint float;       

SET SQL_SAFE_UPDATES = 0;


UPDATE features_labels_table_os2 SET time_until_add_to_sprint= TIMESTAMPDIFF(minute, created, time_add_to_sprint)/60;

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os2 SET time_until_add_to_sprint= 
CASE
	WHEN time_until_add_to_sprint is null then 0
	ELSE time_until_add_to_sprint
END;



