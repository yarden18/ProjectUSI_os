####################################################################
########################### open source   ##########################
####################################################################


#################################################################################
/* add calculation fields to the main table which will use us in the model */
#################################################################################



###################################################################
/* create new column which include the combination of the 
summary test field and description text field to 1 text field */
###################################################################


### calculated columns which i didn't add yet

Alter table main_table_os add column summary_description_acceptance MEDIUMTEXT;       

SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET summary_description_acceptance= 
CASE
	WHEN main_table_os.description is null then CONCAT(summary, " ", "$end$") 
	ELSE CONCAT(summary, " ", "$end$", " ", main_table_os.description) 
END;


SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET summary_description_acceptance= 
CASE
	WHEN acceptance_criteria is null then CONCAT(summary_description_acceptance, " ", "$acceptance criteria:$")
	ELSE CONCAT(summary_description_acceptance, " ", "$acceptance criteria:$", " ", acceptance_criteria) 
END;


###################################################################
/* create new column which include the combination of the original
summary test field and description text field to 1 text field */
###################################################################


Alter table main_table_os add column original_summary_description_acceptance MEDIUMTEXT;       

SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET original_summary_description_acceptance= 
CASE
	WHEN original_description is null then CONCAT(original_summary, " ", "$end$") 
	ELSE CONCAT(original_summary, " ", "$end$", " ", original_description) 
END;


SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET original_summary_description_acceptance= 
CASE
	WHEN original_acceptance_criteria is null then CONCAT(original_summary_description_acceptance, " ", "$acceptance criteria:$")
	ELSE CONCAT(original_summary_description_acceptance, " ", "$acceptance criteria:$", " ", original_acceptance_criteria) 
END;



###################################################################
/* create new column which calculated the number of changes in the 
text fields */
###################################################################

Alter table main_table_os add column num_changes_summary_description_acceptance INT(11);       

SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET num_changes_summary_description_acceptance= 
CASE
	WHEN num_changes_summary_new > 0 or num_changes_description_new > 0  or num_changes_acceptance_criteria_new > 0
    then num_changes_summary_new + num_changes_description_new + num_changes_acceptance_criteria_new
	ELSE 0
END;

###################################################################
/* create new column which indicates if there were changes in the 
text fields */
###################################################################


Alter table main_table_os add column has_changes_summary_description_acceptance INT(11);       

SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET has_changes_summary_description_acceptance= 
CASE
	WHEN num_changes_summary_description_acceptance > 0  then 1
	ELSE 0
END;


###################################################################
/* create new column which calculated the number of words in the 
text fields */
###################################################################


Alter table main_table_os add column num_different_words_all_text INT(11);       

SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET num_different_words_all_text = different_words_minus_summary + 
different_words_plus_summary + different_words_minus_description + different_words_plus_description + 
different_words_minus_acceptance_criteria + different_words_plus_acceptance_criteria;


###################################################################
/* create new column which calculate the ratio difference
in words between the first version of the text and the last */
###################################################################


Alter table main_table_os add column num_different_ratio_words_all_text FLOAT;       

SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET num_different_ratio_words_all_text = 
case
	when original_acceptance_criteria is null and original_description is null then different_words_ratio_all_summary
    when original_acceptance_criteria is null and original_description is not null then (different_words_ratio_all_summary*(LENGTH(original_summary) - LENGTH(replace(original_summary, ' ', '')))
	+ different_words_ratio_all_description*(LENGTH(original_description) - LENGTH(replace(original_description, ' ', ''))))/(
	(LENGTH(original_summary) - LENGTH(replace(original_summary, ' ', '')))+ 
	(LENGTH(original_description) - LENGTH(replace(original_description, ' ', ''))))
    when original_acceptance_criteria is not null and original_description is null then (different_words_ratio_all_summary*(LENGTH(original_summary) - LENGTH(replace(original_summary, ' ', '')))
	+ different_words_ratio_all_acceptance_criteria*(LENGTH(original_acceptance_criteria) - LENGTH(replace(original_acceptance_criteria, ' ', ''))))/(
	(LENGTH(original_summary) - LENGTH(replace(original_summary, ' ', '')))+ 
	(LENGTH(original_acceptance_criteria) - LENGTH(replace(original_acceptance_criteria, ' ', ''))))
	else (different_words_ratio_all_summary*(LENGTH(original_summary) - LENGTH(replace(original_summary, ' ', '')))
	+ different_words_ratio_all_description*(LENGTH(original_description) - LENGTH(replace(original_description, ' ', '')))
	+ different_words_ratio_all_acceptance_criteria*(LENGTH(original_acceptance_criteria) - LENGTH(replace(original_acceptance_criteria, ' ', ''))))/(
	(LENGTH(original_summary) - LENGTH(replace(original_summary, ' ', '')))+ 
	(LENGTH(original_description) - LENGTH(replace(original_description, ' ', '')))+
	(LENGTH(original_acceptance_criteria) - LENGTH(replace(original_acceptance_criteria, ' ', ''))))
    
end;



###################################################################
/* do the same for the text when enter the sprint and the last 
version */
###################################################################



# after sprint: 
### calculated columns which i didn't add yet
Alter table main_table_os add column original_summary_description_acceptance_sprint MEDIUMTEXT;       

SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET original_summary_description_acceptance_sprint= 
CASE
	WHEN original_description_sprint is null then CONCAT(original_summary_sprint, " ", "$end$") 
	ELSE CONCAT(original_summary_sprint, " ", "$end$", " ", original_description_sprint) 
END;

SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET original_summary_description_acceptance_sprint= 
CASE
	WHEN original_acceptance_criteria_sprint is null then CONCAT(original_summary_description_acceptance_sprint, " ", "$acceptance criteria:$")
	ELSE CONCAT(original_summary_description_acceptance_sprint, " ", "$acceptance criteria:$", " ", original_acceptance_criteria_sprint) 
END;


Alter table main_table_os add column num_changes_summary_description_acceptance_sprint INT(11);       

SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET num_changes_summary_description_acceptance_sprint= 
CASE
	WHEN num_changes_summary_new_sprint > 0 or num_changes_description_new_sprint > 0  or num_changes_acceptance_criteria_new_sprint > 0 
    then num_changes_summary_new_sprint + num_changes_description_new_sprint + num_changes_acceptance_criteria_new_sprint
	ELSE 0
END;

Alter table main_table_os add column has_changes_summary_description_acceptance_sprint INT(11);       

SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET has_changes_summary_description_acceptance_sprint= 
CASE
	WHEN num_changes_summary_description_acceptance_sprint > 0  then 1
	ELSE 0
END;


Alter table main_table_os add column num_different_words_all_text_sprint INT(11);       

SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET num_different_words_all_text_sprint = different_words_minus_summary_sprint + 
different_words_plus_summary_sprint + different_words_minus_description_sprint + different_words_plus_description_sprint + 
different_words_minus_acceptance_criteria_sprint + different_words_plus_acceptance_criteria_sprint;

Alter table main_table_os add column num_different_ratio_words_all_text_sprint FLOAT;       

SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET num_different_ratio_words_all_text_sprint = 
case
	when original_acceptance_criteria_sprint is null and original_description_sprint is null then different_words_ratio_all_summary_sprint
    when original_acceptance_criteria_sprint is null and original_description_sprint is not null then (different_words_ratio_all_summary_sprint*(LENGTH(original_summary_sprint) - LENGTH(replace(original_summary_sprint, ' ', '')))
	+ different_words_ratio_all_description_sprint*(LENGTH(original_description_sprint) - LENGTH(replace(original_description_sprint, ' ', ''))))/(
	(LENGTH(original_summary_sprint) - LENGTH(replace(original_summary_sprint, ' ', '')))+ 
	(LENGTH(original_description_sprint) - LENGTH(replace(original_description_sprint, ' ', ''))))
    when original_acceptance_criteria_sprint is not null and original_description_sprint is null then (different_words_ratio_all_summary_sprint*(LENGTH(original_summary_sprint) - LENGTH(replace(original_summary_sprint, ' ', '')))
	+ different_words_ratio_all_acceptance_criteria_sprint*(LENGTH(original_acceptance_criteria_sprint) - LENGTH(replace(original_acceptance_criteria_sprint, ' ', ''))))/(
	(LENGTH(original_summary_sprint) - LENGTH(replace(original_summary_sprint, ' ', '')))+ 
	(LENGTH(original_acceptance_criteria_sprint) - LENGTH(replace(original_acceptance_criteria_sprint, ' ', ''))))
	else (different_words_ratio_all_summary_sprint*(LENGTH(original_summary_sprint) - LENGTH(replace(original_summary_sprint, ' ', '')))
	+ different_words_ratio_all_description_sprint*(LENGTH(original_description_sprint) - LENGTH(replace(original_description_sprint, ' ', '')))
	+ different_words_ratio_all_acceptance_criteria_sprint*(LENGTH(original_acceptance_criteria_sprint) - LENGTH(replace(original_acceptance_criteria_sprint, ' ', ''))))/(
	(LENGTH(original_summary_sprint) - LENGTH(replace(original_summary_sprint, ' ', '')))+ 
	(LENGTH(original_description_sprint) - LENGTH(replace(original_description_sprint, ' ', '')))+
	(LENGTH(original_acceptance_criteria_sprint) - LENGTH(replace(original_acceptance_criteria_sprint, ' ', ''))))
    
end;



Alter table main_table_os add column num_changes_text_before_sprint INT;       

SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET num_changes_text_before_sprint = (num_changes_summary_description_acceptance-num_changes_summary_description_acceptance_sprint);

Alter table main_table_os add column num_changes_story_point_before_sprint INT;       

SET SQL_SAFE_UPDATES = 0;
UPDATE main_table_os SET num_changes_story_point_before_sprint = (num_changes_story_points_new-num_changes_story_points_new_sprint);




