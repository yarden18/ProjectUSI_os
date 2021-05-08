###################################################################
########################### open source ###########################
###################################################################


####################### add table and than combine to feature table #######################


### num issues
 create table help_cal_features select issue_key, created as created2, creator as creator2, (select count(*) from features_labels_table_os m
									WHERE (f.creator = m.creator and f.project_key = m.project_key 
                                    and f.created > m.created)) as num_issues
from features_labels_table_os f;

## add the wanted column to feature table:

# First add Age column in table1

ALTER TABLE features_labels_table_os ADD COLUMN num_issues_cretor_prev INT(11);
# then update that column using blow query

UPDATE features_labels_table_os t1
INNER JOIN help_cal_features t2 ON t1.issue_key = t2.issue_key 
SET t1.num_issues_cretor_prev = t2.num_issues;

drop table help_cal_features;



#### num words: 

# calculate num unusable issue text words 1

create table help_cal_features_bad_text select issue_key, created as created2, creator as creator2, (select count(*) from features_labels_table_os m
									WHERE (f.creator = m.creator and f.project_key = m.project_key 
                                    and f.created > m.created and is_change_text_num_words_1 > 0 )) as num_unusable_text
from features_labels_table_os f;

## add the wanted column to feature table:

# First add Age column in table1

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_word_1 INT(11);
# then update that column using blow query

UPDATE features_labels_table_os t1
INNER JOIN help_cal_features_bad_text t2 ON t1.issue_key = t2.issue_key 
SET t1.num_unusable_issues_cretor_prev_text_word_1 = t2.num_unusable_text;

drop table help_cal_features_bad_text;


# calculate ratio bas issues text words 1: 

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_word_1_ratio float;

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os SET num_unusable_issues_cretor_prev_text_word_1_ratio= 
CASE
	WHEN num_unusable_issues_cretor_prev_text_word_1 = 0 then 0  
	ELSE num_unusable_issues_cretor_prev_text_word_1/num_issues_cretor_prev  
END;



# calculate num unusable issue text words 5

create table help_cal_features_bad_text select issue_key, created as created2, creator as creator2, (select count(*) from features_labels_table_os m
									WHERE (f.creator = m.creator and f.project_key = m.project_key 
                                    and f.created > m.created and is_change_text_num_words_5 > 0 )) as num_unusable_text
from features_labels_table_os f;

## add the wanted column to feature table:

# First add Age column in table1

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_word_5 INT(11);
# then update that column using blow query

UPDATE features_labels_table_os t1
INNER JOIN help_cal_features_bad_text t2 ON t1.issue_key = t2.issue_key 
SET t1.num_unusable_issues_cretor_prev_text_word_5 = t2.num_unusable_text;

drop table help_cal_features_bad_text;


# calculate ratio bas issues text words 5: 

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_word_5_ratio float;

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os SET num_unusable_issues_cretor_prev_text_word_5_ratio= 
CASE
	WHEN num_unusable_issues_cretor_prev_text_word_5 = 0 then 0  
	ELSE num_unusable_issues_cretor_prev_text_word_5/num_issues_cretor_prev  
END;



# calculate num unusable issue text words 10

create table help_cal_features_bad_text select issue_key, created as created2, creator as creator2, (select count(*) from features_labels_table_os m
									WHERE (f.creator = m.creator and f.project_key = m.project_key 
                                    and f.created > m.created and is_change_text_num_words_10 > 0 )) as num_unusable_text
from features_labels_table_os f;

## add the wanted column to feature table:

# First add Age column in table1

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_word_10 INT(11);
# then update that column using blow query

UPDATE features_labels_table_os t1
INNER JOIN help_cal_features_bad_text t2 ON t1.issue_key = t2.issue_key 
SET t1.num_unusable_issues_cretor_prev_text_word_10 = t2.num_unusable_text;

drop table help_cal_features_bad_text;


# calculate ratio bas issues text words 10: 

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_word_10_ratio float;

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os SET num_unusable_issues_cretor_prev_text_word_10_ratio= 
CASE
	WHEN num_unusable_issues_cretor_prev_text_word_10 = 0 then 0  
	ELSE num_unusable_issues_cretor_prev_text_word_10/num_issues_cretor_prev  
END;




# calculate num unusable issue text words 15

create table help_cal_features_bad_text select issue_key, created as created2, creator as creator2, (select count(*) from features_labels_table_os m
									WHERE (f.creator = m.creator and f.project_key = m.project_key 
                                    and f.created > m.created and is_change_text_num_words_15 > 0 )) as num_unusable_text
from features_labels_table_os f;

## add the wanted column to feature table:

# First add Age column in table1

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_word_15 INT(11);
# then update that column using blow query

UPDATE features_labels_table_os t1
INNER JOIN help_cal_features_bad_text t2 ON t1.issue_key = t2.issue_key 
SET t1.num_unusable_issues_cretor_prev_text_word_15 = t2.num_unusable_text;

drop table help_cal_features_bad_text;


# calculate ratio bas issues text words 15: 

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_word_15_ratio float;

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os SET num_unusable_issues_cretor_prev_text_word_15_ratio= 
CASE
	WHEN num_unusable_issues_cretor_prev_text_word_15 = 0 then 0  
	ELSE num_unusable_issues_cretor_prev_text_word_15/num_issues_cretor_prev  
END;



# calculate num unusable issue text words 20

create table help_cal_features_bad_text select issue_key, created as created2, creator as creator2, (select count(*) from features_labels_table_os m
									WHERE (f.creator = m.creator and f.project_key = m.project_key 
                                    and f.created > m.created and is_change_text_num_words_20 > 0 )) as num_unusable_text
from features_labels_table_os f;

## add the wanted column to feature table:

# First add Age column in table1

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_word_20 INT(11);
# then update that column using blow query

UPDATE features_labels_table_os t1
INNER JOIN help_cal_features_bad_text t2 ON t1.issue_key = t2.issue_key 
SET t1.num_unusable_issues_cretor_prev_text_word_20 = t2.num_unusable_text;

drop table help_cal_features_bad_text;


# calculate ratio bas issues text words 20: 

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_word_20_ratio float;

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os SET num_unusable_issues_cretor_prev_text_word_20_ratio= 
CASE
	WHEN num_unusable_issues_cretor_prev_text_word_20 = 0 then 0  
	ELSE num_unusable_issues_cretor_prev_text_word_20/num_issues_cretor_prev  
END;





### ratio words:

# calculate num unusable issue text ratio 005

create table help_cal_features_bad_text select issue_key, created as created2, creator as creator2, (select count(*) from features_labels_table_os m
									WHERE (f.creator = m.creator and f.project_key = m.project_key 
                                    and f.created > m.created and is_change_text_ratio_words_005 > 0 )) as num_unusable_text
from features_labels_table_os f;

## add the wanted column to feature table:

# First add Age column in table1

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_ratio_005 INT(11);
# then update that column using blow query

UPDATE features_labels_table_os t1
INNER JOIN help_cal_features_bad_text t2 ON t1.issue_key = t2.issue_key 
SET t1.num_unusable_issues_cretor_prev_text_ratio_005 = t2.num_unusable_text;

drop table help_cal_features_bad_text;


# calculate ratio bas issues text ratio 005: 

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_ratio_005_ratio float;

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os SET num_unusable_issues_cretor_prev_text_ratio_005_ratio= 
CASE
	WHEN num_unusable_issues_cretor_prev_text_ratio_005 = 0 then 0  
	ELSE num_unusable_issues_cretor_prev_text_ratio_005/num_issues_cretor_prev  
END;



# calculate num unusable issue text ratio 01

create table help_cal_features_bad_text select issue_key, created as created2, creator as creator2, (select count(*) from features_labels_table_os m
									WHERE (f.creator = m.creator and f.project_key = m.project_key 
                                    and f.created > m.created and is_change_text_ratio_words_01 > 0 )) as num_unusable_text
from features_labels_table_os f;

## add the wanted column to feature table:

# First add Age column in table1

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_ratio_01 INT(11);
# then update that column using blow query

UPDATE features_labels_table_os t1
INNER JOIN help_cal_features_bad_text t2 ON t1.issue_key = t2.issue_key 
SET t1.num_unusable_issues_cretor_prev_text_ratio_01 = t2.num_unusable_text;

drop table help_cal_features_bad_text;


# calculate ratio bas issues text ratio 01: 

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_ratio_01_ratio float;

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os SET num_unusable_issues_cretor_prev_text_ratio_01_ratio= 
CASE
	WHEN num_unusable_issues_cretor_prev_text_ratio_01 = 0 then 0  
	ELSE num_unusable_issues_cretor_prev_text_ratio_01/num_issues_cretor_prev  
END;



# calculate num unusable issue text ratio 025

create table help_cal_features_bad_text select issue_key, created as created2, creator as creator2, (select count(*) from features_labels_table_os m
									WHERE (f.creator = m.creator and f.project_key = m.project_key 
                                    and f.created > m.created and is_change_text_ratio_words_025 > 0 )) as num_unusable_text
from features_labels_table_os f;

## add the wanted column to feature table:

# First add Age column in table1

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_ratio_025 INT(11);
# then update that column using blow query

UPDATE features_labels_table_os t1
INNER JOIN help_cal_features_bad_text t2 ON t1.issue_key = t2.issue_key 
SET t1.num_unusable_issues_cretor_prev_text_ratio_025 = t2.num_unusable_text;

drop table help_cal_features_bad_text;


# calculate ratio bas issues text ratio 025: 

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_ratio_025_ratio float;

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os SET num_unusable_issues_cretor_prev_text_ratio_025_ratio= 
CASE
	WHEN num_unusable_issues_cretor_prev_text_ratio_025 = 0 then 0  
	ELSE num_unusable_issues_cretor_prev_text_ratio_025/num_issues_cretor_prev  
END;




# calculate num unusable issue text ratio 05

create table help_cal_features_bad_text select issue_key, created as created2, creator as creator2, (select count(*) from features_labels_table_os m
									WHERE (f.creator = m.creator and f.project_key = m.project_key 
                                    and f.created > m.created and is_change_text_ratio_words_05 > 0 )) as num_unusable_text
from features_labels_table_os f;

## add the wanted column to feature table:

# First add Age column in table1

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_ratio_05 INT(11);
# then update that column using blow query

UPDATE features_labels_table_os t1
INNER JOIN help_cal_features_bad_text t2 ON t1.issue_key = t2.issue_key 
SET t1.num_unusable_issues_cretor_prev_text_ratio_05 = t2.num_unusable_text;

drop table help_cal_features_bad_text;


# calculate ratio bas issues text ratio 05: 

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_ratio_05_ratio float;

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os SET num_unusable_issues_cretor_prev_text_ratio_05_ratio= 
CASE
	WHEN num_unusable_issues_cretor_prev_text_ratio_05 = 0 then 0  
	ELSE num_unusable_issues_cretor_prev_text_ratio_05/num_issues_cretor_prev  
END;




# calculate num unusable issue text ratio 1

create table help_cal_features_bad_text select issue_key, created as created2, creator as creator2, (select count(*) from features_labels_table_os m
									WHERE (f.creator = m.creator and f.project_key = m.project_key 
                                    and f.created > m.created and is_change_text_ratio_words_1 > 0 )) as num_unusable_text
from features_labels_table_os f;

## add the wanted column to feature table:

# First add Age column in table1

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_ratio_1 INT(11);
# then update that column using blow query

UPDATE features_labels_table_os t1
INNER JOIN help_cal_features_bad_text t2 ON t1.issue_key = t2.issue_key 
SET t1.num_unusable_issues_cretor_prev_text_ratio_1 = t2.num_unusable_text;

drop table help_cal_features_bad_text;


# calculate ratio bas issues text ratio 1: 

ALTER TABLE features_labels_table_os ADD COLUMN num_unusable_issues_cretor_prev_text_ratio_1_ratio float;

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table_os SET num_unusable_issues_cretor_prev_text_ratio_1_ratio= 
CASE
	WHEN num_unusable_issues_cretor_prev_text_ratio_1 = 0 then 0  
	ELSE num_unusable_issues_cretor_prev_text_ratio_1/num_issues_cretor_prev  
END;










###################################################################
########################### open source ###########################
###################################################################

# calculate num issues:
Alter table features_labels_table add column num_issues_cretor_prev INT(11);

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table SET num_issues_cretor_prev= (select count(*) from features_labels_table m
									WHERE (f.creator = m.creator and f.project_key = m.project_key and f.created > m.created));


# calculate num bad issues text:

Alter table features_labels_table add column num_unusable_issues_cretor_prev_text INT(11);

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table SET num_unusable_issues_cretor_prev_text= (select count(*) from features_labels_table m
									WHERE (f.creator = m.creator and f.project_key = m.project_key and f.created > m.created 
                                    and has_changes_summary_description_sprint > 0 ));
                                    

# calculate num bad issues story_point:

Alter table features_labels_table add column num_unusable_issues_cretor_prev_storyp INT(11);

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table SET num_unusable_issues_cretor_prev_storyp= (select count(*) from features_labels_table m
									WHERE (f.creator = m.creator and f.project_key = m.project_key and f.created > m.created 
                                    and has_changes_story_point_sprint > 0 ));
                                    
                                    
# calculate num bad issues more than 1 sprint:

Alter table features_labels_table add column num_unusable_issues_cretor_prev_sprint INT(11);

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table SET num_unusable_issues_cretor_prev_sprint= (select count(*) from features_labels_table m
									WHERE (f.creator = m.creator and f.project_key = m.project_key and f.created > m.created 
                                    and num_sprints > 1 ));
                                    
                                    
# calculate num bad issues all :

Alter table features_labels_table add column num_unusable_issues_cretor_prev INT(11);

SET SQL_SAFE_UPDATES = 0;
UPDATE features_labels_table SET num_unusable_issues_cretor_prev= num_unusable_issues_cretor_prev_sprint 
									+ num_unusable_issues_cretor_prev_storyp + num_unusable_issues_cretor_prev_text;





