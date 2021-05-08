# add columns

# open sources
ALTER TABLE data_base_os.main_table_os ADD COLUMN time_status_close datetime; # python

ALTER TABLE data_base_os.all_changes_os ADD COLUMN time_add_to_sprint datetime; # here
ALTER TABLE data_base_os.all_changes_os ADD COLUMN is_after_sprint int(10); # here
ALTER TABLE data_base_os.all_changes_os ADD COLUMN time_from_sprint float; # here
ALTER TABLE data_base_os.all_changes_os ADD COLUMN is_after_close INT(10); # python



SET SQL_SAFE_UPDATES = 0;
UPDATE data_base_os.all_changes_os t1
INNER JOIN data_base_os.main_table_os t2 ON t1.issue_key = t2.issue_key and (t1.field='summary' or t1.field='description' or 
t1.field='Acceptance Criteria' or t1.field='sprint' or t1.field='story points' or t1.field='link')
SET t1.time_add_to_sprint = t2.time_add_to_sprint;


SET SQL_SAFE_UPDATES = 0;
UPDATE data_base_os.all_changes_os SET is_after_sprint= 
CASE
	WHEN time_add_to_sprint < created then 1
	ELSE 0
END;

UPDATE data_base_os.all_changes_os SET time_from_sprint= TIMESTAMPDIFF(minute, time_add_to_sprint, created)/60;




