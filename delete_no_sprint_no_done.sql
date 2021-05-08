

UPDATE main_table_os SET num_comments_before_sprint = (select count(*)
														from comments c1
														where created <= (select time_add_to_sprint 
																			from main_table_os c2 
																			where c2.issue_key = c1.issue_key));
                                                                            
                                                                            

SET SQL_SAFE_UPDATES = 0;
delete 
from main_table2 
where num_sprints = 0;


delete 
from main_table2 
where status_name != 'Done' and status_name != 'Closed';



# ######################### done  #########################


delete 
from changes_description_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);  



SET SQL_SAFE_UPDATES = 0;
delete 
from changes_summary_os 
where issue_key2 NOT IN (SELECT m.issue_key
						FROM main_table_os m);
 


SET SQL_SAFE_UPDATES = 0;
delete 
from changes_story_points_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);
                        

SET SQL_SAFE_UPDATES = 0;
delete 
from changes_criteria_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);
                        

SHOW PROCESSLIST;
                        
delete 
from comments_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);
  
  
                        
SET SQL_SAFE_UPDATES = 0;
delete 
from changes_sprint_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);
 
 

delete 
from commits_info_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);


                        
delete 
from components_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);
  
  
                        
delete 
from fix_versions_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);


                        
delete 
from sprints_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);


                        
delete 
from issue_links_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);


                        
delete 
from names_bugs_issue_links_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);

                      
delete 
from sab_task_names_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);
 
                        
delete 
from labels_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);
                        

delete 
from versions_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);
                        
                        
                        
# ######################### still not done   #########################

 
                        
                        
delete 
from all_changes_os
where issue_key NOT IN (SELECT m.issue_key
						FROM main_table_os m);
