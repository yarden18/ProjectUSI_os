import mysql
import pandas as pd
import mysql.connector
import pandasql as ps


def choose_data_base(db_name):
    mysql_con = mysql.connector.connect(user='root', password='', host='localhost',
                                        database='{}'.format(db_name), auth_plugin='mysql_native_password',
                                        use_unicode=True)
    return mysql_con


def remove_extracted_links(link_data):

    for j in range(0, len(link_data)):
        for i in range(j, len(link_data)):
            try:
                if (link_data['if_before'][j] == 1 and link_data['if_before'][i] == 1 and
                        link_data['issue_key'][j] == link_data['issue_key'][i] and
                        link_data['to_string'][j] == link_data['from_string'][i] and
                        link_data['created_link'][j] < link_data['created_link'][i]):
                    link_data['to_string'][j] == ""
            except:
                a = 8

    return link_data


def check_if_link_in_string(is_before, link_str, word):
    try:
        for i in range(0, len(word)):
            if link_str.lower().count(word[i]) > 0 and is_before == 1:
                return 1
        return 0
    except:
        return 0


def create_issue_links_features(data_help, original_data):

    data_help['block'] = data_help.apply(lambda x: check_if_link_in_string(x['if_before'], x['to_string'], ['blocks',
                                                                'has to be done before', 'is triggering']), axis=1)
    data_help['block_by'] = data_help.apply(lambda x: check_if_link_in_string(x['if_before'], x['to_string'],
                                                                              ['has to be done after',
                      'is blocked by', 'is triggered by', 'depends on', 'depended on by', 'is depended on by']), axis=1)
    data_help['duplicate'] = data_help.apply(lambda x: check_if_link_in_string(x['if_before'], x['to_string'],
                                                                               ['clones', 'cloned from', 'duplicates',
                                                                                'is clone of']), axis=1)
    data_help['duplicate_by'] = data_help.apply(
        lambda x: check_if_link_in_string(x['if_before'], x['to_string'], ['cloned to', 'is cloned by',
                                                                           'is duplicated by']), axis=1)
    data_help['relates'] = data_help.apply(lambda x: check_if_link_in_string(x['if_before'], x['to_string'],
                                                                             ['relates to', 'is related to',
                                                                              'is related to by']), axis=1)

    data_help2 = (ps.sqldf("""select issue_key1, sum(block) as num_block, sum(block_by) as num_block_by, 
                              sum(duplicate) as num_duplicate, sum(duplicate_by) as num_duplicate_by,
                              sum(relates) as num_relates
                              from data_help
                              group by issue_key1 
                              """))

    original_data['block'] = data_help2['num_block']
    original_data['block_by'] = data_help2['num_block_by']
    original_data['duplicate_by'] = data_help2['num_duplicate_by']
    original_data['relates'] = data_help2['num_relates']
    original_data['duplicate'] = data_help2['num_duplicate']

    return original_data


def create_issue_links_all(data_developer, data_repo, data_xd, data_dm):

    db_name_os = 'data_base_os'
    mysql_con_os = choose_data_base(db_name_os)

    help_data_developer = pd.read_sql("select t3.issue_key as issue_key1, t2.issue_key as issue_key2, "
                                      "t3.time_add_to_sprint, t2.created, t2.from_string, t2.to_string, t2.field, "
                                      "t3.time_add_to_sprint>t2.created as if_before from "
                                      "data_base_os.features_labels_table_os2 t3 left join "
                                      "data_base_os.all_changes_os t2 ON t3.issue_key = t2.issue_key "
                                      "and t2.field = 'Link' where t3.issue_key is not null and "
                                      "t3.project_key = 'DEVELOPER' ", con=mysql_con_os)
    help_data_developer = remove_extracted_links(help_data_developer)
    data_developer = create_issue_links_features(help_data_developer, data_developer)

    help_data_repo = pd.read_sql("select t3.issue_key as issue_key1, t2.issue_key as issue_key2, "
                                 "t3.time_add_to_sprint, t2.created, t2.from_string, t2.to_string, t2.field, "
                                 "t3.time_add_to_sprint>t2.created as if_before from "
                                 "data_base_os.features_labels_table_os2 t3 left join "
                                 "data_base_os.all_changes_os t2 ON t3.issue_key = t2.issue_key "
                                 "and t2.field = 'Link' where t3.issue_key is not null and "
                                 "t3.project_key = 'REPO' ", con=mysql_con_os)
    help_data_repo = remove_extracted_links(help_data_repo)
    data_repo = create_issue_links_features(help_data_repo, data_repo)

    help_data_xd = pd.read_sql("select t3.issue_key as issue_key1, t2.issue_key as issue_key2, "
                               "t3.time_add_to_sprint, t2.created, t2.from_string, t2.to_string, t2.field, "
                               "t3.time_add_to_sprint>t2.created as if_before from "
                               "data_base_os.features_labels_table_os2 t3 left join "
                               "data_base_os.all_changes_os t2 ON t3.issue_key = t2.issue_key "
                               "and t2.field = 'Link' where t3.issue_key is not null and "
                               "t3.project_key = 'XD' ", con=mysql_con_os)
    help_data_xd = remove_extracted_links(help_data_xd)
    data_xd = create_issue_links_features(help_data_xd, data_xd)

    help_data_dm = pd.read_sql("select t3.issue_key as issue_key1, t2.issue_key as issue_key2, "
                               "t3.time_add_to_sprint, t2.created, t2.from_string, t2.to_string, t2.field, "
                               "t3.time_add_to_sprint>t2.created as if_before from "
                               "data_base_os.features_labels_table_os2 t3 left join "
                               "data_base_os.all_changes_os t2 ON t3.issue_key = t2.issue_key "
                               "and t2.field = 'Link' where t3.issue_key is not null and "
                               "t3.project_key = 'DM' ", con=mysql_con_os)
    help_data_dm = remove_extracted_links(help_data_dm)
    data_dm = create_issue_links_features(help_data_dm, data_dm)

    return data_developer, data_repo, data_xd, data_dm
