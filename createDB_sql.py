from jira import JIRA
import pandas as pd
import requests
import git
import time
import mysql.connector
import datetime
import difflib

"""
the function next are getting the deatails of the issue and extract the wanted fields from it ant return this data 
"""

def get_issue_story_points(issue, name_map):
    try:
        story_points = getattr(issue.fields, name_map['Story Points'])
    except AttributeError:
        story_points = None
    return story_points


def get_team(issue, name_map):
    try:
        team = getattr(issue.fields, name_map['Team'])
    except:
        team = None
    return team


def get_attachment(issue):
    try:
        attachment = issue.fields.attachment
        if attachment is not None:
            is_attachment = 1
        else:
            is_attachment = 0
    except:
        attachment = None
        is_attachment = 0
    return attachment, is_attachment


def get_image(issue):
    try:
        image1 = issue.fields.thumbnail
        if image1 is not None:
            is_image = 1
        else:
            is_image = 0
    except:
        image1 = None
        is_image = 0
    return image1, is_image


def get_issue_acceptance_cri(issue, name_map):

    try:
        issue_acceptance_cri = getattr(issue.fields, name_map['Acceptance Criteria'])
    except:
        issue_acceptance_cri = None

    return issue_acceptance_cri


def get_issue_pull_request_url(issue, name_map):

    try:
        pull_request_url = getattr(issue.fields, name_map['Pull Request URL'])
    except:
        pull_request_url = None

    if project_key == 'REPO':
        try:
            pull_request_url = getattr(issue.fields, name_map['External issue URL'])
        except:
            pull_request_url = None

    return pull_request_url


def get_issue_priority(project_name, issue, name_map):
    issue_priority = ""
    if project_name == 'DEVELOPER':
        try:
            issue_priority = getattr(issue.fields, name_map['Class of work']).value
            issue_priority = str(issue_priority)
        except AttributeError:
            issue_priority = ""
    if project_name == 'REPO':
        issue_priority = getattr(issue.fields, name_map['Priority Level'])
        issue_priority = str(issue_priority)
    if project_name == 'XD' or project_name == 'USERGRID' or project_name == 'DM':
        issue_priority = str(issue.fields.priority)
    return issue_priority


def get_issue_versions(mydb, sql_versions,  issue, issue_key, project_key):
    num_versions = 0
    lst_versions = []
    if project_key == 'REPO' or project_key == 'USERGRID' or project_key == 'XD':
        version = issue.fields.versions
        num_versions = len(version)
        for i in range(0, num_versions):
            lst_versions.append((issue_key, project_key, version[i].name, i+1))
    if len(lst_versions) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_versions, lst_versions)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
    return num_versions


def get_issue_fix_versions(mydb, sql_fix_versions,  issue, issue_key, project_key):
    fix_versions = issue.fields.fixVersions
    num_fix_versions = len(fix_versions)
    lst_fix_versions = []
    for i in range(0, num_fix_versions):
        lst_fix_versions.append((issue_key, project_key, fix_versions[i].name, i+1))
    if len(lst_fix_versions) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_fix_versions, lst_fix_versions)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
    return num_fix_versions


def get_issue_labels(mydb, sql_labels,  issue, issue_key, project_key):
    labels = issue.fields.labels
    num_labels = len(labels)
    lst_labels = []
    for i in range(0, num_labels):
        lst_labels.append((issue_key, project_key, labels[i], i+1))
    if len(lst_labels) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_labels, lst_labels)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
    return num_labels


def get_issue_components(mydb, sql_components,  issue, issue_key, project_key):
    components = issue.fields.components
    num_components = len(components)
    lst_components = []
    for i in range(0, num_components):
        lst_components.append((issue_key, project_key, components[i].name, i+1))
    if len(lst_components) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_components, lst_components)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
    return num_components


def data_of_first_response(project_name, issue, name_map):
    issue_data_of_first_response = None
    if project_name == 'DEVELOPER':
        issue_data_of_first = getattr(issue.fields, name_map['Date of First Comment'])
        if issue_data_of_first is not None:
            issue_data_of_first_response = datetime.datetime.strptime(issue_data_of_first[:-5], '%Y-%m-%dT%H:%M:%S.%f')
        else:
            issue_data_of_first_response = issue_data_of_first
    if project_name == 'XD':
        issue_data_of_first = getattr(issue.fields, name_map['First Response Date'])
        if issue_data_of_first is not None:
            issue_data_of_first_response = datetime.datetime.strptime(issue_data_of_first[:-5], '%Y-%m-%dT%H:%M:%S.%f')
        else:
            issue_data_of_first_response = issue_data_of_first
    if project_name == 'USERGRID':
        issue_data_of_first = getattr(issue.fields, name_map['Date of First Response'])
        if issue_data_of_first is not None:
            issue_data_of_first_response = datetime.datetime.strptime(issue_data_of_first[:-5], '%Y-%m-%dT%H:%M:%S.%f')
        else:
            issue_data_of_first_response = issue_data_of_first
    return issue_data_of_first_response


def get_sub_tasks_info(mydb, sql_sab_task_names, issue_sub, issue_key, project_key):
    """ return the sub tasks information of an issue
    param: issue_sub: issue key
    return: sub_task_info, list of names of the sub tasks issue, and number of sub tasks
    """
    sub_tasks = issue_sub.fields.subtasks
    lst_sub_tasks_names = []
    if len(sub_tasks) != 0:
        num_sub_tasks = len(sub_tasks)
        for i in range(0, num_sub_tasks):
            lst_sub_tasks_names.append((issue_key, project_key, sub_tasks[i].key, i+1))
    else:
        num_sub_tasks = 0
    if len(lst_sub_tasks_names) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_sab_task_names, lst_sub_tasks_names)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
    return num_sub_tasks


def get_issue_links_info(mydb, sql_issue_links, issue_l, issue_key, project_key):
    """the function return the issue links information - names of issues link, names of issue links
    type bus, number of issue links and number of issue links type bug.
    param: issue_l: issue
    return: issue_link_info,
    """
    issue_links = issue_l.fields.issuelinks
    num_issue_links = len(issue_links)
    lst_issue_links = []
    for i in range(0, num_issue_links):
        try:
            lst_issue_links.append((issue_key, project_key, issue_l.fields.issuelinks[i].inwardIssue.key,
                                          issue_l.fields.issuelinks[i].raw['type']['inward'], i+1))
        except:
            lst_issue_links.append((issue_key, project_key, issue_l.fields.issuelinks[i].outwardIssue.key,
                                    issue_l.fields.issuelinks[i].raw['type']['outward'], i+1))
    if len(lst_issue_links) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_issue_links, lst_issue_links)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
    return num_issue_links


def get_issue_link_bug_name(mydb, sql_names_bugs_issue_links, issue_l, issue_key, project_key):
    """the function return the issue links names of issues link from type bug.
        param: issue_l: issue
        return: lst_issue_links_bugs, list of names of issues link from type bug.
    """
    issue_links = issue_l.fields.issuelinks
    num_issue_links_bugs = 0
    lst_issue_links_bugs = []
    for i in range(0, len(issue_links)):
        try:
            lst_issue_links_bugs1 = issue_links[i].inwardIssue.fields.issuetype.name
            if lst_issue_links_bugs1 == 'Bug':
                num_issue_links_bugs += 1
                lst_issue_links_bugs.append((issue_key, project_key, issue_links[i].inwardIssue.key, i+1))
        except:
            lst_issue_links_bugs1 = issue_links[i].outwardIssue.fields.issuetype.name
            if lst_issue_links_bugs1 == 'Bug':
                num_issue_links_bugs += 1
                lst_issue_links_bugs.append((issue_key, project_key, issue_links[i].outwardIssue.key, i+1))
    if len(lst_issue_links_bugs) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_names_bugs_issue_links, lst_issue_links_bugs)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
    return num_issue_links_bugs


def get_sprint_info(mydb, sql_sprints, issue_s, name_map, issue_key, project_key):
    """the function return the issue sprint information
        param: issue_s: issue,  name_map: name of all fields
        return: sprint_info, list 3 lists - names of sprints, start dates of sprints, end dates of sprints, and num of
        sprints.
        """
    sprint = getattr(issue_s.fields, name_map['Sprint'])
    sprints = []
    if sprint is not None:
        for i in range(0, len(sprint)):
            start_date1 = sprint[i].split("startDate=", 1)[1][:10]
            end_date1 = sprint[i].split("endDate=", 1)[1][:10]
            is_over = 1
            if end_date1[1] == 'n':
                start_date = None
                end_date = None
                is_over = 0
            else:
                start_date = datetime.datetime.strptime(start_date1, '%Y-%m-%d')
                end_date = datetime.datetime.strptime(end_date1, '%Y-%m-%d')
            sprints.append((issue_key, project_key, sprint[i].split("name=")[1].split(",", 1)[0], start_date, end_date,
                            is_over, i+1))
        num_sprints = len(sprint)
    else:
        num_sprints = 0
    if len(sprints) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_sprints, sprints)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
    return num_sprints


def get_comments_info(mydb, sql_comments, issue_comment, auth_jira_comment, issue_key, project_key):
    """the function return the issue comments information
        param: issue_comment: issue,  auth_jira_work: auth jira_comment
        return: lst_comments_info, dictionary
    """
    lst_comments_info = []
    comments = auth_jira_comment.comments(issue_comment.id)
    num_comments = len(comments)
    for i in range(0, num_comments):
        created1 = comments[i].created
        created = datetime.datetime.strptime(created1[:-5], '%Y-%m-%dT%H:%M:%S.%f')
        lst_comments_info.append(
            (issue_key, project_key, comments[i].author.displayName, comments[i].id, created,
             comments[i].body, i+1))
    if len(lst_comments_info) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_comments, lst_comments_info)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")

    return num_comments


def clean_commit_message(commit_message):
    if "git-svn-id" in commit_message:
        return commit_message.split("git-svn-id")[0]
    return commit_message


def commits_and_issues(mydb, sql_commits, repo, issue_commit, issue_key1, project_key):
    def get_num_from_comit_summary(commit_text, issue_key):
        s = commit_text.lower().replace(":", "").replace("#", "").replace("/", " ").replace("_", " ").replace(".", "").replace("'", "").replace(",", "").split()
        if issue_key in s:
            return 1
        return 0
    issue_key = issue_commit.key.lower()
    count = 1
    lst_commits_info = []
    if issue_commit.fields.project.key == 'DM':
        for i in range(0, 3):
            for git_commit in repo[i].iter_commits():
                return_if_word_found_in_commit = get_num_from_comit_summary(git_commit.summary, issue_key)
                if return_if_word_found_in_commit != 0:
                    lst_commits_info.append(
                        (issue_key1, project_key, git_commit.author.name, git_commit.stats.total['insertions'],
                         git_commit.stats.total['deletions'], git_commit.stats.total['lines'],
                         git_commit.stats.total['files'], git_commit.summary,
                         git_commit.message, git_commit.hexsha, count))
                    count += 1
    else:
        for git_commit in repo.iter_commits():
            return_if_word_found_in_commit = get_num_from_comit_summary(git_commit.summary, issue_key)
            if return_if_word_found_in_commit != 0:
                lst_commits_info.append((
                issue_key1, project_key, git_commit.author.name, git_commit.stats.total['insertions'],
                git_commit.stats.total['deletions'], git_commit.stats.total['lines'], git_commit.stats.total['files'],
                git_commit.summary, git_commit.message, git_commit.hexsha, count))
                count += 1
    if len(lst_commits_info) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_commits, lst_commits_info)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
    return count


def get_different_between_string(from_str, to_str):
    if from_str is None:
        from_str = ""
    if to_str is None:
        to_str = ""
    ratio_char = difflib.SequenceMatcher(None, from_str, to_str).ratio()
    ratio_words = difflib.SequenceMatcher(None, from_str.split(), to_str.split()).ratio()

    diff_char_minus = len([(i, li) for i, li in enumerate(difflib.ndiff(from_str, to_str)) if li[0] == '-'])
    diff_char_plus = len([(i, li) for i, li in enumerate(difflib.ndiff(from_str, to_str)) if li[0] == '+'])
    diff_char_all = len([(i, li) for i, li in enumerate(difflib.ndiff(from_str, to_str)) if li[0] != ' '])

    diff_word_minus = len([(i, li) for i, li in enumerate(difflib.ndiff(from_str.split(), to_str.split())) if li[0] == '-'])
    diff_word_plus = len([(i, li) for i, li in enumerate(difflib.ndiff(from_str.split(), to_str.split())) if li[0] == '+'])
    diff_word_all = len([(i, li) for i, li in enumerate(difflib.ndiff(from_str.split(), to_str.split())) if li[0] != ' '])

    return ratio_char, ratio_words, diff_char_minus, diff_char_plus, diff_char_all, diff_word_minus, diff_word_plus, diff_word_all


def get_changes_issue(mydb, sql_all_changes, sql_changes_summary, sql_changes_description, sql_changes_story_points,
                      sql_changes_sprints, issue_change, issue_key, project_key, time_created_issue, summary_last,
                      description_last, acceptance_last):
    ''' the function get the issue and returns 4 lists of all types of his change, while each is a dict type
    with headlines of author, created, from string, to string, and in the dict of all changes also field.
    param: issue_change: the issue
    return: 4 lists, of 4 types of changes in issue- all, story point, summary, description
    '''
    histories = issue_change.changelog.histories
    lst_all_changes = []
    lst_changes_summary = []
    lst_changes_description = []
    lst_changes_acceptance_criteria = []
    lst_changes_story_points = []
    lst_changes_sprints = []
    num_changes = 0
    num_changes_summary = 0
    num_changes_description = 0
    num_changes_acceptance_criteria = 0
    num_changes_story_point = 0
    num_changes_sprint = 0
    count_changes = 0
    count_changes_summary = 0
    count_changes_description = 0
    count_changes_acceptance_criteria = 0
    count_changes_story_point = 0
    count_changes_sprint = 0
    if len(histories) != 0:
        for i in range(0, len(histories)):
            for j in range(0, len(histories[i].items)):
                if histories[i].items[j].fromString is None:
                    is_first_setup = 1
                else:
                    num_changes += 1
                    is_first_setup = 0
                count_changes += 1
                created1 = histories[i].created
                created = datetime.datetime.strptime(created1[:-5], '%Y-%m-%dT%H:%M:%S.%f')
                from_string = histories[i].items[j].fromString
                to_string = histories[i].items[j].toString
                field = histories[i].items[j].field
                different_dates = created - time_created_issue
                different_dates_in_hours = different_dates.total_seconds() / 3600
                if different_dates_in_hours < 1:
                    if_change_first_hour = 1
                else:
                    if_change_first_hour = 0
                try:
                    author = histories[i].author.displayName
                except AttributeError:
                    try:
                        author = str(histories[i].author)
                    except AttributeError:
                        author = None
                lst_all_changes.append((issue_key, project_key, author, created, from_string, to_string, field,
                                        if_change_first_hour, different_dates_in_hours, is_first_setup, count_changes))
                if field == 'Story Points':
                    count_changes_story_point += 1
                    if is_first_setup == 0:
                        num_changes_story_point += 1
                    lst_changes_story_points.append((issue_key, project_key, author, created, from_string, to_string,
                                                     if_change_first_hour, different_dates_in_hours, is_first_setup,
                                                     count_changes_story_point))
                if field == 'summary':
                    count_changes_summary += 1
                    if is_first_setup == 0:
                        num_changes_summary += 1

                    ratio_char_next, ratio_words_next, diff_char_minus_next, diff_char_plus_next, diff_char_all_next, diff_word_minus_next, diff_word_plus_next, diff_word_all_next = get_different_between_string(from_string, to_string)
                    ratio_char_last, ratio_words_last, diff_char_minus_last, diff_char_plus_last, diff_char_all_last, diff_word_minus_last, diff_word_plus_last, diff_word_all_last = get_different_between_string(
                        from_string, summary_last)
                    if diff_char_all_last < 10:
                        is_diff_more_than_ten = 0
                    else:
                        is_diff_more_than_ten = 1

                    lst_changes_summary.append((issue_key, project_key, author, created, from_string, to_string,
                                                if_change_first_hour, different_dates_in_hours, is_first_setup,
                                                is_diff_more_than_ten, count_changes_summary, ratio_char_next,
                                                ratio_words_next, diff_char_minus_next, diff_char_plus_next,
                                                diff_char_all_next, diff_word_minus_next, diff_word_plus_next,
                                                diff_word_all_next, ratio_char_last, ratio_words_last,
                                                diff_char_minus_last, diff_char_plus_last, diff_char_all_last,
                                                diff_word_minus_last, diff_word_plus_last, diff_word_all_last))
                if field == 'description':
                    count_changes_description += 1
                    if is_first_setup == 0:
                        num_changes_description += 1

                    ratio_char_next, ratio_words_next, diff_char_minus_next, diff_char_plus_next, diff_char_all_next, diff_word_minus_next, diff_word_plus_next, diff_word_all_next = get_different_between_string(
                        from_string, to_string)
                    ratio_char_last, ratio_words_last, diff_char_minus_last, diff_char_plus_last, diff_char_all_last, diff_word_minus_last, diff_word_plus_last, diff_word_all_last = get_different_between_string(
                        from_string, description_last)
                    if diff_char_all_last < 10:
                        is_diff_more_than_ten = 0
                    else:
                        is_diff_more_than_ten = 1

                    lst_changes_description.append((issue_key, project_key, author, created, from_string, to_string,
                                                    if_change_first_hour, different_dates_in_hours, is_first_setup,
                                                    is_diff_more_than_ten, count_changes_description, ratio_char_next,
                                                    ratio_words_next, diff_char_minus_next, diff_char_plus_next,
                                                    diff_char_all_next, diff_word_minus_next, diff_word_plus_next,
                                                    diff_word_all_next, ratio_char_last, ratio_words_last,
                                                    diff_char_minus_last, diff_char_plus_last, diff_char_all_last,
                                                    diff_word_minus_last, diff_word_plus_last, diff_word_all_last))

                if field == 'Acceptance Criteria':
                    count_changes_acceptance_criteria += 1
                    if is_first_setup == 0:
                        num_changes_acceptance_criteria += 1

                    ratio_char_next, ratio_words_next, diff_char_minus_next, diff_char_plus_next, diff_char_all_next, diff_word_minus_next, diff_word_plus_next, diff_word_all_next = get_different_between_string(
                        from_string, to_string)
                    ratio_char_last, ratio_words_last, diff_char_minus_last, diff_char_plus_last, diff_char_all_last, diff_word_minus_last, diff_word_plus_last, diff_word_all_last = get_different_between_string(
                        from_string, acceptance_last)
                    if diff_char_all_last < 10:
                        is_diff_more_than_ten = 0
                    else:
                        is_diff_more_than_ten = 1

                    lst_changes_acceptance_criteria.append((issue_key, project_key, author, created, from_string,
                                                            to_string, if_change_first_hour, different_dates_in_hours,
                                                            is_first_setup, is_diff_more_than_ten,
                                                            count_changes_acceptance_criteria, ratio_char_next,
                                                            ratio_words_next, diff_char_minus_next, diff_char_plus_next,
                                                            diff_char_all_next, diff_word_minus_next,
                                                            diff_word_plus_next, diff_word_all_next, ratio_char_last,
                                                            ratio_words_last, diff_char_minus_last, diff_char_plus_last,
                                                            diff_char_all_last, diff_word_minus_last,
                                                            diff_word_plus_last, diff_word_all_last))

                if field == 'Sprint':
                    count_changes_sprint += 1
                    if is_first_setup == 0:
                        num_changes_sprint += 1
                    lst_changes_sprints.append((issue_key, project_key, author, created, from_string, to_string,
                                                if_change_first_hour, different_dates_in_hours, is_first_setup,
                                                count_changes_sprint))
    if len(lst_all_changes) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_all_changes, lst_all_changes)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
    if len(lst_changes_summary) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_changes_summary, lst_changes_summary)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
    if len(lst_changes_description) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_changes_description, lst_changes_description)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
    if len(lst_changes_acceptance_criteria) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_changes_acceptance_criteria, lst_changes_acceptance_criteria)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
    if len(lst_changes_story_points) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_changes_story_points, lst_changes_story_points)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
    if len(lst_changes_sprints) != 0:
        mycursor = mydb.cursor()
        try:
            mycursor.executemany(sql_changes_sprints, lst_changes_sprints)
            mydb.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")

    return num_changes, num_changes_summary, num_changes_description, num_changes_acceptance_criteria, num_changes_story_point, num_changes_sprint


def get_projects_info(project_num):
    """ get project num ad return his important info - project name, auth_jira, csv path, and repo. you need to save the git and update the adress in your local pc. 
    param: project_num from 1 to 5
    return: project info (auth, project name, url, repo)
    """
    auth_jira = ""
    project_name = ""
    url = ""
    git_path = ""
    repo = ""
    csv_path = ""
    if project_num == 1:
        auth_jira = JIRA('https://issues.apache.org/jira', basic_auth=('', ''))
        project_name = 'USERGRID'
        url = 'https://issues.apache.org/jira'
        git_path = r""
        repo = git.Repo(git_path)
        csv_path = ''
    if project_num == 2:
        auth_jira = JIRA('https://jira.spring.io', basic_auth=('', ''))
        project_name = 'XD'
        url = 'https://jira.spring.io'
        git_path = r""
        repo = git.Repo(git_path)
        csv_path = ''
    if project_num == 3:
        auth_jira = JIRA('https://issues.jboss.org', basic_auth=('', ''))
        project_name = 'DEVELOPER'
        url = 'https://issues.jboss.org'
        git_path = r""
        repo = git.Repo(git_path)
        csv_path = ''
    if project_num == 4:
        auth_jira = JIRA('https://issues.alfresco.com/jira', basic_auth=('', ''))
        project_name = 'REPO'
        url = 'https://issues.alfresco.com/jira'
        git_path = r""
        repo = git.Repo(git_path)
        csv_path = ''
    if project_num == 5:
        auth_jira = JIRA('https://jira.lsstcorp.org')
        project_name = 'DM'
        url = 'https://jira.lsstcorp.org'
        git_path = []
        git_path.append(r"pipe_tasks")
        git_path.append(r"pipe_base")
        git_path.append(r"dm_dev_guide")
        repo = []
        repo.append(git.Repo(git_path[0]))
        repo.append(git.Repo(git_path[1]))
        repo.append(git.Repo(git_path[2]))
        csv_path = ''
    return {'auth_jira': auth_jira, 'project_name': project_name, 'repo': repo, 'csv_path': csv_path}


if __name__ == "__main__":
    # connect to SQL
    mysql_con = mysql.connector.connect(user='root', password='',
                                        host='localhost', database='data_base_os',
                                        auth_plugin='mysql_native_password', use_unicode=True)
    # Create a cursor
    cursor = mysql_con.cursor()

    # Enforce UTF-8 for the connection
    cursor.execute('SET NAMES utf8mb4')
    cursor.execute("SET CHARACTER SET utf8mb4")
    cursor.execute("SET character_set_connection=utf8mb4")
    # the SQL queries to enter the data to the tables in sql 
    sql_comments = """INSERT INTO comments_os (issue_key, project_key, author, id, created, body, 
                                                chronological_number) VALUES (%s, %s, %s, %s, %s, %s, %s)"""
    sql_commits = """INSERT INTO commits_info_os (issue_key, project_key, author, insertions, code_deletions, code_lines, files,
                     summary, message, commit, chronological_number) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    sql_sprints = """INSERT INTO sprints_os (issue_key, project_key, sprint_name, start_date, end_date, is_over, 
                      chronological_number) VALUES (%s, %s, %s, %s, %s, %s, %s)"""
    sql_sab_task_names = """INSERT INTO sab_task_names_os (issue_key, project_key, sub_task_name, 
                             chronological_number) VALUES (%s, %s, %s, %s)"""
    sql_versions = """INSERT INTO versions_os (issue_key, project_key, version, 
                       chronological_number) VALUES (%s, %s, %s, %s)"""
    sql_issue_links = """INSERT INTO issue_links_os (issue_key, project_key, issue_link, issue_link_name_relation,
                          chronological_number) VALUES (%s, %s, %s, %s, %s)"""
    sql_names_bugs_issue_links = """INSERT INTO names_bugs_issue_links_os (issue_key, project_key, bug_issue_link, 
                                     chronological_number) VALUES (%s, %s, %s, %s)"""
    sql_fix_versions = """INSERT INTO fix_versions_os (issue_key, project_key, fix_version, 
                           chronological_number) VALUES (%s, %s, %s, %s)"""
    sql_labels = """INSERT INTO labels_os (issue_key, project_key, label, 
                     chronological_number) VALUES (%s, %s, %s, %s)"""
    sql_components = """INSERT INTO components_os (issue_key, project_key, component, 
                         chronological_order) VALUES (%s, %s, %s, %s)"""
    sql_all_changes = """INSERT INTO all_changes_os (issue_key, project_key, author, created, from_string, to_string,
                          field, if_change_first_hour, different_time_from_creat, is_first_setup,
                          chronological_number) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    sql_changes_description = """INSERT INTO changes_description_os (issue_key, project_key, author, created, from_string,
                                  to_string, if_change_first_hour, different_time_from_creat, is_first_setup,
                                  is_diff_more_than_ten, chronological_number, ratio_different_char_next, 
                                  ratio_different_word_next, num_different_char_minus_next, num_different_char_plus_next,
                                  num_different_char_all_next, num_different_word_minus_next, 
                                  num_different_word_plus_next, num_different_word_all_next, ratio_different_char_last,
                                  ratio_different_word_last, num_different_char_minus_last, num_different_char_plus_last,
                                  num_different_char_all_last, num_different_word_minus_last, 
                                  num_different_word_plus_last, num_different_word_all_last) VALUES (%s, %s, %s, %s, %s, 
                                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                                  %s, %s)"""
    sql_changes_summary = """INSERT INTO changes_summary_os (issue_key, project_key, author, created, from_string,
                                  to_string, if_change_first_hour, different_time_from_creat, is_first_setup,
                                  is_diff_more_than_ten, chronological_number, ratio_different_char_next, 
                                  ratio_different_word_next, num_different_char_minus_next, num_different_char_plus_next,
                                  num_different_char_all_next, num_different_word_minus_next, 
                                  num_different_word_plus_next, num_different_word_all_next, ratio_different_char_last,
                                  ratio_different_word_last, num_different_char_minus_last, num_different_char_plus_last,
                                  num_different_char_all_last, num_different_word_minus_last, 
                                  num_different_word_plus_last, num_different_word_all_last) VALUES (%s, %s, %s, %s, %s, 
                                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                                  %s, %s)"""
    sql_changes_story_points = """INSERT INTO changes_story_points_os (issue_key, project_key, author, created, 
                                   from_string, to_string, if_change_first_hour, different_time_from_creat, 
                                   is_first_setup, chronological_number) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    sql_changes_sprint = """INSERT INTO changes_sprint_os (issue_key, project_key, author, created, 
                             from_string, to_string, if_change_first_hour, different_time_from_creat, 
                             is_first_setup, chronological_number) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    sql_changes_acceptance_criteria = """INSERT INTO changes_criteria_os (issue_key, project_key, author, created, from_string,
                                  to_string, if_change_first_hour, different_time_from_creat, is_first_setup,
                                  is_diff_more_than_ten, chronological_number, ratio_different_char_next, 
                                  ratio_different_word_next, num_different_char_minus_next, num_different_char_plus_next,
                                  num_different_char_all_next, num_different_word_minus_next, 
                                  num_different_word_plus_next, num_different_word_all_next, ratio_different_char_last,
                                  ratio_different_word_last, num_different_char_minus_last, num_different_char_plus_last,
                                  num_different_char_all_last, num_different_word_minus_last, 
                                  num_different_word_plus_last, num_different_word_all_last) VALUES (%s, %s, %s, %s, %s, 
                                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                                  %s, %s)"""
    sql_main_table = """INSERT INTO main_table_os (issue_key, issue_id, project_key, created, creator, reporter, 
                         assignee, date_of_first_response, epic_link, issue_type, last_updated, priority,    
                         prograss, prograss_total, resolution, resolution_date, status_name, status_description, 
                         time_estimate, time_origion_estimate, time_spent, attachment, is_attachment, pull_request_url,
                         images, is_images, team, story_point, summary, description, acceptance_criteria,
                         num_all_changes, num_bugs_issue_link, num_changes_summary, num_changes_description, 
                         num_changes_acceptance_criteria, num_changes_story_point, num_comments, num_issue_links, 
                         num_of_commits, num_sprints, num_sub_tasks, num_watchers, 
                         num_worklog, num_versions, num_fix_versions, num_labels, num_components) 
                         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                         %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                         %s, %s, %s, %s, %s)"""

    # run for all the 5 projects:
    for num in range(1, 6):
        print(num)
        # *********get project info, the loop moves on all 5 projects*****************
        project_info = get_projects_info(num)
        auth_jira = project_info['auth_jira']
        project_name = project_info['project_name']
        repo = project_info['repo']
        csv_path = project_info['csv_path']
        all_fields = auth_jira.fields()
        name_map = {field['name']: field['id'] for field in all_fields}
        size = 100
        initial = 0
        while True:
            start = initial * size
            try:
                issues = auth_jira.search_issues("project={0}".format(project_name), start, size, expand='changelog')
            except requests.exceptions.RequestException as e:
                print(e)
                issues = auth_jira.search_issues("project={0}".format(project_name), start, size)
            if len(issues) == 0:
                break
            initial += 1
            # *************** run over all issues****************
            # for each issue, extract all the data fields from the net into the sql tables, some by using the fucntion in the start
            for issue in issues:
                time.sleep(1)
                issue_type = issue.fields.issuetype.name
                if issue_type == 'Bug':
                    continue
                issue_key = issue.key
                if issue_key == 'DM-590':
                    continue
                if issue_key == 'DM-591':
                    continue
                if issue_key == 'DM-592':
                    continue
                if issue_key == 'DM-593':
                    continue
                num_worklog = len(auth_jira.worklogs(issue))
                project_key = issue.fields.project.key
                status_name = issue.fields.status.name
                if status_name != 'Done' and status_name != 'Closed':
                    continue
                resolution = str(issue.fields.resolution)
                if resolution != 'Done' and resolution != 'Complete' and resolution != 'Fixed':
                    continue
                num_sprints = get_sprint_info(mysql_con, sql_sprints, issue, name_map, issue_key, project_key)
                if num_sprints == 0:
                    continue
                num_comments = get_comments_info(mysql_con, sql_comments, issue, auth_jira, issue_key, project_key)
                num_commits = commits_and_issues(mysql_con, sql_commits, repo, issue, issue_key, project_key)
                num_sub_tasks = get_sub_tasks_info(mysql_con, sql_sab_task_names, issue, issue_key, project_key)
                num_issue_links = get_issue_links_info(mysql_con, sql_issue_links, issue, issue_key, project_key)
                num_issue_link_bug = get_issue_link_bug_name(mysql_con, sql_names_bugs_issue_links, issue, issue_key,
                                                                                 project_key)
                num_components = get_issue_components(mysql_con, sql_components, issue, issue_key, project_key)
                num_labels = get_issue_components(mysql_con, sql_labels, issue, issue_key, project_key)
                num_versions = get_issue_components(mysql_con, sql_versions, issue, issue_key, project_key)
                num_fix_versions = get_issue_components(mysql_con, sql_fix_versions, issue, issue_key, project_key)
                updated1 = issue.fields.updated
                updated = datetime.datetime.strptime(updated1[:-5], '%Y-%m-%dT%H:%M:%S.%f')
                resolution_date1 = issue.fields.resolutiondate
                pull_request_url = get_issue_pull_request_url(issue, name_map)
                team = get_team(issue, name_map)
                attachment, is_attachment = get_attachment(issue)
                image1, is_image = get_image(issue)
                if resolution_date1 is not None:
                    resolution_date = datetime.datetime.strptime(resolution_date1[:-5], '%Y-%m-%dT%H:%M:%S.%f')
                else:
                    resolution_date = resolution_date1
                created1 = issue.fields.created
                created = datetime.datetime.strptime(created1[:-5], '%Y-%m-%dT%H:%M:%S.%f')
                summary = issue.fields.summary
                description = issue.fields.description
                acceptance_criteria = get_issue_acceptance_cri(issue, name_map)
                num_all_changes, num_changes_summary, num_changes_description, num_changes_acceptance_criteria,\
                num_changes_story_point, num_changes_sprint = get_changes_issue(mysql_con, sql_all_changes,
                                                                                sql_changes_summary,
                                                                                sql_changes_description,
                                                                                sql_changes_story_points,
                                                                                sql_changes_sprint, issue, issue_key,
                                                                                project_key, created, summary,
                                                                                description, acceptance_criteria)

                story_point = get_issue_story_points(issue, name_map)
                assignee = str(issue.fields.assignee)
                reporter = str(issue.fields.reporter)
                priority = get_issue_priority(project_name, issue, name_map)
                image1, is_image = get_image(issue)
                issue_id = issue.id
                creator = str(issue.fields.creator)
                main_table = (issue_key, issue_id, project_key, created, creator, reporter,
                              assignee, data_of_first_response(project_name, issue, name_map),
                              getattr(issue.fields, name_map['Epic Link']), issue_type, updated, priority,
                              issue.fields.progress.progress, issue.fields.progress.total, resolution,
                              resolution_date, status_name, issue.fields.status.description,
                              issue.fields.timeestimate, issue.fields.timeoriginalestimate, issue.fields.timespent,
                              attachment, is_attachment, pull_request_url, image1, is_image, str(team), story_point,
                              summary, description, acceptance_criteria, num_all_changes, num_issue_link_bug,
                              num_changes_summary, num_changes_description, num_changes_acceptance_criteria,
                              num_changes_story_point, num_comments, num_issue_links, num_commits,
                              num_sprints, num_sub_tasks, issue.fields.watches.watchCount,
                              num_worklog, num_versions, num_fix_versions, num_labels, num_components)
                # enter the data to the main table in SQL
                mycursor = mysql_con.cursor()
                try:
                    mycursor.execute(sql_main_table, main_table)
                    mysql_con.commit()
                    mycursor.close()
                except mysql.connector.IntegrityError:
                    print("ERROR: Kumquat already exists!")
                print(issue_key, issue_id, start)
    mysql_con.close()
