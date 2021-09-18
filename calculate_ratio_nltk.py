import nltk
import mysql
import pandas as pd
import mysql.connector



def choose_data_base(db_name):
    '''
    function which creates connection to the data base in SQL,
    input: server name
    output: SQL connection
    '''
    mysql_con = mysql.connector.connect(user='root', password='', host='localhost',
                                        database='{}'.format(db_name), auth_plugin='mysql_native_password',
                                        use_unicode=True)
    return mysql_con


if __name__ == "__main__":

    ########################################################################################################
    # ######################################## choose data base: ###########################################
    ########################################################################################################

    db_name = 'data_base_os'
    mysql_con = choose_data_base(db_name)

    # add 2 new columns in SQL table and update it by the calculation in the for loop

    data = pd.read_sql("SELECT * FROM main_table_os", con=mysql_con)
    sql_add_columns_first_different = """alter table main_table_os add column 
                                            num_different_words_all_text_sprint_new INT(11)"""
    sql_add_columns_first_ratio = """alter table main_table_os add column num_ratio_words_all_text_sprint_new FLOAT"""
    sql_updata_columns = """UPDATE main_table_os SET num_different_words_all_text_sprint_new =%s,
                            num_ratio_words_all_text_sprint_new =%s
                           WHERE (issue_key=%s)"""

    mycursor = mysql_con.cursor()
    try:
        mycursor.execute(sql_add_columns_first_different)
        mycursor.execute(sql_add_columns_first_ratio)
        mysql_con.commit()
        mycursor.close()
    except mysql.connector.IntegrityError:
        print("ERROR: Kumquat already exists!")

    # calculate the num of changes and the ration    
    for i in range(0, len(data)):
        issue_key = data['issue_key'][i]
        original_text = data['original_summary_description_acceptance_sprint'][i]
        text_last = data['summary_description_acceptance'][i]
        different = nltk.edit_distance(original_text.split(), text_last.split())
        length_text_original = len(original_text.split())
        if length_text_original == 0:
            length_text_original = 1
        ratio = different/length_text_original

        # update the results in the SQL table
        mycursor = mysql_con.cursor()
        try:
            mycursor.execute(sql_updata_columns, (int(different), float(ratio), issue_key))
            mysql_con.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
        print(issue_key)


