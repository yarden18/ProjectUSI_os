import pandas as pd
import mysql.connector
import re


if __name__ == '__main__':
    # open source: data_base_os end of _os to each table
    mysql_con = mysql.connector.connect(user='root', password='', host='localhost',
                                        database='data_base_os', auth_plugin='mysql_native_password',
                                        use_unicode=True)
    comments_data = pd.read_sql('SELECT * FROM comments_os', con=mysql_con)

    sql_add_columns_first = """alter table comments_os add column clean_comment MEDIUMTEXT"""
    sql_add_columns = """UPDATE comments_os SET clean_comment =%s
                       WHERE (issue_key=%s and chronological_number=%s)"""

    for i in range(0, len(comments_data)):
        body = comments_data['body'][i]
        issue_key = comments_data['issue_key'][i]
        chronological_number = comments_data['chronological_number'][i]
        if body is not None:
            clean_body = re.sub(r'<.+?>', "", body)
            clean_body = re.sub(r'&nbsp;', " ", clean_body)
            clean_body = re.sub(r"http\S+", "url", clean_body)
        else:
            clean_body = body
        mycursor = mysql_con.cursor()
        try:
            mycursor.execute(sql_add_columns, (clean_body, issue_key, int(chronological_number)))
            mysql_con.commit()
            mycursor.close()
        except mysql.connector.IntegrityError:
            print("ERROR: Kumquat already exists!")
