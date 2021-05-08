import pandas as pd
import mysql.connector


if __name__ == '__main__':

    mysql_con = mysql.connector.connect(user='root', password='', host='localhost',
                                        database='data_base_os', auth_plugin='mysql_native_password',
                                        use_unicode=True)

    sql_add_columns_label_only_empty = """alter table features_labels_table_os2 add column label_is_empty INT"""
    sql_update_columns_label_only_empty = """UPDATE features_labels_table_os2 SET label_is_empty= 
                                            CASE
                                                WHEN original_description_sprint='TBD' or original_description_sprint='TODO' 
                                                    or original_description_sprint='<p>TBD</p>\r\n' 
                                                    or original_description_sprint='<p>c</p>\r\n' or 
                                                    original_description_sprint='<p>...</p>\r\n' or 
                                                    original_description_sprint='tbd'  or original_description_sprint='todo' 
                                                    or original_description_sprint is null then 1
                                                    ELSE 0
                                            END"""

    mycursor = mysql_con.cursor()
    try:
        mycursor.execute(sql_add_columns_label_only_empty)
        mycursor.execute(sql_update_columns_label_only_empty)
        mysql_con.commit()
        mycursor.close()
    except mysql.connector.IntegrityError:
        print("ERROR: Kumquat already exists!")


