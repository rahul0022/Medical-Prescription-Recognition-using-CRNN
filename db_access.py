from flask import Flask,render_template, request
from flask_mysqldb import MySQL
 
def get_alternates(conditions):
    appp = Flask(__name__)
    appp.config['MYSQL_HOST'] = 'localhost'
    appp.config['MYSQL_USER'] = 'root'
    appp.config['MYSQL_PASSWORD'] = ''
    appp.config['MYSQL_DB'] = 'medicines'
    mysql = MySQL(appp)
    with appp.app_context():
        cur = mysql.connection.cursor()
        conditions = str(conditions)
        if ('\'' in conditions):
            print("yes")
            conditions = conditions.replace("\'","\\\'")
        
        
        print(conditions)

        cur.execute('select distinct medicine_name from medicine where conditions like \'%'+ conditions +'%\' ')

        id2 = cur.fetchall()
        alternates = []
        for i in id2:
            alternates.append(i[0])
        print(len(id2))

        print(alternates)

        cur.close()

        return alternates

def get_conditions(medicine_name):
    appp = Flask(__name__)
    appp.config['MYSQL_HOST'] = 'localhost'
    appp.config['MYSQL_USER'] = 'root'
    appp.config['MYSQL_PASSWORD'] = ''
    appp.config['MYSQL_DB'] = 'medicines'
    mysql = MySQL(appp)
    with appp.app_context():
        cur = mysql.connection.cursor()

        cur.execute('select distinct conditions from medicine where medicine_name like \'%'+ medicine_name +'%\' ')

        cond = cur.fetchall()
        print(len(cond))
        if (len(cond) == 0):
            return []
        conditions = []
        for i in cond:
            i=i[0]
            if(len(i)>2 and str(i[-2])=='\\'):
                i=str(i[:-2])
            conditions.append(i)

        cur.close()
       

        return conditions[0]


