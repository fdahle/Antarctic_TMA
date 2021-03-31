"""
This algorithm does the following things:
- connecting to the database
- reading/writing from the database

connection can be done with a accdb database (used for the initial development)
and a psql database
"""

import pyodbc #to communicate with the access database
import pandas as pd

#which connection should be used
connection_type = "MS_Access"

#specify location of database
access_file_paths = ["C:/Users/Felix/Google Drive/antarctic_tma.accdb"]



class Connector:

    def __init__(self):

        if connection_type == "MS_Access":

            #connect to database and save cursor
            for elem in access_file_paths:
                try:
                    self.conn = pyodbc.connect(
                        r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};' +
                        'DBQ=' + elem + ';')
                    break
                except:
                    continue
            self.cursor = self.conn.cursor()

    #build the Filter (WHERE ..) string
    def build_filter(self, filters):

        filterString = ""

        #build the OR criterias
        for elem in filters:

            filterString = filterString + "("

            #build the AND criterias
            for key in elem:
                filterVal = elem[key]
                #not numeric filtervalues must be treated different
                isNumber = True
                try:
                    _  = int(filterVal)
                except:
                    isNumber = False

                #some filtervalues must be handled seperatly
                if filterVal == "NULL":
                    filterVal = " IS NULL"
                elif filterVal == "NOT NULL":
                    filterVal = " IS NOT NULL"
                elif filterVal == "TRUE()":
                    filterVal = "= TRUE"
                elif filterVal == "FALSE()":
                    filterVal = "= FALSE"
                elif not isNumber:
                    filterVal = "='" + filterVal + "'"
                else:
                    filterVal = "=" + str(filterVal)

                filterString = filterString  + key + filterVal + " AND "

            filterString = filterString[:-5] + ")"
            filterString = filterString + " OR "

        #add the WHERE if there are filters active
        if filterString != "":
            filterString = filterString[:-4]
            filterString = " WHERE (" + filterString + ")"

        return filterString

    #extract data from database and return columns & data
    def get_data(self, tables, fields="*", filters=None, join=None):

        #build fields string based on fields
        if fields != "*":
            fieldsString = ""

            #for every table
            for i, tableName in enumerate(tables):

                #for every field in table
                for elem in fields[i]:
                    fieldsString = fieldsString + tableName + "." + elem + \
                    " AS " + elem + ","

            fieldsString = fieldsString[:-1]
        else:
            fieldsString = "*"

        #if multiple tables are there join them
        tableString = ""
        for elem in tables:
            tableString = tableString + elem + " INNER JOIN "
        tableString = tableString[:-12]

        #build the string that tells how multiple tables are joined
        joinString = ""
        if join is not None:
            joinString = " ON "
            for tableName in tables:
                joinString = joinString + tableName + "." + join[0] + " = "
            joinString = joinString [:-3]

        # build sql string
        sqlString = "SELECT " + fieldsString + " FROM " + tableString + joinString

        #build the filterString
        if filters != None:
            filterString = self.build_filter(filters)
            sqlString = sqlString + filterString

        #get Data
        dataFrame = pd.read_sql(sqlString, self.conn)

        return dataFrame

    #count the number of entries in a table
    def count_data(self, table, filters=None):

        # build sql string
        sqlString = "SELECT COUNT(*) FROM " + table

        #filters is an array of dict:
        #- everything in a dict is combined with AND
        #- every dict is combined with an OR

        #build the filterString
        filterString = self.build_filter(filters)

        if filterString != "":
            sqlString = sqlString + filterString

        #get data
        self.cursor.execute(sqlString)

        # get the results
        count = self.cursor.fetchall()[0][0]

        return count

    #add data
    def add_data(self, table, attributes):

        valuesString = ""
        attributeString = ""

        #build the strings
        for key in attributes:

            valuesString = valuesString + key + ","

            attr = attributes[key]

            #not numeric filtervalues must be treated different
            if not attr.isnumeric():
                attr = "'" + attr + "'"

            attributeString = attributeString + attr + ","

        #add date
        valuesString = valuesString + "last_change"
        attributeString = attributeString + "Date()"

        #create sqlString
        sqlString = "INSERT INTO " + table + " (" + valuesString + ") " + \
                    "VALUES (" + attributeString + ")"

        self.cursor.execute(sqlString)
        self.conn.commit()

    #edit data
    def edit_data(self, table, id, attributes):

        sqlString = "UPDATE " + table + " SET "

        #create the updatestring that containts the attributes and their updated values
        updateString = ""
        for key in attributes:

            attr = attributes[key]

            #not numeric filtervalues must be treated different
            isNumber = True
            try:
                _  = int(attr)
            except:
                isNumber = False

            if attr is None:
                attr = "'NULL'"
            elif attr == "DATE()":
                attr = "Date()"
            elif attr == "NULLDATE()":
                attr = "NULL"
            elif attr == "TRUE()":
                attr = "TRUE"
            elif attr == "FALSE()":
                attr = "FALSE"
            elif not isNumber:
                attr = "'" + attr + "'"

            updateString = updateString + key + "=" + str(attr) + ","

        updateString = updateString[:-1]
        updateString = updateString + ",last_change=Date()"


        filterId = list(id.keys())[0]
        filterString = " WHERE " + filterId + "='" + id[filterId] + "'"

        sqlString = sqlString + updateString + filterString

        self.cursor.execute(sqlString)
        self.conn.commit()
