# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 10:29:32 2019

@author: Yahia
"""

#customer
"""
_id => objectId
name => string
email => string
phoneNumber => String
"""

#agent
"""
_id => objectId
spectial_id => string
name => string
email => string
password => string
reports => array objectIds
"""

#report
"""
_id => objectId
customer_id => objectId
agent_id => objectId
gender => String
emotion => string
"""


#from pymongo.objectid import ObjectId   
import pymongo

myclient = pymongo.MongoClient("mongodb://yahiaIB:MOHAMED203050@ds129914.mlab.com:29914/mydatabase")

mydb = myclient["mydatabase"]


def insert(data , tabelName):
    try:
        tabel = mydb[tabelName]
        tabel.insert_one(data)
        return data
    except Exception as e:
        print(e)
        
    
def delete(tableName , query):
    try:
        tabel = mydb[tableName]
        tabel.delete_one(query)
        return
    except Exception as e:
        print(e)

def find_Many(tableName , query = {} ):
    try:
        tabel = mydb[tableName]
        data = tabel.find(query)
        return data
    except Exception as e:
        print(e)
    
def find_one(tableName , query = {}):
    try:
        tabel = mydb[tableName]
        data = tabel.find_one(query)
        return data
    except Exception as e:
        print(e)




#mydict = insert( { "name": "mont", "email": "yahiaIbrahim300@gmail.com" , "phoneNumber":"01111192675" } , 'customers')

#agent1 = insert( { "name": "yahia", "email": "yahiaIbrahim300@gmail.com" , "phoneNumber":"01111192675" , "password":"123456789", "spectial_id":"A5B" } , 'agents')


#report1 = insert( { "gender": "Man", "emotion": "Angry" ,"customer_id": mydict['_id'] , "agent_id": agent1['_id'] } , 'reports')

#print(find_one('customers',{'_id':mydict['_id']}))

#delete({"name":"John"} ,'customers' )


#for x in find_Many('reports'):
#   print(x)
