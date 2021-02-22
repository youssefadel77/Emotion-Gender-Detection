# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:18:33 2019

@author: Yahia
"""
from flask import Flask  , render_template
import mongodb as db
from bson import ObjectId


app = Flask(__name__)


def serial(dct):
    for k in dct:
        if isinstance(dct[k], ObjectId):
            dct[k] = 'objectid'   # 试过 dct[k] = str(dct[k]) 不行
    return dct


@app.route("/")
def agents():
    agents = db.find_Many('agents')
    return render_template('agents.html', agents = agents)  
   
    
def get_customers(reports):
    customers = []
    for report in reports:
        customer = db.find_one("customers",{"_id": report['customer_id']})
        customers.append(customer)
    return customers
    
    

@app.route("/profile/<spectial_id>")
def profile(spectial_id):
    agent = db.find_one('agents',{"spectial_id":spectial_id})
    final_reports = []
    reports = db.find_Many('reports',{'agent_id':agent["_id"]})
    for x in reports:
        final_reports.append(x)
    customers = get_customers(final_reports)
    count = len(final_reports)
    reports_data = [serial(item) for item in final_reports]
    return render_template('profile-page.html', reports = reports_data , agent = agent , customers = customers , count = count )  


"""
agent = db.find_one('agents',{"spectial_id":"A5B"})
print(agent["_id"])
reportes = db.find_Many('reports',{"agent_id": agent["_id"]})
customers = get_customers(reportes)

for x in reportes:
   print(x["customer_id"])

#5d04f14604beeaa221e3f6e4
"""


    
if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=80)