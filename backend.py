# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 15:40:19 2019

@author: yahia
"""
import mongodb as db
import collections

class back :
    def login(self, spectial_id , password):
        try:
            query={"spectial_id":spectial_id ,"password":password}
            agent = db.find_one("agents",query)
            if(agent):
                return True
            else:
                return False
        except Exception as e:
            print(e)
        
    
    #print(login("A5B","123456789"))
    def getMostInArray(self,list):
        count = 0
        value = ''
        list_collection = collections.Counter(list)
        for key in list_collection.keys():
            if list_collection[key] > count :
                count = list_collection[key]
                value = key
        return value
    
            
    def report(self , spectial_id , customer_number , gender , emotion):
        try:
            agent = db.find_one("agents",{"spectial_id":spectial_id})
            customer = db.find_one("customers",{"phoneNumber":customer_number});
            if(not customer):
                customer = db.insert({"name":"test","email":"test@gmail.com","phoneNumber":customer_number},'customers')
            if(agent):
                collection_gender = collections.Counter(gender)
                if(collection_gender['man'] < collection_gender['woman']):
                    new_gender = "Woman"
                else :
                    new_gender = "Man"
                new_emotion = self.getMostInArray(emotion[:int(len(emotion)*.5)]) + ' - ' + self.getMostInArray(emotion[int(len(emotion)*.5):])
                report = {"customer_id":customer['_id'] ,"agent_id":agent['_id'] , "gender":new_gender , "emotion": new_emotion }
                report = db.insert(report,"reports")
                return True
            else :
                return False
        except Exception as e:
            print(e)
            
    


#count = 0
#gender = ''
#list = ["man" , "woman" ,"man" , "woman" ,"man" , "woman" ,"man" , "woman" ,"woman" ,"woman","woman","woman","woman"]
#list_collection = collections.Counter(list)
#list_keys = []
#for key in list_collection.keys():
#  if list_collection[key] > count :
#      count = list_collection[key]
#      gender = key
 
    
#print(report("A5B","01111192675","Man","Angry"))
    
    
