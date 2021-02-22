# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 04:35:22 2019

@author: Mont
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from OtherWindow import Ui_OtherWindow
from backend import back
#from camera_real_time import Camera
import sys

#CA = Camera() 

B=back()

class Ui_MainWindow(object):
    def __init__(self):
        self.textbox=None
        
    def openWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_OtherWindow(self.textbox)
        self.ui.setupUi(self.window)
        self.QMainWindow.hide()
        self.window.show()
                
        
    def setupUi(self, MainWindow):
        app = QtWidgets.QApplication(sys.argv)
        self.QMainWindow = QtWidgets.QWidget()
        self.QMainWindow.resize(500, 450)
        self.QMainWindow.setWindowTitle('Facial-Recognation')
        self.QMainWindow.setWindowIcon(QtGui.QIcon('the-flash-season-4-confirmed.jpg'))
        
        self.pushButton = QtWidgets.QPushButton("Sign In", self.QMainWindow)
        self.pushButton.resize(100, 60)
        self.pushButton.move(210, 100)
        self.pushButton.setToolTip('Click To Log In')
        self.pushButton.clicked.connect(self.loginForm)
        
        
        self.pushButton2 = QtWidgets.QPushButton("Connect To Your Work", self.QMainWindow)
        self.pushButton2.hide()
        self.pushButton2.resize(200, 60)
        self.pushButton2.move(210, 190)
        self.pushButton2.clicked.connect(self.openWindow)
        
        
        """
        
        lable=QtWidgets.QLabel(QMainWindow,text='<h1>Enter Your password</h1> ')
        lable.move(210, 150)
        self.textbox1 = QtWidgets.QLineEdit(QMainWindow)
        self.textbox1.move(210, 190)
        self.textbox1.resize(200,30)
        
        self.pushButton = QtWidgets.QPushButton("Sign In", QMainWindow)
        self.pushButton.resize(100, 60)
        self.pushButton.move(210, 290)
        self.pushButton.setToolTip('Click To Log In')
        self.pushButton.clicked.connect(self.openWindow)

        """
        self.QMainWindow.show()
        sys.exit(app.exec_())
        
    def loginForm (self) :
                self.QMainWindow.hide()
                textbox , status = QtWidgets.QInputDialog.getText(None,'ID','Enter Your ID :')
                #print(textbox)
                textbox2 , ok2 = QtWidgets.QInputDialog.getText(None,'Password','Enter Your Password :',QtWidgets.QLineEdit.Password)
                print(textbox2)
                print(ok2)
                if ok2:
                    print(B.login(textbox,textbox2))
                    if (B.login(textbox,textbox2)):
                        #CA.GetAgentId(textbox)
                        self.QMainWindow.show()
                        self.pushButton.hide()
                        self.pushButton2.show()
                    else:
                        self.QMainWindow.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

