from PyQt5 import QtCore, QtGui, QtWidgets
from camera_real_time import Camera
import sys

C = Camera()  

class Ui_OtherWindow (object):
    
    
    def __init__(self,AgentId):
        self.Q=None
        self.AgentId=AgentId
        
    def setupUi(self, QMainWindow):
        
        self.Q = QMainWindow
        self.Q  = QtWidgets.QWidget()
        self.Q .resize(500, 450)
        self.Q .setWindowTitle('Facial-Recognation')
        self.Q .setWindowIcon(QtGui.QIcon('the-flash-season-4-confirmed.jpg'))
        
        
        self.pushButton= QtWidgets.QPushButton("Your Client", self.Q )
        self.pushButton.resize(100, 60)
        self.pushButton.move(210, 90)
        self.pushButton.clicked.connect(self.PhoneNumD)
        
        self.pushButton1 = QtWidgets.QPushButton("Start" , self.Q )
        self.pushButton1.hide()
        self.pushButton1.resize(100, 60)
        self.pushButton1.move(210, 100)
        self.pushButton1.setToolTip('click to start')
        self.pushButton1.clicked.connect(C.run)
        
        self.pushButton2 = QtWidgets.QPushButton("Exit", self.Q )
        self.pushButton2.hide()
        self.pushButton2.resize(100, 60)
        self.pushButton2.move(210, 240)
        self.pushButton2.setToolTip('click to Exit')
        self.pushButton2.clicked.connect(C.Exit)
        
        
        
        self.Q .show()
        sys.exit(app.exec_())
        
    def PhoneNumD (self) :
                self.Q.hide()
                ph , status = QtWidgets.QInputDialog(self.Q).getText(None,'Phone','Enter Your Client Phone Number :')
                #print(ph)
                #print(status)
                
                if status:
                        C.GetPhoneNum(ph)
                        C.GetAgentId(self.AgentId)
                        self.pushButton.hide()
                        self.Q.show()
                        self.pushButton1.show()
                        self.pushButton2.show()
                        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    OtherWindow = QtWidgets.QMainWindow()
    ui = Ui_OtherWindow()
    ui.setupUi(OtherWindow)
    OtherWindow.show()
    sys.exit(app.exec_())


