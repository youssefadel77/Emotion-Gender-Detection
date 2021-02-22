import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from CameraRealTime import Camera

C = Camera()        

class SecondWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(SecondWindow, self).__init__(parent)
        app = QtWidgets.QApplication(sys.argv)
        QMainWindow = QtWidgets.QWidget()
        QMainWindow.resize(500, 450)
        QMainWindow.setWindowTitle('Facial-Recognation')
        QMainWindow.setWindowIcon(QtGui.QIcon('the-flash-season-4-confirmed.jpg'))
        self.pushButton = QtWidgets.QPushButton("Start" , QMainWindow)
        self.pushButton.resize(100, 60)
        self.pushButton.move(210, 90)
        self.pushButton.setToolTip('click to start')
        self.pushButton.clicked.connect(C.run)
        self.pushButton = QtWidgets.QPushButton("Exit", QMainWindow)
        self.pushButton.resize(100, 60)
        self.pushButton.move(210, 190)
        self.pushButton.setToolTip('click to Exit')
        self.pushButton.clicked.connect(C.Exit)
        
        lable=QtWidgets.QLabel(QMainWindow,text='<h1>Enter Client Name</h1> ')
        lable.move(210, 290)
        self.textbox = QtWidgets.QLineEdit(QMainWindow)
        self.textbox.move(210, 320)
        self.textbox.resize(200,30)
        
        
        QMainWindow.show()
        sys.exit(app.exec_())



def main():
    app = QtWidgets.QApplication(sys.argv)
    main = SecondWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()