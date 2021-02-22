import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from GUI import SecondWindow 

Second = SecondWindow()
class FirstWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(FirstWindow, self).__init__(parent)
        app = QtWidgets.QApplication(sys.argv)
        QMainWindow = QtWidgets.QWidget()
        QMainWindow.resize(500, 450)
        QMainWindow.setWindowTitle('Facial-Recognation')
        QMainWindow.setWindowIcon(QtGui.QIcon('the-flash-season-4-confirmed.jpg'))


        lable=QtWidgets.QLabel(QMainWindow,text='<h1>Enter Your Name</h1> ')
        lable.move(210, 50)
        self.textbox = QtWidgets.QLineEdit(QMainWindow)
        self.textbox.move(210, 90)
        self.textbox.resize(200,30)
        
        lable=QtWidgets.QLabel(QMainWindow,text='<h1>Enter Your password</h1> ')
        lable.move(210, 150)
        self.textbox = QtWidgets.QLineEdit(QMainWindow)
        self.textbox.move(210, 190)
        self.textbox.resize(200,30)
        
        self.pushButton = QtWidgets.QPushButton("Sign In", QMainWindow)
        self.pushButton.resize(100, 60)
        self.pushButton.move(210, 290)
        self.pushButton.setToolTip('Click To Log In')
        self.pushButton.clicked.connect(Second.main)

        
        QMainWindow.show()
        sys.exit(app.exec_())

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = First()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()