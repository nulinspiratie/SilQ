import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from QLed import QLed
from time import sleep
#from . import start_gui

#import s!ilq
#silq.experiments_folder = r'C:\Users\Wyatt\Documents\GitHub\Experiments' #Wyatt's Desktop
#silq.experiments_folder = r'E:\Experiments' #Tallulah PC
#silq.experiments_folder = r'D:\Experiments' #Tallulah NI PC
#silq.initialize('Cavity Coupling',mode='measurement')!
#from . import start_gui

#smu = NISmu_4139(name='smu')

        
#def start_NI_PXIe_4139_gui(mini=False):
#    start_gui(NI_PXIe_4139_gui, 'smu_gui')


class LED(QAbstractButton):
    
    def __init__(self,color):
        super().__init__()
        if color == 'red':
            self.setIcon(QIcon('red_circle.jpg'))


class NI_PXIe_4139_gui(QWidget):
    '''
    A gui for the National Instruments NI_PXIe_4139 smu
    Requires PyQt5 (install via anaconda or pip) and QLed (install via pip)
    
    Wyatt Vine, October 2019
    
    Good documentation for PyQT5 at https://doc.qt.io/qt-5
    '''
    
    def __init__(self,smu):
        super().__init__()

        self._unit_prefixes = {'raise':{'nano':1E9,
                                            'micro':1E6,
                                            'milli':1E3,
                                            'unity':1},
                                'lower':{'nano':1E-9,
                                         'micro':1E-6,
                                         'milli':1E-3,
                                         'unity':1}}
    
        self.smu = smu
        self.mode = smu.mode() #'current_source' or "voltage_source", output mode of smu
        self.meas_unit_prefix = "milli" #the displayed unit prefix of the measured quantity
        self.limit_unit_prefix = "milli" #the displayed unit prefix of the smu's limit
        self.source_unit_prefix = "micro" #the displayed unit prefix of the smu's output
        self.current_limit = self.smu.current_limit()
        self.voltage_limit = smu.voltage_limit()
        
        if self.mode == 'current_source':
            self.source = smu.current() #in A
            self.meas = smu.voltage() #in V
        if self.mode == 'voltage_source':
            self.source = smu.voltage() #in V
            self.meas = smu.current() #in A
        self.in_compliance = smu.in_compliance() #True if the smu is in compliance, False if not
        
        #create the sub-widgets of the main gui display
        self._create_mode_display()
        self._create_source_display()
        self._create_limit_display()
        self._create_meas_display()
        self._create_compliance_display()
        self._create_on_off_display()
        
        #add each of the widgets to the main window
        self.main_layout = QGridLayout()
        self.main_layout.addWidget(self.mode_display,1,0)
        self.main_layout.addWidget(self.source_display,2,0)
        self.main_layout.addWidget(self.on_off_display,3,0)
        self.main_layout.addWidget(self.meas_display,1,1)
        self.main_layout.addWidget(self.limit_display,2,1)
        self.main_layout.addWidget(self.compliance_indicator_display,3,1)
        self.setLayout(self.main_layout)
        self.setWindowTitle('National Instruments PXIe-4139 SMU')
        self._init_signal_response()
        
    def _create_mode_display(self):
        self.mode_display = QGroupBox("SMU Mode")
        self.voltage_source_mode_button = QRadioButton("Source Voltage - Measure Current")
        self.current_source_mode_button = QRadioButton("Source Current - Measure Voltage")
        
        if self.mode == "voltage_source":
            self.voltage_source_mode_button.setChecked(True)
        elif self.mode == 'current_source':
            self.current_source_mode_button.setChecked(True)
        else:
            raise ValueError('SMU gui is in incorrect mode')
                
        layout = QVBoxLayout()
        layout.addWidget(self.voltage_source_mode_button)
        layout.addWidget(self.current_source_mode_button)
        self.mode_display.setLayout(layout)  
        
    def _create_source_display(self):
        
        if self.mode == 'current_source':
            self.source_display = QGroupBox('Current Source')
            radioButton1 = QRadioButton('nA')
            radioButton2 = QRadioButton('uA')
            radioButton3 = QRadioButton('mA')
            if self.source_unit_prefix == 'nano':
                radioButton1.setChecked(True)
            elif self.source_unit_prefix == 'micro':
                radioButton2.setChecked(True)
            elif self.source_unit_prefix == 'milli':
                radioButton3.setChecked(True)
            else:
                raise ValueError('SMU gui source_unit_prefix is invalid value')
            radioButton1.pressed.connect(lambda : self._change_source_unit_prefix('nano'))
            radioButton2.pressed.connect(lambda : self._change_source_unit_prefix('micro'))
            radioButton3.pressed.connect(lambda : self._change_source_unit_prefix('milli'))
                
        elif self.mode == "voltage_source":
            self.source_display = QGroupBox('Voltage Source')
            radioButton1 = QRadioButton('uV')
            radioButton2 = QRadioButton('mV')
            radioButton3 = QRadioButton('V')
            if self.source_unit_prefix == 'micro':
                radioButton1.setChecked(True)
            elif self.source_unit_prefix == 'milli':
                radioButton2.setChecked(True)
            elif self.source_unit_prefix == 'unity':
                radioButton3.setChecked(True)
            else:
                raise ValueError('SMU gui source_unit_prefix is invalid value')
            radioButton1.pressed.connect(lambda : self._change_source_unit_prefix('micro'))
            radioButton2.pressed.connect(lambda : self._change_source_unit_prefix('milli'))
            radioButton3.pressed.connect(lambda : self._change_source_unit_prefix('unity'))
        else:
            raise ValueError('SMU gui is in incorrect mode')

        self.source_display_box = QLineEdit(self.source_display)
        self.source_display_box.setText(str(self.source*self._unit_prefixes['raise'][self.source_unit_prefix]))
        self.source_display_box.editingFinished.connect(lambda : self._change_source(float(self.source_display_box.text())))
        layout = QGridLayout()
        layout.addWidget(self.source_display_box,1,0)
        layout.addWidget(radioButton1,1,1)
        layout.addWidget(radioButton2,2,1)
        layout.addWidget(radioButton3,3,1)
        self.source_display.setLayout(layout)
        
    def _create_on_off_display(self):
        
         self.on_off_display = QGroupBox()
         self.on_button = QRadioButton('Output On')
         self.off_button = QRadioButton('Output Off')
         self.off_button.setChecked(True)
        
         layout = QGridLayout()
         layout.addWidget(self.on_button)
         layout.addWidget(self.off_button)
         self.on_off_display.setLayout(layout)
        
    def _create_limit_display(self):
        
        self.limit_display_box = QLineEdit()
        
        if self.mode == "voltage_source":
            self.limit_display = QGroupBox('Current Limit')
            radioButton1 = QRadioButton('nA')
            radioButton2 = QRadioButton('uA')
            radioButton3 = QRadioButton('mA')
            if self.limit_unit_prefix == 'nano':
                radioButton1.setChecked(True)
            elif self.limit_unit_prefix == 'micro':
                radioButton2.setChecked(True)
            elif self.limit_unit_prefix == 'milli':
                radioButton3.setChecked(True)
            else:
                raise ValueError('SMU gui limit_unit_prefix is invalid value')
            radioButton1.pressed.connect(lambda : self._change_limit_unit_prefix('nano'))
            radioButton2.pressed.connect(lambda : self._change_limit_unit_prefix('micro'))
            radioButton3.pressed.connect(lambda : self._change_limit_unit_prefix('milli'))
            self.limit_display_box.setText(str(self.current_limit*self._unit_prefixes['raise'][self.limit_unit_prefix]))    
            
        elif self.mode == 'current_source':
            self.limit_display = QGroupBox('Voltage Limit')
            radioButton1 = QRadioButton('uV')
            radioButton2 = QRadioButton('mV')
            radioButton3 = QRadioButton('V')
            if self.limit_unit_prefix == 'micro':
                radioButton1.setChecked(True)
            elif self.limit_unit_prefix == 'milli':
                radioButton2.setChecked(True)
            elif self.limit_unit_prefix == 'unity':
                radioButton3.setChecked(True)
            else:
                raise ValueError('SMU gui limit_unit_prefix is invalid value')
            radioButton1.pressed.connect(lambda : self._change_limit_unit_prefix('micro'))
            radioButton2.pressed.connect(lambda : self._change_limit_unit_prefix('milli'))
            radioButton3.pressed.connect(lambda : self._change_limit_unit_prefix('unity'))
            self.limit_display_box.setText(str(self.voltage_limit*self._unit_prefixes['raise'][self.limit_unit_prefix]))
        
        else:
            raise ValueError('SMU gui is in incorrect mode')
            
        self.limit_display_box.editingFinished.connect(lambda : self._change_limit(float(self.limit_display_box.text())))
        layout = QGridLayout()
        layout.addWidget(self.limit_display_box,1,0)
        layout.addWidget(radioButton1,1,1)
        layout.addWidget(radioButton2,2,1)
        layout.addWidget(radioButton3,3,1)
        self.limit_display.setLayout(layout)

    def _create_meas_display(self):
        
        self.meas_display = QGroupBox("Measuring")
        
        if self.mode == 'voltage_source':
            radioButton1 = QRadioButton('nA')
            radioButton2 = QRadioButton('uA')
            radioButton3 = QRadioButton('mA')
            if self.meas_unit_prefix == 'nano':
                radioButton1.setChecked(True)
            elif self.meas_unit_prefix == 'micro':
                radioButton2.setChecked(True)
            elif self.meas_unit_prefix == 'milli':
                radioButton3.setChecked(True)
            else:
                raise ValueError('SMU gui meas_unit_prefix is invalid value')
            radioButton1.pressed.connect(lambda : self._change_meas_unit_prefix('nano'))
            radioButton2.pressed.connect(lambda : self._change_meas_unit_prefix('micro'))
            radioButton3.pressed.connect(lambda : self._change_meas_unit_prefix('milli'))
                
        elif self.mode == 'current_source':
            radioButton1 = QRadioButton('uV')
            radioButton2 = QRadioButton('mV')
            radioButton3 = QRadioButton('V')
            if self.meas_unit_prefix == 'micro':
                radioButton1.setChecked(True)
            elif self.meas_unit_prefix == 'milli':
                radioButton2.setChecked(True)
            elif self.meas_unit_prefix == 'unity':
                radioButton3.setChecked(True)
            else:
                raise ValueError('SMU gui meas_unit_prefix is invalid value')
            radioButton1.pressed.connect(lambda : self._change_meas_unit_prefix('micro'))
            radioButton2.pressed.connect(lambda : self._change_meas_unit_prefix('milli'))
            radioButton3.pressed.connect(lambda : self._change_meas_unit_prefix('unity'))
                
        else:
            raise ValueError('SMU gui is in incorrect mode')
        
        self.meas_display_box = QLineEdit() #create text box
        self.meas_display_box.setReadOnly(True)
        self.meas_display_box.setText(str(self.meas*self._unit_prefixes['raise'][self.meas_unit_prefix]))
        layout = QGridLayout()
        layout.addWidget(self.meas_display_box,1,0)
        layout.addWidget(radioButton1,1,1)
        layout.addWidget(radioButton2,2,1)
        layout.addWidget(radioButton3,3,1)
        self.meas_display.setLayout(layout)
        
    def _create_compliance_display(self):
        
        self.compliance_indicator_display = QGroupBox("In Compliance")
        if self.in_compliance == True:
            self.compliance_indicator = QLed(self, onColour=QLed.Green, shape=QLed.Circle)
        elif self.in_compliance == False:
            self.compliance_indicator = QLed(self, onColour=QLed.Red, shape=QLed.Circle)
        else:
            raise ValueError('Value of in_compliance is not valid')
        self.compliance_indicator.value = True
        layout = QGridLayout()
        layout.addWidget(self.compliance_indicator)
        self.compliance_indicator_display.setLayout(layout)
        
    def _init_signal_response(self):
        self.voltage_source_mode_button.clicked.connect(lambda x: self._change_mode('voltage_source','micro','micro','milli'))
        self.current_source_mode_button.clicked.connect(lambda x: self._change_mode('current_source','milli','milli','micro'))
        self.on_button.pressed.connect(lambda : self._change_on_off)
        self.off_button.pressed.connect(lambda : self._change_on_off)
        
    def _change_mode(self,*args):
        
        if args[0] == self.mode:
            return
        
        #turn off source
        self.off_button.setChecked(True)
        
        #update mode and unit multipliers
        self.mode = args[0]
        self.meas_unit_prefix = args[1]
        self.limit_unit_prefix = args[2]
        self.source_unit_prefix = args[3]
        
        #give reasonable values for new limits and source
        if self.mode == 'current_source':
            self.voltage_limit = 1E-3
            self.source = 1E-6
        else:
            self.voltage_limit = 1E-6
            self.source = 1E-3
        
        #redraw the widgets to reflect change of mode
        self.source_display.close()
        self._create_source_display()
        self.main_layout.addWidget(self.source_display,2,0)
        self.meas_display.close()
        self._create_meas_display()
        self.main_layout.addWidget(self.meas_display,1,1)
        self.limit_display.close()
        self._create_limit_display()
        self.main_layout.addWidget(self.limit_display,2,1)
        
    def _change_on_off(self,*args):
        print('test')
        
    def _change_limit(self,new_limit):
        if self.mode == 'voltage_source':
            self.current_limit = new_limit * self._unit_prefixes['lower'][self.limit_unit_prefix]
        else:
            self.voltage_limit = new_limit * self._unit_prefixes['lower'][self.limit_unit_prefix]
        
    def _change_limit_unit_prefix(self,new_limit_unit_prefix):        
        self.limit_unit_prefix = new_limit_unit_prefix
        self.limit_display_box.setText(str(self.limit*self._unit_prefixes['raise'][self.limit_unit_prefix]))
        
    def _change_meas_unit_prefix(self,new_meas_unit_prefix):
        self.meas_unit_prefix = new_meas_unit_prefix
        self.meas_display_box.setText(str(self.meas*self._unit_prefixes['raise'][self.meas_unit_prefix]))
        
    def _change_source(self,new_source):
        self.source = new_source * self._unit_prefixes['lower'][self.source_unit_prefix]
        if self.mode == 'voltage_source':
            smu.voltage(self.source)
        elif self.mode == 'current_source':
            smu.current(self.source)
        
    def _change_source_unit_prefix(self,new_source_unit_prefix):
        self.source_unit_prefix = new_source_unit_prefix
        self.source_display_box.setText(str(self.source * self._unit_prefixes['raise'][self.source_unit_prefix]))
        
    def _change_compliance_indicator(self):
        self.compliance_indicator_display.close()
        self._create_compliance_display()
        self.main_layout.addWidget(self.compliance_indicator_display,3,1)
        
    def _refresh(self):
        
        if smu.mode() == 'voltage_source':
            if self.mode != 'voltage_source':
                    self._change_mode('voltage_source','micro','micro','milli')
            self.source = smu.voltage()
            self.meas = smu.current()
            self.source_display_box.setText(str(self.source*self._unit_prefixes['raise'][self.source_unit_prefix]))
            self.meas_display_box.setText(str(self.meas*self._unit_prefixes['raise'][self.meas_unit_prefix]))
            if self.in_compliance != smu.in_compliance():
                self.in_compliance = smu.in_compliance()
                self._change_compliance_indicator()
                
        
        elif smu.mode() == 'current_source':
            if self.mode != 'current_source':    
                self._change_mode('current_source','milli','milli','micro')
            self.source = smu.current()
            self.meas = smu.voltage()
            self.source_display_box.setText(str(self.source*self._unit_prefixes['raise'][self.source_unit_prefix]))
            self.meas_display_box.setText(str(self.meas*self._unit_prefixes['raise'][self.meas_unit_prefix]))
            if self.in_compliance != smu.in_compliance():
                self.in_compliance = smu.in_compliance()
                self._change_compliance_indicator()
        
        else:
            raise ValueError('SMU is not in a mode recognized by the gui')
        
            
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = NI_PXIe_4139_gui(smu)
    ex.show()
#    for i in range(5):
#        ex._refresh()
#        ex.show()
#        sleep(1)
    sys.exit(app.exec_())
       

        