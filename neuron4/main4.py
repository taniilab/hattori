#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ver1.xx    畳み込み積分＆積分発火型
ver2.xx    積分発火型
ver3.xx    ver2にGUIを実装
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random
import time
from neuron4 import Neuron4

class Main(QtGui.QWidget):
    deltatime = 0.001
    phightime = 0.002
    numneu = 10
    vth = -55
    vrm = -70
    tau = 0.01
    phigh = 10
    e_syn = 0
    def_w = 0.7
    sigmoid_gain = 0.3
    ab_refacttime = 0.005
    ab_weak = 0.05
    simtime = 10
   
    def __init__(self, parent=None):
        super(Main, self).__init__()
        
        #GUI
        numneuLabel = QtGui.QLabel('ニューロン数')
        vthLabel = QtGui.QLabel('スレッショルド電圧[mV]')
        vrmLabel = QtGui.QLabel('静止膜電位[mV]')
        tauLabel = QtGui.QLabel('時定数[t]')
        deltatimeLabel = QtGui.QLabel('単位時間[t]')
        phighLabel = QtGui.QLabel('発火時電圧[mV]')
        phightimeLabel = QtGui.QLabel('発火時間[t]')
        e_synLabel = QtGui.QLabel('シナプス逆転電位')
        def_wLabel = QtGui.QLabel('シナプス結合荷重(初期値)')
        sigmoid_gainLabel = QtGui.QLabel('シグモイドゲイン')
        ab_refacttimeLabel = QtGui.QLabel('不応期時間[t]')
        ab_weakLabel = QtGui.QLabel('不応期時の上がりにくさ[0~1]')
        simtimeLabel = QtGui.QLabel('シミュレーション時間[t]')

        self.numneuEdit = QtGui.QLineEdit()
        self.vthEdit = QtGui.QLineEdit()
        self.vrmEdit = QtGui.QLineEdit() 
        self.tauEdit = QtGui.QLineEdit()
        self.deltatimeEdit = QtGui.QLineEdit()
        self.phighEdit = QtGui.QLineEdit()
        self.phightimeEdit = QtGui.QLineEdit()       
        self.e_synEdit = QtGui.QLineEdit()
        self.def_wEdit = QtGui.QLineEdit()
        self.sigmoid_gainEdit = QtGui.QLineEdit()
        self.ab_refacttimeEdit = QtGui.QLineEdit()
        self.ab_weakEdit = QtGui.QLineEdit()
        self.simtimeEdit = QtGui.QLineEdit()
                
        self.numneuEdit.setText(str(Main.numneu))
        self.vthEdit.setText(str(Main.vth))
        self.vrmEdit.setText(str(Main.vrm))
        self.tauEdit.setText(str(Main.tau))
        self.deltatimeEdit.setText(str(Main.deltatime))
        self.phighEdit.setText(str(Main.phigh))
        self.phightimeEdit.setText(str(Main.phightime))
        self.e_synEdit.setText(str(Main.e_syn))
        self.def_wEdit.setText(str(Main.def_w))
        self.sigmoid_gainEdit.setText(str(Main.sigmoid_gain))
        self.ab_refacttimeEdit.setText(str(Main.ab_refacttime))
        self.ab_weakEdit.setText(str(Main.ab_weak))
        self.simtimeEdit.setText(str(Main.simtime))

        startBtn = QtGui.QPushButton('START')
        saveBtn = QtGui.QPushButton('SAVE')

        grid = QtGui.QHBoxLayout()
        
        self.groupBox = QtGui.QGroupBox("パラメータ設定")
        vbox = QtGui.QVBoxLayout()

        layout1 = QtGui.QHBoxLayout()
        layout1.addWidget(numneuLabel)
        layout1.addWidget(self.numneuEdit)
        layout2 = QtGui.QHBoxLayout()
        layout2.addWidget(vthLabel)
        layout2.addWidget(self.vthEdit)
        layout3 = QtGui.QHBoxLayout()
        layout3.addWidget(vrmLabel)
        layout3.addWidget(self.vrmEdit)
        layout4 = QtGui.QHBoxLayout()
        layout4.addWidget(tauLabel)
        layout4.addWidget(self.tauEdit)
        layout5 = QtGui.QHBoxLayout()
        layout5.addWidget(deltatimeLabel)
        layout5.addWidget(self.deltatimeEdit)
        layout6 = QtGui.QHBoxLayout()
        layout6.addWidget(phighLabel)
        layout6.addWidget(self.phighEdit)
        layout7 = QtGui.QHBoxLayout()
        layout7.addWidget(phightimeLabel)
        layout7.addWidget(self.phightimeEdit)
        layout8 = QtGui.QHBoxLayout() 
        layout8.addWidget(e_synLabel)
        layout8.addWidget(self.e_synEdit)
        layout9 = QtGui.QHBoxLayout() 
        layout9.addWidget(def_wLabel)
        layout9.addWidget(self.def_wEdit)
        layout10 = QtGui.QHBoxLayout()
        layout10.addWidget(sigmoid_gainLabel)
        layout10.addWidget(self.sigmoid_gainEdit)
        layout11 = QtGui.QHBoxLayout()        
        layout11.addWidget(ab_refacttimeLabel)
        layout11.addWidget(self.ab_refacttimeEdit)
        layout12 = QtGui.QHBoxLayout()
        layout12.addWidget(ab_weakLabel)
        layout12.addWidget(self.ab_weakEdit)
        layout13 = QtGui.QHBoxLayout()
        layout13.addWidget(simtimeLabel)
        layout13.addWidget(self.simtimeEdit)
        layout14 = QtGui.QVBoxLayout()
        layout14.addWidget(startBtn)
        layout14.addWidget(saveBtn)
        
        vbox.addLayout(layout1)
        vbox.addLayout(layout2)
        vbox.addLayout(layout3)
        vbox.addLayout(layout4)
        vbox.addLayout(layout5)
        vbox.addLayout(layout6) 
        vbox.addLayout(layout7)
        vbox.addLayout(layout8)
        vbox.addLayout(layout9)
        vbox.addLayout(layout10)
        vbox.addLayout(layout11)
        vbox.addLayout(layout12)
        vbox.addLayout(layout13)
        vbox.addLayout(layout14)
        
        self.groupBox.setLayout(vbox)
        grid.addWidget(self.groupBox)

        self.groupBox2 = QtGui.QGroupBox("プロット")
        mainbox = QtGui.QVBoxLayout()        
 
        #ニューロン作成
        self.neuron1 = Neuron4(int(self.numneuEdit.text()),
                            float(self.vthEdit.text()),
                            float(self.vrmEdit.text()),
                            float(self.tauEdit.text()), 
                            float(self.deltatimeEdit.text()), 
                            float(self.phightimeEdit.text()), 
                            float(self.phighEdit.text()),
                            float(self.e_synEdit.text()), 
                            float(self.def_wEdit.text()), 
                            float(self.sigmoid_gainEdit.text()), 
                            float(self.ab_refacttimeEdit.text()), 
                            float(self.ab_weakEdit.text()),
                            float(self.simtimeEdit.text()))
        
        #シナプス結合荷重(0.8)
        w = np.ones((Main.numneu, Main.numneu)) * Main.def_w
        for i in range(0, Main.numneu):
            w[i][i] = 0
            w[0][0] = 1
            #w[1][1] = 1    
        self.neuron1.set_weight(w)
       
        #タブごとにグラフを作成
        qtab = QtGui.QTabWidget()
        self.tab1 = TabwaveWidget(self.neuron1)
        self.tab2 = TablasterWidget(self.neuron1)
        qtab.addTab(self.tab1, '波形')
        qtab.addTab(self.tab2, 'ラスター')                  
        
        #スライダー(グラフのズーム・アウト)
        self.sliderh = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sliderh.setRange(1, 100)  
        self.sliderh.setValue(10)  
        
        self.sliderv = QtGui.QSlider(QtCore.Qt.Vertical) 
        self.sliderv.setRange(1, 100)  
        self.sliderv.setValue(10) 

        hsbox = QtGui.QHBoxLayout()
        hsbox.addWidget(self.sliderh)
        hsbox.setAlignment(self.sliderh, QtCore.Qt.AlignVCenter)

        vsbox = QtGui.QVBoxLayout()
        vsbox.addWidget(self.sliderv)
        vsbox.setAlignment(self.sliderv, QtCore.Qt.AlignHCenter)    
        
        pltbox = QtGui.QHBoxLayout()
        pltbox.addWidget(qtab)
        pltbox.addLayout(vsbox)
        
        mainbox.addLayout(pltbox)
        mainbox.addLayout(hsbox)
        self.groupBox2.setLayout(mainbox)

        #比率調整
        sizePolicy1 = self.groupBox.sizePolicy()
        sizePolicy2 = self.groupBox.sizePolicy()
        sizePolicy1.setHorizontalStretch(2)
        sizePolicy2.setHorizontalStretch(7)
        self.groupBox.setSizePolicy(sizePolicy1)
        self.groupBox2.setSizePolicy(sizePolicy2)

        grid.addWidget(self.groupBox2)        
        self.setLayout(grid) 

        #***シグナル***
        startBtn.clicked.connect(self.start_simulation)
        saveBtn.clicked.connect(self.save_settings)
        self.sliderh.valueChanged.connect(self.on_slideh)        
        self.sliderv.valueChanged.connect(self.on_slidev)
        self.sliderh.sliderReleased.connect(self.slidend)
        self.sliderv.sliderReleased.connect(self.slidend)
        
        #ウィンドウ調整       
        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('にゅうろん！')    
        self.resize(3000, 1800)  
        #最前面
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)      
        self.show()
        
    #***以下スロット***
    #スタートボタン
    def start_simulation(self):
        
        #データ設定
        self.neuron1.set_neuron_palameter(int(self.numneuEdit.text()),
                                    float(self.vthEdit.text()),
                                    float(self.vrmEdit.text()),
                                    float(self.tauEdit.text()), 
                                    float(self.deltatimeEdit.text()), 
                                    float(self.phightimeEdit.text()), 
                                    float(self.phighEdit.text()),
                                    float(self.e_synEdit.text()),  
                                    float(self.def_wEdit.text()), 
                                    float(self.sigmoid_gainEdit.text()), 
                                    float(self.ab_refacttimeEdit.text()), 
                                    float(self.ab_weakEdit.text()),
                                    float(self.simtimeEdit.text()))
        
        w = np.ones((self.neuron1.numneu, self.neuron1.numneu)) * float(self.def_wEdit.text())
        for i in range(0, self.neuron1.numneu):
            w[i][i] = 0
            w[0][0] = 1
            #w[1][1] = 1    
        self.neuron1.set_weight(w)        
        
        self.tab1.replot(self.neuron1)
        self.tab2.replot(self.neuron1)
        
    #figure横サイズ
    def on_slideh(self):
        self.tab1.fig.set_figwidth(float(self.sliderh.value()))

        #キャンバス上にFigureを再セットしないとスクロールバーの長さが変更されない        
        self.tab1.canvas = FigureCanvas(self.tab1.fig)
        self.tab1.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.tab1.scroll.setWidget(self.tab1.canvas)

        #ツールバーも再セット        
        self.tab1.navi_toolbar.setVisible(False)        
        self.tab1.layout.removeWidget(self.tab1.navi_toolbar)        
        self.tab1.navi_toolbar = NavigationToolbar(self.tab1.canvas, self)
        
         
    #figure縦サイズ
    def on_slidev(self):
        self.tab1.fig.set_figheight(float(self.sliderv.value()))
        
        #キャンバス上にFigureを再セットしないとスクロールバーの長さが変更されない        
        self.tab1.canvas = FigureCanvas(self.tab1.fig)
        self.tab1.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.tab1.scroll.setWidget(self.tab1.canvas)
        
        #ツールバーも再セット
        self.tab1.navi_toolbar.setVisible(False)
        self.tab1.layout.removeWidget(self.tab1.navi_toolbar)        
        self.tab1.navi_toolbar = NavigationToolbar(self.tab1.canvas, self)

    #スライドバー開放時にツールバーを再セット
    def slidend(self):
        self.tab1.layout.addWidget(self.tab1.navi_toolbar)
                
        
    def save_settings(self):
        print("てすと")    
        

#波形表示  
class TabwaveWidget(QtGui.QWidget):

    #初期描画
    def __init__(self, neuron):
        super().__init__()     
        self.plot(neuron)
        
        #グラフをタブにセット        
        self.canvas = FigureCanvas(self.fig)
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        self.layout = QtGui.QVBoxLayout()
        self.scroll = QtGui.QScrollArea()
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setWidget(self.canvas)        
        self.layout.addWidget(self.scroll)
        self.layout.addWidget(self.navi_toolbar)
        self.setLayout(self.layout)
    
    #グラフを更新する場合
    def replot(self, neuron):
        self.layout.removeWidget(self.navi_toolbar)           
        plt.close()
        self.layout.removeWidget(self.scroll)
        self.plot(neuron)
        
        #再セット
        if neuron.numneu < 11:
            self.canvas = FigureCanvas(self.fig)
            self.navi_toolbar = NavigationToolbar(self.canvas, self)
            self.scroll = QtGui.QScrollArea()
            self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            self.scroll.setWidget(self.canvas)        
            self.layout.addWidget(self.scroll)    
            self.layout.addWidget(self.navi_toolbar)
        
    #メイン処理
    def plot(self, neuron):
        
        self.fig, self.ax = plt.subplots(nrows = neuron.numneu, figsize=(16, 32))
        
        #軸
        self.t = np.arange(0, (neuron.simtime*neuron.deltatime), neuron.deltatime)
        self.vplt = np.ones((neuron.numneu, np.size(self.t))) * (-70) 

        #グラフ設定
        for i in range(0, neuron.numneu):
            if neuron.numneu == 1:
                self.ax.set_xlim((self.t.min(), self.t.max()))
                self.ax.set_ylim(neuron.vrm[0]-20, neuron.phigh[0]+10)
                self.ax.grid(which='major',color='thistle',linestyle='-')
                self.ax.spines["top"].set_color("indigo")
                self.ax.spines["bottom"].set_color("indigo")
                self.ax.spines["left"].set_color("indigo")
                self.ax.spines["right"].set_color("indigo")

            else:            
                self.ax[i].set_xlim((self.t.min(), self.t.max()))
                self.ax[i].set_ylim(neuron.vrm[0]-20, neuron.phigh[0]+10)
                self.ax[i].grid(which='major',color='thistle',linestyle='-')
                self.ax[i].spines["top"].set_color("indigo")
                self.ax[i].spines["bottom"].set_color("indigo")
                self.ax[i].spines["left"].set_color("indigo")
                self.ax[i].spines["right"].set_color("indigo")
        
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.05, bottom=0.03)

        self.fig.text(0.5, 0.01, 'time [t]', fontsize=16, ha='center', va='center')        
        self.fig.text(0.01, 0.5, 'membrane potential [mV]', fontsize=16, ha='center', va='center', rotation='vertical')        
        self.fig.patch.set_facecolor('whitesmoke')   
        
        #初期化
        self.lines = []
        for i in range(0, neuron.numneu):
            self.lines.append([])
            if neuron.numneu == 1:
                self.lines, = self.ax.plot(self.t, self.vplt[i], color="indigo")
            else:
                self.lines[i], = self.ax[i].plot(self.t, self.vplt[i], color="indigo")
        
        #信号伝達 
        self.start_time = time.time()
        for i in range(0, np.size(self.t)):      
            self.vplt[0:neuron.numneu, i] = neuron.propagation("sigmoid")
        self.elapsed_time = time.time() - self.start_time
        print("elapsed_time:{0}".format(self.elapsed_time) + "[sec]")       
       
        #プロット
        if neuron.numneu < 11:
            for i in range(0, neuron.numneu):        
                base = np.ones(np.size(self.t)) * (-100)            
                if neuron.numneu == 1:
                    self.lines.set_data(self.t, self.vplt)
                    self.ax.fill_between(self.t, self.vplt, base, facecolor="thistle", alpha=0.2)           
                else:
                    self.lines[i].set_data(self.t, self.vplt[i])              
                    self.ax[i].fill_between(self.t, self.vplt[i], base, facecolor="thistle", alpha=0.2)                
        

        
#ラスター表示
class TablasterWidget(QtGui.QWidget):
    
        #初期描画
    def __init__(self, neuron):
        super().__init__()     
        self.plot(neuron)
        
        #グラフをタブにセット        
        self.canvas = FigureCanvas(self.fig)
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        self.layout = QtGui.QVBoxLayout()     
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.navi_toolbar)
        self.setLayout(self.layout)
    
    #グラフを更新する場合
    def replot(self, neuron):
        self.layout.removeWidget(self.navi_toolbar)           
        plt.close()
        self.layout.removeWidget(self.canvas)
        self.plot(neuron)
        
        #再セット
        self.canvas = FigureCanvas(self.fig)
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.canvas)    
        self.layout.addWidget(self.navi_toolbar)

    #メイン処理
    def plot(self, neuron):
        
        self.fig, self.ax = plt.subplots(nrows = 1, figsize=(10, 15))
        self.t = np.arange(0, (neuron.simtime*neuron.deltatime), neuron.deltatime)    

        #グラフ設定
        self.ax.set_xlabel("time[s]")  
        self.ax.set_ylabel("Neuron")
        self.ax.set_title("raster plot")
        self.ax.set_yticks(np.arange(0, (neuron.numneu + 1), 1.0))
        self.ax.set_xlim((self.t.min(), self.t.max()))
        self.ax.set_ylim([0, neuron.numneu + 1])
        self.ax.grid(which='major',color='thistle',linestyle='-')
        self.ax.spines["top"].set_color("indigo")
        self.ax.spines["bottom"].set_color("indigo")
        self.ax.spines["left"].set_color("indigo")
        self.ax.spines["right"].set_color("indigo")
        self.fig.patch.set_facecolor('whitesmoke')   

        #棒作成
        for i in range(0, neuron.numneu):
            for j in range(0, np.size(self.t)):
                if neuron.raster[i][j] != 0:
                    self.ax.vlines(self.t[j], neuron.raster[i][j] - 0.49, neuron.raster[i][j] + 0.49, "indigo")
              
def main():
    app = QtGui.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Yu Gothic UI"))
    main = Main()
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()
