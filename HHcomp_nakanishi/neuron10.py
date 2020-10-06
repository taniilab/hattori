# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:26:04 2017

@author: Nakanishi
"""

# シナプス分割モデル
# 細胞体へのフィードバックなし
# ノイズの平均と分散を操作できるように変更
# ノイズが面積に応じて入ると仮定
import numpy as np
import random


class Neuron():

    def __init__(self, dt, simtime, numneu, vres, Pmax, Pnoise, ave, gH, numsyn):
        self.dt = dt   # 時間分解能
        self.simtime = simtime  # シミュレーション時間
        self.cycle = simtime / dt  # 試行回数
        self.numneu = numneu  # ニューロン数
        self.vin = np.ones((self.numneu, self.cycle)) * vres  # 電圧初期設定
        self.vde = np.ones((self.numneu, self.cycle)) * vres
        self.numsyn = numsyn
        self.dsyn = 1.5 / self.dt
        # np.onesで全要素1の行列
        self.dvin = np.ones(self.numneu)
        self.dvde = np.ones(self.numneu)
        self.C = 1.0  # 時定数
        self.vres = vres
        # synapse
        self.Pmax = np.ones(self.numsyn) * Pmax
        self.tau = np.ones(self.numsyn) * (2.5 / self.dt)
        self.syntm = np.ones((self.numneu, self.cycle)) * (-100)
        self.gsyn = np.zeros(((self.numneu, self.numneu, self.numsyn)))
        self.d = np.zeros(self.numsyn)
        self.vsyn = 0
        self.isynd = np.zeros((self.numneu, self.cycle))
        self.Pmax = Pmax
        self.vth = -30
        # noise
        self.pnoise = Pnoise  # 平均
        self.ave = ave  # 分散
        self.noise = 0
        # 各チャネルの静止膜電位 mV
        self.vresna = 55
        self.vresk = -90
        self.vresl = -50
        self.vresca = 120
        self.vresH = -40
        # 各チャネルのコンダクタンス mS
        self.gna = 45
        self.gk = 18
        self.gm = 0.4
        self.gl = 0.05
        self.gnap = 0.14
        self.gks = 9
        self.gca = 1
        self.gkca = 15
        self.gH = gH
        # コンパートメントのおまけ
        self.As = 33 * 0.15
        self.Ad = 33 * 0.85
        self.Rc = 15  # Morm
        self.taumM = np.ones(self.numneu)
        self.minfM = np.ones(self.numneu)
        self.minfnap = np.zeros((self.numneu, self.cycle))
        self.minfks = np.ones(self.numneu)
        self.taumks = np.ones(self.numneu)
        self.hinf = np.ones(self.numneu)
        self.tauh = np.ones(self.numneu)
        self.minfca = np.zeros((self.numneu, self.cycle))
        self.Hinf = np.ones(self.numneu)
        self.tauH = np.ones(self.numneu)
        self.ca = np.zeros((self.numneu, self.cycle))
        self.dca = np.ones(self.numneu)
        self.Dk = 30
        # self.input = np.ones((self.numneu, self.cycle)
        self.nowstep = 0  # 行列の行
        # チャネルの開閉率初期値要勉強
        self.m = np.zeros((self.numneu, self.cycle))
        self.h = np.zeros((self.numneu, self.cycle))
        self.n = np.zeros((self.numneu, self.cycle))
        self.M = np.zeros((self.numneu, self.cycle))
        self.mks = np.zeros((self.numneu, self.cycle))
        self.hks = np.zeros((self.numneu, self.cycle))
        self.mkca = np.zeros((self.numneu, self.cycle))
        self.H = np.zeros((self.numneu, self.cycle))
        self.am = np.ones(self.numneu)
        self.bm = np.ones(self.numneu)
        self.ah = np.ones(self.numneu)
        self.bh = np.ones(self.numneu)
        self.an = np.ones(self.numneu)
        self.bn = np.ones(self.numneu)
        self.dm = np.ones(self.numneu)
        self.dh = np.ones(self.numneu)
        self.dn = np.ones(self.numneu)
        self.dM = np.ones(self.numneu)
        self.dmks = np.ones(self.numneu)
        self.dhks = np.ones(self.numneu)
        self.dH = np.ones(self.numneu)
        self.Ina = np.ones(self.numneu)
        self.Ik = np.ones(self.numneu)
        self.IM = np.ones(self.numneu)
        self.Il = np.ones(self.numneu)
        self.Ild = np.ones(self.numneu)
        self.Inap = np.ones(self.numneu)
        self.Iks = np.ones(self.numneu)
        self.Ica = np.ones(self.numneu)
        self.Ikca = np.ones(self.numneu)
        self.IH = np.ones(self.numneu)
        self.k1 = np.ones(self.numneu)
        self.k2 = np.ones(self.numneu)
        self.k3 = np.ones(self.numneu)
        self.k4 = np.ones(self.numneu)
        print(self.vin)
        '''
        for i in range(self.numsyn):
            self.tau[i] = random.randint(2000, 3000)
            self.d[i] = random.randint(1000, 2000)
        '''

    def alpha(self, t):
        for i in range(0, self.numsyn):
            if t <= 0:
                return 0
            elif self.Pmax * (t  / self.tau[i]) * np.exp(-t / self.tau[i]) < 0.00001:
                return 0
            else:
                return self.Pmax * (t / self.tau[i]) * np.exp(-t / self.tau[i])

    def synin(self, i):
        for j in range(0, self.numneu):
            if self.vin[j, self.nowstep-1] > self.vth:
                self.syntm[j, self.nowstep:] = self.nowstep  # 発火時間を記録
        for j in range(0, self.numneu):
            for k in range(0, self.numsyn):
                self.gsyn[i, j, k] = self.alpha(self.nowstep - self.syntm[j, self.nowstep] - self.d[k] -self.dsyn)
                self.isynd[i, self.nowstep] += self.gsyn[i, j, k] * (self.vde[j, self.nowstep-1] - self.vsyn)

    def propagation(self):
        for i in range(0, self.numneu):
            '''
            チャネルの開く確率計算
            '''

            self.am[i] = -0.1 * (self.vin[i, self.nowstep] + 32) / (np.exp(-0.1 * (self.vin[i, self.nowstep] + 32)) - 1)
            self.bm[i] = 4 * np.exp( - (self.vin[i, self.nowstep] + 57)/ 18)
            self.k1[i] = 10 * (self.am[i] * (1 - self.m[i, self.nowstep]) - self.bm[i] * self.m[i, self.nowstep])
            self.k2[i] = 10 * (self.am[i] * (1 - (self.m[i, self.nowstep] + self.dt * self.k1[i] / 2)) - self.bm[i] * self.m[i, self.nowstep])
            self.k3[i] = 10 * (self.am[i] * (1 - (self.m[i, self.nowstep] + self.dt * self.k2[i] / 2)) - self.bm[i] * self.m[i, self.nowstep])
            self.k4[i] = 10 * (self.am[i] * (1 - (self.m[i, self.nowstep] + self.dt * self.k3[i])) - self.bm[i] * self.m[i, self.nowstep])
            self.dm[i] = self.dt * (self.k1[i] + 2 * self.k2[i] + 2 * self.k3[i] + self.k4[i]) / 6
            self.m[i, self.nowstep+1] = self.m[i, self.nowstep] + self.dm[i]

            self.ah[i] = 0.07 * np.exp( - (self.vin[i, self.nowstep] + 44) / 20)
            self.bh[i] = 1 / (np.exp(-0.1 *  (self.vin[i, self.nowstep] + 14)) + 1)
            self.k1[i] = 10 * (self.ah[i] * (1 - self.h[i, self.nowstep]) - self.bh[i] * self.h[i, self.nowstep])
            self.k2[i] = 10 * (self.ah[i] * (1 - (self.h[i, self.nowstep] + self.dt * self.k1[i] / 2)) - self.bh[i] * self.h[i, self.nowstep])
            self.k3[i] = 10 * (self.ah[i] * (1 - (self.h[i, self.nowstep] + self.dt * self.k2[i] / 2)) - self.bh[i] * self.h[i, self.nowstep])
            self.k4[i] = 10 * (self.ah[i] * (1 - (self.h[i, self.nowstep] + self.dt * self.k3[i])) - self.bh[i] * self.h[i, self.nowstep])
            self.dh[i] = self.dt * (self.k1[i] + 2 * self.k2[i] + 2 * self.k3[i] + self.k4[i]) / 6
            self.h[i, self.nowstep+1] = self.h[i, self.nowstep] + self.dh[i]

            self.an[i] = -0.01 * (self.vin[i, self.nowstep] + 30) / (np.exp(-0.1 * (self.vin[i, self.nowstep] + 30)) - 1)
            self.bn[i] = 0.125 * np.exp( - (self.vin[i, self.nowstep] + 40) / 80)
            self.k1[i] = 15 * (self.an[i] * (1 - self.n[i, self.nowstep]) - self.bn[i] * self.n[i, self.nowstep])
            self.k2[i] = 15 * (self.an[i] * (1 - (self.n[i, self.nowstep] + self.dt * self.k1[i] / 2)) - self.bn[i] * self.n[i, self.nowstep])
            self.k3[i] = 15 * (self.an[i] * (1 - (self.n[i, self.nowstep] + self.dt * self.k2[i] / 2)) - self.bn[i] * self.n[i, self.nowstep])
            self.k4[i] = 15 * (self.an[i] * (1 - (self.n[i, self.nowstep] + self.dt * self.k3[i])) - self.bn[i] * self.n[i, self.nowstep])
            self.dn[i] = self.dt * (self.k1[i] + 2 * self.k2[i] + 2 * self.k3[i] + self.k4[i]) / 6
            self.n[i, self.nowstep+1] = self.n[i, self.nowstep] + self.dn[i]

            self.minfM[i] = 1 / (1 + np.exp(-(self.vin[i, self.nowstep] + 44) / 6))
            self.taumM[i] = 100 / (np.exp(-(self.vin[i, self.nowstep] + 44) / 12) + np.exp((self.vin[i, self.nowstep] + 44) / 12))
            self.k1[i] = 10 * (self.minfM[i] - self.M[i, self.nowstep]) / self.taumM[i]
            self.k2[i] = 10 * (self.minfM[i] - (self.M[i, self.nowstep] + self.dt * self.k1[i] / 2)) / self.taumM[i]
            self.k3[i] = 10 * (self.minfM[i] - (self.M[i, self.nowstep] + self.dt * self.k2[i] / 2)) / self.taumM[i]
            self.k4[i] = 10 * (self.minfM[i] - (self.M[i, self.nowstep] + self.dt * self.k3[i])) / self.taumM[i]
            self.dM[i] = self.dt * (self.k1[i] + 2 * self.k2[i] + 2 * self.k3[i] + self.k4[i]) / 6
            self.M[i, self.nowstep+1] = self.M[i, self.nowstep] + self.dM[i]

            self.minfnap[i, self.nowstep + 1] = 1 / (1 + np.exp(-(self.vde[i, self.nowstep] + 45) / 5))

            self.minfks[i] = 1 / (1 + np.exp(-(self.vde[i, self.nowstep] + 34) / 6.5))
            self.taumks[i] = 8 / (np.exp(-(self.vde[i, self.nowstep] + 55) / 30) + np.exp((self.vde[i, self.nowstep] + 55) / 30))
            self.k1[i] = 10 * (self.minfks[i] - self.mks[i, self.nowstep]) / self.taumks[i]
            self.k2[i] = 10 * (self.minfks[i] - (self.mks[i, self.nowstep] + self.dt * self.k1[i] / 2)) / self.taumks[i]
            self.k3[i] = 10 * (self.minfks[i] - (self.mks[i, self.nowstep] + self.dt * self.k2[i] / 2)) / self.taumks[i]
            self.k4[i] = 10 * (self.minfks[i] - (self.mks[i, self.nowstep] + self.dt * self.k3[i])) / self.taumks[i]
            self.dmks[i] = self.dt * (self.k1[i] + 2 * self.k2[i] + 2 * self.k3[i] + self.k4[i]) / 6
            self.mks[i, self.nowstep+1] = self.mks[i, self.nowstep] + self.dmks[i]

            self.hinf[i] = 1 / (1 + np.exp((self.vde[i, self.nowstep] + 65) / 6.6))
            self.tauh[i] = 100 / (1 + np.exp(-(self.vde[i, self.nowstep] + 65) / 6.8)) + 100
            self.k1[i] = 10 * (self.hinf[i] - self.hks[i, self.nowstep]) / self.taumks[i]
            self.k2[i] = 10 * (self.hinf[i] - (self.hks[i, self.nowstep] + self.dt * self.k1[i] / 2)) / self.tauh[i]
            self.k3[i] = 10 * (self.hinf[i] - (self.hks[i, self.nowstep] + self.dt * self.k2[i] / 2)) / self.tauh[i]
            self.k4[i] = 10 * (self.hinf[i] - (self.hks[i, self.nowstep] + self.dt * self.k3[i])) / self.tauh[i]
            self.dhks[i] = self.dt * (self.k1[i] + 2 * self.k2[i] + 2 * self.k3[i] + self.k4[i]) / 6
            self.hks[i, self.nowstep+1] = self.hks[i, self.nowstep] + self.dhks[i]

            self.minfca[i, self.nowstep + 1] = 1 / (1 + np.exp(-(self.vde[i, self.nowstep] + 20) / 10))

            self.k1[i] = -0.002 * self.Ica[i] - self.ca[i, self.nowstep] / 200
            self.k2[i] = -0.002 * self.Ica[i] - (self.ca[i, self.nowstep] + self.dt * self.k1[i] / 2) / 200
            self.k3[i] = -0.002 * self.Ica[i] - (self.ca[i, self.nowstep] + self.dt * self.k2[i] / 2) / 200
            self.k4[i] = -0.002 * self.Ica[i] - (self.ca[i, self.nowstep] + self.dt * self.k3[i]) / 200
            self.dca[i] = self.dt * (self.k1[i] + 2 * self.k2[i] + 2 * self.k3[i] + self.k4[i]) / 6
            self.ca[i, self.nowstep+1] = self.ca[i, self.nowstep] + self.dca[i]

            self.mkca[i, self.nowstep + 1] = self.ca[i, self.nowstep] / (self.ca[i, self.nowstep] + 30)

            self.Hinf[i] = 1 / (1 + np.exp((self.vin[i, self.nowstep] + 69) / 7.1))
            self.tauH[i] = 1000 / (np.exp((self.vin[i, self.nowstep] + 66.4) / 9.3) + np.exp(-(self.vin[i, self.nowstep] + 81.6) / 13))
            self.k1[i] = self.dt * ((self.Hinf[i] - self.H[i, self.nowstep]) / self.tauH[i])
            self.k2[i] = self.dt * ((self.Hinf[i] - (self.H[i, self.nowstep] + self.dt * self.k1[i] / 2)) / self.tauH[i])
            self.k3[i] = self.dt * ((self.Hinf[i] - (self.H[i, self.nowstep] + self.dt * self.k2[i] / 2)) / self.tauH[i])
            self.k4[i] = self.dt * ((self.Hinf[i] - (self.H[i, self.nowstep] + self.dt * self.k3[i])) / self.tauH[i])
            self.dH[i] = self.dt * (self.k1[i] + 2 * self.k2[i] + 2 * self.k3[i] + self.k4[i]) / 6
            self.H[i, self.nowstep+1] = self.H[i, self.nowstep] + self.dH[i]

            self.Ina[i] = self.gna * (self.m[i, self.nowstep] ** 3) * self.h[i, self.nowstep] * (self.vin[i, self.nowstep] - self.vresna)
            self.Ik[i] = self.gk * (self.n[i, self.nowstep] ** 4) * (self.vin[i, self.nowstep] - self.vresk)
            self.IM[i] = self.gm * self.M[i, self.nowstep] * (self.vin[i, self.nowstep] - self.vresk)
            self.Il[i] = self.gl * (self.vin[i, self.nowstep] - self.vresl)

            self.Inap[i] = self.gnap * self.minfnap[i, self.nowstep] * (self.vde[i, self.nowstep] - self.vresna)
            self.Iks[i] = self.gks * self.mks[i, self.nowstep+1] * self.hks[i, self.nowstep+1] * (self.vde[i, self.nowstep] - self.vresk)
            self.Ikca[i] = self.gkca * self.mkca[i, self.nowstep] *  (self.vde[i, self.nowstep] - self.vresk)
            self.Ica[i] = self.gca * (self.minfca[i, self.nowstep] ** 2) * (self.vde[i, self.nowstep] - self.vresca)
            self.Ild[i] = self.gl * (self.vde[i, self.nowstep] - self.vresl)
            self.IH[i] = self.gH * (self.H[i, self.nowstep] ** 2) * (self.vin[i, self.nowstep] - self.vresH)


            self.synin(i)
            self.noise = np.random.normal(self.pnoise, self.ave, 1)

            self.dvin[i] = self.dt * (-self.As * (self.Il[i] + self.Ina[i] + self.Ik[i] + self.IM[i] - self.noise) - (self.vin[i, self.nowstep] - self.vde[i, self.nowstep]) / self.Rc) / (self.C * self.As)

            self.dvde[i] = self.dt * (-self.Ad * (self.Ild[i] + self.Inap[i] + self.Iks[i] + self.Ica[i] + self.Ikca[i] + self.IH[i] + self.isynd[i, self.nowstep]  - self.noise) - (self.vde[i, self.nowstep] - self.vin[i, self.nowstep]) / self.Rc) / (self.C * self.Ad)


            self.vin[i, self.nowstep + 1] = self.vin[i, self.nowstep] + self.dvin[i]

            self.vde[i, self.nowstep + 1] = self.vde[i, self.nowstep] + self.dvde[i]

        self.nowstep += 1  # 次の行へ
