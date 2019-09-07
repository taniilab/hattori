import numpy as np
np.set_printoptions(threshold=np.inf)


class ReservoirNetWork:
    def __init__(self, Time=30, dt=0.02, N=4):
        self.dt = dt
        self.N = N
        self.time = np.arange(0, Time, self.dt)
        self.allsteps = len(self.time)
        self.Vth = -20
        self.emem = -65
        self.V = -65 * np.ones((self.N, self.allsteps))
        self.t_fire = -100000*np.ones((self.N, self.N))
        self.Isyn = np.zeros((self.N, self.allsteps))
        self.esyn = 0
        #self.W = np.zeros((self.N, self.N))
        self.W = np.abs(np.random.randn(self.N, self.N))
        for i in range(0, self.N):
            self.W[i,i] = 0
        #self.W = np.ones((self.N, self.N))
        self.Iext = np.zeros((self.N, self.allsteps))
        self.input = 30*np.sin(0.5*self.time)
        self.Iext[0, :] = self.input
        #self.Iext[0, int(0.5*self.allsteps):] = 0
        self.tau = 1
        self.Isyn = np.zeros((self.N, self.allsteps))
        self.tau_syn = 5
        self.Pmax = 30
        self.curstep = 0


    def alpha_function(self, t):
            if t < 0:
                return 0
            elif ((self.Pmax * t / self.tau_syn * 0.1) *
                  np.exp(-t / self.tau_syn * 0.1)) < 0.00001:
                return 0
            else:
                return (self.Pmax * t / self.tau_syn) * np.exp(-t / self.tau_syn)


    def calc_synaptic_input(self, i):
        for j in range(0, self.N):
            self.Isyn[i, self.curstep] += \
                self.W[i, j]*self.alpha_function(self.curstep*self.dt - self.t_fire[i, j])


    def propagation(self):
        for i in range(0, self.N):
            self.calc_synaptic_input(i)

        self.V[:, self.curstep+1] \
            = self.V[:, self.curstep]\
              + self.dt*(self.tau*(self.emem-self.V[:, self.curstep])\
                         + self.Iext[:, self.curstep]\
                         + self.Isyn[:, self.curstep])

        for i in range(0, self.N):
            if self.V[i, self.curstep] >= -40:
                self.t_fire[:, i] = self.curstep*self.dt
                self.V[i, self.curstep+1] = -65

        self.curstep+=1


    def train(self):
        pass


    def learning(self):
        # Ridge Regression
        self.learning_steps = 1000
        self.targets = 10*np.sin(0.5*self.time[:self.learning_steps])
        self.lambda0 = 0.1
        self.output_nodes = self.V[:, 0:len(self.targets)].T # 多分qiitaの記事とは逆なので
        numneu = 4
        E_lambda0 = np.identity(numneu) * self.lambda0  # regularization
        inv_v = np.linalg.inv(self.output_nodes.T @ self.output_nodes + E_lambda0)
        """
        print(np.shape(E_lambda0))
        print(np.shape(self.output_nodes))
        print(np.shape(self.output_nodes.T))
        print(np.shape(self.output_nodes.T @ self.output_nodes))
        print(np.shape(self.targets))
        print(np.shape(inv_v))
        print(np.shape(inv_v @ self.output_nodes.T))
        """
        self.weights_output = (inv_v @ self.output_nodes.T) @ self.targets
        print(self.weights_output)
        pass

    def predict(self, steps):
        """
        for i in range(0, steps):
            self.time = np.append(self.time, self.time[-1]+self.dt)
        """
        self.predict_time = self.time[self.learning_steps:]
        self.predicted_results = self.weights_output @ self.V[:, self.learning_steps:]
        print(np.shape(self.predicted_results))
