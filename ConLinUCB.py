import numpy as np
import math
import random


class ConLinUCB_UserStruct:
    def __init__(self, uid, dim, para, theta=None, tilde_theta=None, init='zero', gtheta_norm=None):
        self.uid = uid
        self.dim = dim
        self.para = para
        self.time = 1
        self.suparms_time = 1

        # intialize the feedback on arm slides

        self.M = para['lambda'] * np.identity(n=self.dim)
        self.Y = np.zeros((self.dim, 1))
        self.Minv = np.linalg.inv(self.M)

        if init == 'random':
            # self.tilde_theta = np.random.rand((self.dim, 1))
            self.theta = np.random.rand((self.dim, 1))
        else:
            # self.tilde_theta = np.zeros((self.dim, 1))
            self.theta = np.zeros((self.dim, 1))

        # self.cal_alpha = False
        self.gtheta_norm = gtheta_norm
        self.alpha = self.para['alpha']

    def get_X_M(self, X):
        self.X_M = np.dot(X, self.Minv)

    def getCredit(self, fv):  # this is the value used to select k
        result_a = np.dot(self.X_M, fv)
        result_b = 1 + np.dot(np.dot(fv.T, self.Minv), fv)
        norm_M = np.linalg.norm(result_a)
        return norm_M * norm_M / result_b


    def getRadius(self,fv,armpool):
        radius=0
        new_M_inv = self.getInv(self.Minv, fv)
        for aid,arminfo in armpool.items():
            radius+=np.dot(np.dot(arminfo.fv.T,new_M_inv),arminfo.fv)
        return radius

    def getRadius2(self,fv,armpool):
        new_M_inv=self.getInv(self.Minv,fv)
        max_P = float('-inf')
        radius=0
        for aid,arminfo in armpool.items():
            x_pta=np.dot(self.theta.T,arminfo.fv)
            var1=np.sqrt(np.dot(np.dot(arminfo.fv.T,new_M_inv),arminfo.fv))
            pta=x_pta+self.alpha*var1
            if pta>max_P:
                max_P=pta
                radius=var1
        return radius

    def getKeytermRadius(self,fv):
        radius = np.sqrt(np.dot(np.dot(fv.T, self.Minv), fv))
        return radius

    def getMaxEigen(self,fv):
        new_M_inv = self.getInv(self.Minv, fv)
        max_eigen=np.max(np.linalg.eig(new_M_inv)[0])
        return max_eigen

    def getProb(self, fv):

        mean = np.dot(self.theta.T, fv)
        # X_M_tM = np.dot(fv.T, self.M_tildeM_M)
        # X_M_tM_M_X = np.dot(X_M_tM, fv)
        var1 = np.sqrt(np.dot(np.dot(fv.T, self.Minv), fv))
        # var2 = np.sqrt(X_M_tM_M_X)
        #pta = mean + self.para['lambda'] * self.alpha * var1 + (1 - self.para['lambda']) * self.tilde_alpha * var2
        pta = mean + self.alpha * var1 # changed
        return pta

    def getInv(self, old_Minv, nfv):
        # new_M=old_M+nfv*nfv'
        # try to get the inverse of new_M
        tmp_a = np.dot(np.outer(np.dot(old_Minv, nfv), nfv), old_Minv)
        tmp_b = 1 + np.dot(np.dot(nfv.T, old_Minv), nfv)
        new_Minv = old_Minv - tmp_a / tmp_b
        return new_Minv

    def updateParameters(self, a_fv, reward):
        self.Minv = self.getInv(self.Minv, a_fv)
        self.M += np.outer(a_fv, a_fv)
        self.Y += a_fv * reward
        self.theta = np.dot(self.Minv, self.Y)
        self.time += 1


class ConLinUCB:

    def __init__(self, dim, para, suparm_strategy='random', init='zero', bt=lambda t: t + 1):
        self.dim = dim
        self.para = para
        self.init = init
        self.suparm_strategy = suparm_strategy
        self.users = {}
        self.bt = bt

    def get_suparm_budget(self, uid, norm):
        try:
            tmp = self.users[uid]
        except:
            self.users[uid] = ConLinUCB_UserStruct(uid, self.dim, self.para, gtheta_norm=norm)
        left_budget = self.bt(self.users[uid].time) - self.bt(self.users[uid].time - 1)
        if left_budget > 0:
            return int(left_budget)
        else:
            return -1

    def decide_suparms(self, pool_suparm, uid, norm, arms, X_t, debug_fw=None):
        try:
            tmp = self.users[uid]
        except:
            self.users[uid] = ConLinUCB_UserStruct(uid, self.dim, self.para, gtheta_norm=norm)

        if self.suparm_strategy == 'random':
            selected_index = np.random.randint(0, len(pool_suparm) - 1)
            # print('[ConUCB_decide_suparms] selected suparm : %d'%selected_index)
            return pool_suparm[selected_index]
        elif self.suparm_strategy == 'optimal_greedy':
            picked_suparm = None
            max_C = float('-inf')
            self.users[uid].get_X_M(X_t)

            for x, xinfo in pool_suparm.items():
                x_pta = self.users[uid].getCredit(xinfo.fv)
                if x_pta > max_C:
                    picked_suparm = xinfo
                    max_C = x_pta
            return picked_suparm
        elif self.suparm_strategy == 'forced exploration with BS':
            all_span_vectors=[]
            with open('saved_spanner.txt','r') as f:
                for line in f:
                    all_span_vectors=line
            b = list(all_span_vectors.strip('[').strip(']').split(', '))
            selected_index=int(random.choice(b))
            return pool_suparm[selected_index]
        elif self.suparm_strategy =='UCB':
            picked_suparm = None
            max_C = float('-inf')
            for x, xinfo in pool_suparm.items():
                x_pta = self.users[uid].getProb(xinfo.fv)
                if x_pta > max_C:
                    picked_suparm = xinfo
                    max_C = x_pta
            return picked_suparm
        elif self.suparm_strategy=='pick key term with max radius':
            picked_suparm = None
            max_C = float('-inf')
            for x, xinfo in pool_suparm.items():
                max_eigen = self.users[uid].getKeytermRadius(xinfo.fv)
                if max_eigen>max_C:
                    picked_suparm = xinfo
                    max_C = max_eigen
            return picked_suparm
        raise AssertionError

    def decide(self, pool_arms, uid, norm, debug_fw=None, best_arm=None):
        try:
            tmp = self.users[uid]
        except:
            self.users[uid] = ConLinUCB_UserStruct(uid, self.dim, self.para, gtheta_norm=norm)

        picked_arm = None
        max_P = float('-inf')

        for x, x_o in pool_arms.items():
            x_pta = self.users[uid].getProb(x_o.fv)
            if x_pta > max_P:
                picked_arm = x_o
                max_P = x_pta

        if picked_arm == None:
            raise AssertionError

        return picked_arm


    def updateParameters(self, picked_arm, reward, uid):
        self.users[uid].updateParameters(picked_arm.fv, reward)

    def getTheta(self, uid):
        return self.users[uid].theta

    def increaseSuparmTimes(self, uid):
        self.users[uid].suparms_time += 1
