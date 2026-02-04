import numpy as np
from src.Models import calibration #if you run from the main.py, delete 'src.'
from src.Models.Heston import HestonModel

coupon_rate=calibration.coupon_rate
barrier_autocall=calibration.barrier_autocall
barrier_coupon=calibration.barrier_coupon
barrier_capital=calibration.barrier_capital
nominal=calibration.nominal
r=calibration.r
T=calibration.T
dt=calibration.dt 

class Phoenix :
    def __init__(self, T, dt, coupon_rate, 
                 barrier_autocall, barrier_coupon, barrier_capital, 
                 nominal):
        self.T=T
        self.dt=dt
        self.coupon_rate=coupon_rate
        self.b_autocall=barrier_autocall
        self.b_coupon=barrier_coupon
        self.b_capital=barrier_capital
        self.nominal=nominal
        self.r=calibration.r
    
        self.observations=np.array([int(252/4*i) for i in range(1,int(4*T+1))])
        
    def evaluate_payoffs_prices(self,paths):
        N=paths.shape[1]
        self.payoffs,self.discounted_payoffs=np.zeros(N),np.zeros(N)

        for i in range(N):
            payoff=0
            S=paths[:,i]
            memory_coupons=0 #coupons not earned
            is_alive=1 #to know if the product is still alive i.e. not autocalled
            for days in self.observations:
                current_time=days/252
                perf=S[days]/S[0]
                if perf>self.b_autocall:
                    payoff+=self.nominal+self.coupon_rate*self.nominal+memory_coupons*self.nominal
                    is_alive=0
                    discount=np.exp(-self.r*current_time)
                    break
                elif perf>self.b_coupon:
                    payoff+=self.coupon_rate*self.nominal+memory_coupons*self.nominal
                    memory_coupons=0
                else:
                    memory_coupons+=self.coupon_rate
            if is_alive:
                discount=np.exp(-self.r*T)
                final_perf=S[-1]/S[0]
                if final_perf<self.b_capital:
                    payoff+=self.nominal*final_perf
                else:
                    payoff+=self.nominal
            self.payoffs[i]=payoff
            self.discounted_payoffs=self.payoffs*discount
        return self.payoffs,self.discounted_payoffs
