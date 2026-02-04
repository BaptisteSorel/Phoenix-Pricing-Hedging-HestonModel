import numpy as np
from src.Models import calibration
from src.Models.Heston import HestonModel
from src.Products.phoenix import Phoenix

def phoenix_price(spot,current_time):
    dt=calibration.dt 
    coupon_rate=calibration.coupon_rate
    barrier_autocall=calibration.barrier_autocall
    barrier_coupon=calibration.barrier_coupon
    barrier_capital=calibration.barrier_capital
    nominal=calibration.nominal
    r=calibration.r
    T=calibration.T
    remaining_T=T-current_time
    N=10**3
    N_year=252

    if remaining_T <= 0.01: 
        S0 = calibration.S0 
        perf_final = spot / S0
        
        payoff = 0.0
        if perf_final >= barrier_coupon: 
            payoff = nominal + (coupon_rate * nominal)
        elif perf_final >= barrier_capital:
            payoff = nominal
        else:
            payoff = nominal * perf_final
        return payoff
    
    np.random.seed(1)

    model=HestonModel(kappa=calibration.kappa,theta=calibration.theta,sigma=calibration.sigma,rho=calibration.rho,v0=calibration.v0,S0=spot,r=r,T=T)

    remaining_steps=int(remaining_T * 252)
    paths=model.simulate_paths(N,remaining_steps)

    phoenix=Phoenix(T=remaining_T,dt=dt,coupon_rate=coupon_rate,barrier_autocall=barrier_autocall,barrier_coupon=barrier_coupon,barrier_capital=barrier_capital,nominal=nominal)
    phoenix.evaluate_payoffs_prices(paths)[0]
    prices=phoenix.evaluate_payoffs_prices(paths)[1]
    return np.mean(prices)

def delta(S,T):
    n_steps=len(S)
    delta=np.zeros(n_steps)
    dt=T/n_steps
    for i in range(n_steps):
        current_time=i*dt
        eps=0.01*S[i]
        delta[i]=(phoenix_price(S[i]+eps,current_time)-phoenix_price(S[i],current_time))/eps
    return delta
