import numpy as np
from src.Models import calibration # Delete ".src" if you run the code from main
from src.Models.Heston import HestonModel
from src.Products.phoenix import Phoenix
from src.Analytics import greeks
import matplotlib.pyplot as plt

nominal=calibration.nominal
T=calibration.T
S0=calibration.S0
r=calibration.r

def sim_delta_hedging():
    model=HestonModel(kappa=calibration.kappa,theta=calibration.theta,sigma=calibration.sigma,rho=calibration.rho,v0=calibration.v0,S0=S0,r=r,T=T)
    N_year=252
    dt=1/N_year

    total_steps=int(T*N_year)
    path=model.simulate_paths(1,total_steps)[:,0]

    pnl_nohedge=np.zeros(total_steps)
    pnl_hedged=np.zeros(total_steps)

    #initialisation
    deltas=greeks.delta(path,T)
    cash=nominal-deltas[0]*path[0]
    pnl_nohedge[0]=nominal-greeks.phoenix_price(path[0],0)
    pnl_hedged[0]=cash-greeks.phoenix_price(path[0],0)+deltas[0]*path[0]

    observation_dates = [int(252/4 * k) for k in range(1, int(4*T + 1))]
    product_is_alive = 1
    final_pnl_value = 0.0

    for i in range(1,total_steps):
        if not product_is_alive:
            pnl_hedged[i] = final_pnl_value
            pnl_nohedge[i] = pnl_nohedge[i-1] # Ou 0, peu importe
            # On continue juste pour remplir le tableau pour le graphique
            continue
        t_current=i*dt
        cash=cash*(1+r*dt)

        if i in observation_dates:
            perf = path[i] / S0 # S0 global ou path[0]
            
            if perf>=calibration.barrier_autocall:
                payoff_autocall=nominal*(1+r)
                
                cash_from_hedge=deltas[i-1] * path[i]
                total_cash=cash+cash_from_hedge
                final_pnl_value=total_cash-payoff_autocall
                pnl_hedged[i]=final_pnl_value
                product_is_alive=0
                continue

        pnl_nohedge[i]=nominal-greeks.phoenix_price(path[i],i/N_year)
        cash-=(deltas[i]-deltas[i-1])*path[i]
        pnl_hedged[i]=-greeks.phoenix_price(path[i],i/N_year)+deltas[i]*path[i]+cash
    
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(pnl_hedged, label="PnL Delta Hedged")
    plt.plot(pnl_nohedge, label="PnL without hedging")
    plt.plot(path,label="underlying's prices for the PnL simulation")
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(y=calibration.barrier_autocall * S0, color='green', linestyle='--', linewidth=2, label=f'Autocall Barrier ({calibration.barrier_autocall*100:.0f}%)')
    plt.axhline(y=calibration.barrier_coupon * S0, color='orange', linestyle='--', linewidth=2, label=f'Coupons Barrier ({calibration.barrier_coupon*100:.0f}%)')
    plt.axhline(y=calibration.barrier_capital * S0, color='red', linestyle='--', linewidth=2, label=f'Capital Barrier ({calibration.barrier_capital*100:.0f}%)')
    plt.title("Simulation of Delta Hedging (Phoenix)")
    plt.ylabel("PnL mark-to-market ($)")
    plt.xlabel("Trading days")
    plt.legend()
    plt.show()

