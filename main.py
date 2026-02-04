import numpy as np
import matplotlib.pyplot as plt

from src.Models import calibration
from src.Models.Heston import HestonModel
from src.Products.phoenix import Phoenix
from src.Analytics import pnl_sim

dt=calibration.dt 
coupon_rate=calibration.coupon_rate
barrier_autocall=calibration.barrier_autocall
barrier_coupon=calibration.barrier_coupon
barrier_capital=calibration.barrier_capital
nominal=calibration.nominal

def run_pricing():
    S0=calibration.S0
    r=calibration.r
    T=calibration.T
    N=10**4
    N_year=252

    model=HestonModel(kappa=calibration.kappa,theta=calibration.theta,sigma=calibration.sigma,rho=calibration.rho,v0=calibration.v0,S0=S0,r=r,T=T)

    total_steps=int(T*N_year)
    paths=model.simulate_paths(N,total_steps)

    phoenix=Phoenix(T=T,dt=dt,coupon_rate=coupon_rate,barrier_autocall=barrier_autocall,barrier_coupon=barrier_coupon,barrier_capital=barrier_capital,nominal=nominal)
    client_payoffs=phoenix.evaluate_payoffs_prices(paths)[0]
    product_prices=phoenix.evaluate_payoffs_prices(paths)[1]

    mean_payoffs=np.mean(client_payoffs)
    std_payoffs=np.std(client_payoffs)
    mean_prices=np.mean(product_prices)
    std_prices=np.std(product_prices)

    error_margin = 1.96 * (std_prices / np.sqrt(N))


    #### RESULTS ########
    print("The average price for this product is "+str(mean_prices)+" with a 0.95 confidence interval : ["+str(mean_prices-error_margin)+","+str(mean_prices+error_margin)+"]")
    if mean_prices<nominal:
        print("Positive margin : "+str(nominal-mean_prices)+"$")
    else:
        print("Negative margin : "+str(nominal-mean_prices)+"$")

    ###### PATHS #######
    nb_paths_to_plot = 100 
    subset_paths = paths[:, :nb_paths_to_plot]

    time_axis = np.linspace(0, T, subset_paths.shape[0])

    plt.figure(figsize=(12, 7))
    plt.plot(time_axis, subset_paths, color='grey', alpha=0.3, linewidth=0.5)
    
    plt.axhline(y=calibration.barrier_autocall * S0, color='green', linestyle='--', linewidth=2, label=f'Autocall Barrier ({phoenix.b_autocall*100:.0f}%)')
    plt.axhline(y=calibration.barrier_coupon * S0, color='orange', linestyle='--', linewidth=2, label=f'Coupons Barrier ({phoenix.b_coupon*100:.0f}%)')
    plt.axhline(y=calibration.barrier_capital * S0, color='red', linestyle='--', linewidth=2, label=f'Capital Barrier ({phoenix.b_capital*100:.0f}%)')

    plt.title(f"Simulation Heston : {nb_paths_to_plot} undelying's paths")
    plt.xlabel("Time (Year)")
    plt.ylabel("Underlying's price($)")
    plt.xlim(0, T)
    plt.ylim(S0 * 0.4, S0 * 1.6) 
    
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    ###### PAYOFFS ######
    plt.figure(figsize=(10, 6))
    plt.hist(product_prices, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_prices, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_prices:.0f}€')
    plt.axvline(1000, color='green', linestyle='dashed', linewidth=2, label='Nominal: 1000€')
    
    plt.title(f"Distribution of the product's payoffs (N={N})")
    plt.xlabel("Discounted payoffs ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_pricing()

if __name__ == "__main__":
    pnl_sim.sim_delta_hedging()
