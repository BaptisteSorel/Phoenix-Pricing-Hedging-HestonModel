import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.optimize import differential_evolution

#### PARAMETERS ########
ticker=yf.Ticker("SPY")
S0=ticker.history(period="1d")["Close"].iloc[-1] 
r=yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1] / 100 #The risk-free rate 
T=3
dt=0.25 
coupon_rate=0.015
barrier_autocall=1
barrier_coupon=0.7
barrier_capital=0.6
nominal=1000


v0=0.04       # current variance
kappa=2.0     #mean reversion speed
theta=0.04    #Long term variance, mean reversion
sigma=0.5     #vol of the volatility
rho=-0.7      #Correlation between the underlying and its vol. Its <0 in equity

#Following calibration did'nt work
'''

def get_SPY_data():
    today=datetime.now()
    ticker=yf.Ticker("SPY")
    expirations=ticker.options
    expirations_filtered = [exp for exp in expirations if 20 < (datetime.strptime(exp, "%Y-%m-%d") - today).days < 400] # not too big, not too little
    maturities=expirations_filtered[0],expirations_filtered[len(expirations_filtered)//3],expirations_filtered[2*len(expirations_filtered)//3],expirations_filtered[-1] # We choose 4 different maturities

    S0=ticker.history(period="1d")["Close"].iloc[-1]
    r=yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1] / 100 #The risk-free rate is proxied using the 13-week US Treasury Bill rate (^IRX) from Yahoo Finance.

    data=[]
    for maturity in maturities:
        exp_date=datetime.strptime(maturity, "%Y-%m-%d")
        days_to_maturity=(exp_date - today).days
        T=days_to_maturity/365.25

        opt_chain=ticker.option_chain(maturity)
        calls=opt_chain.calls
        puts=opt_chain.puts
        options = pd.concat([calls,puts])
        options=options[options['volume'] > 10] # We ignore illiquid options
        options = options[(options['strike'] > S0 * 0.8) & (options['strike'] < S0 * 1.2)] # We ignore deep ITM and deep OTM calls
        options = options.nlargest(5, 'volume')  # garder les 5 plus liquides par maturité


        for index, row in options.iterrows():
            market_price = (row['bid'] + row['ask']) / 2
            data.append({
                'T': T,
                'K': row['strike'],
                'market_price': market_price, #In USD
                'S0': S0,
                'r': r
            })

    return pd.DataFrame(data)

class Heston_calibrator:
    def __init__(self,market_data):
        self.market_data=market_data
    
    def heston_characteristic_function(self,phi,S0,K,T,r,kappa,theta,sigma,rho,v0):
        rsi=rho*sigma*phi*1j
        d=np.sqrt((kappa-rsi)**2 + sigma**2*(phi*1j+phi**2))
        if np.real(d) < 0:
            d = -d
        g=(kappa-rsi-d)/(kappa-rsi+d)

        exp1=np.exp(r*phi*1j*T)
        log_term1=phi*1j*np.log(S0) + (-2*kappa*theta/sigma**2)*np.log((1-g*np.exp(-d*T))/(1-g))
        term2=np.exp((kappa*theta / sigma**2) * (kappa - rsi - d) * T)
        term3=np.exp((v0/sigma**2)*(kappa-rsi-d)*(1-np.exp(-d*T))/(1-g*np.exp(-d*T)))

        return exp1 * np.exp(log_term1) * term2 * term3
    
    def integral_price(self, S0, K, T, r, kappa, theta, sigma, rho, v0):
        phi = np.linspace(1e-4, 200, 500) 

        cf1 = np.array([self.heston_characteristic_function(p - 1j, S0, K, T, r, kappa, theta, sigma, rho, v0)
                        for p in phi])
        cf2 = np.array([self.heston_characteristic_function(p, S0, K, T, r, kappa, theta, sigma, rho, v0)
                        for p in phi])

        integrand_1 = (np.exp(-1j * phi * np.log(K)) * cf1 / (1j * phi * S0 * np.exp(r * T))).real
        integrand_2 = (np.exp(-1j * phi * np.log(K)) * cf2 / (1j * phi)).real

        int_1 = np.trapezoid(integrand_1, phi)
        int_2 = np.trapezoid(integrand_2, phi)

        #Gil Peleaz formula
        P1 = 0.5 + int_1 / np.pi
        P2 = 0.5 + int_2 / np.pi
        return S0 * P1 - K * np.exp(-r * T) * P2
    
    def bs_vega(self, S, K, T, r, sigma_approx=0.2):
        if T <= 0: return 1e-6 # Sécurité
        
        d1=(np.log(S/K)+(r+0.5*sigma_approx**2)*T)/(sigma_approx * np.sqrt(T))
        
        # Vega=S*sqrt(T)*N'(d1)
        vega=S*np.sqrt(T)*norm.pdf(d1)
        
        return max(vega, 1e-4)

    def cost_function(self, params):
        kappa, theta, sigma, rho, v0 = params
        error = 0.0
        
        for index, row in self.market_data.iterrows():
            # 1. Calcul du Prix Heston
            model_price = self.integral_price(
                S0=row['S0'], K=row['K'], T=row['T'], r=row['r'],
                kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0
            )
            
            # 2. Calcul du Vega de l'option (poids)
            vega=self.bs_vega(row['S0'], row['K'], row['T'], row['r'])
            
            # 3. Erreur pondérée (C'est l'astuce !)
            # Cela revient à minimiser l'erreur en volatilité implicite
            diff=model_price-row['market_price']
            weighted_error=diff/vega 
            
            error += weighted_error ** 2
            
        return np.sqrt(error / len(self.market_data))
    
    def calibrate(self):
        #Initial guesses
        # kappa, theta, sigma, rho, v0
        x0 = [2.0, 0.04, 0.3, -0.5, 0.04]
        
        # Bounds
        # kappa > 0.01, theta > 0, sigma > 0, rho entre -1 et 1, v0 > 0
        bnds = ( (0.1, 8.0), (0.01, 0.15),  (0.05, 1.0),    (-0.99, -0.1), (0.005, 0.15)  )

        
        result_global = differential_evolution(self.cost_function, bounds=bnds, maxiter=20, seed=42)
        result = minimize(self.cost_function, result_global.x, method='L-BFGS-B', bounds=bnds)
        
        print(f"Erreur finale (RMSE) : {result.fun:.4f}")
        
        return result.x

# Test
if __name__ == "__main__":
    df = get_SPY_data()
    if not df.empty:
        calib=Heston_calibrator(df)
        params=calib.calibrate()
        labels=["Kappa", "Theta", "Sigma", "Rho", "v0"]
        for l, v in zip(labels, params):
            print(f"{l}: {v:.4f}")
    else:
        print("No data")

'''