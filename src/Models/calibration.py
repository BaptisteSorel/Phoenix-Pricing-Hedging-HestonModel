import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm

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
"""""

def get_SPY_data():
    ticker=yf.Ticker("SPY")
    expirations=ticker.options
    maturities=expirations[2],expirations[3],expirations[4] # We choose 3 different maturities
    today=datetime.now()

    S0=ticker.history(period="1d")["Close"].iloc[-1]
    r=yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1] / 100 #The risk-free rate is proxied using the 13-week US Treasury Bill rate (^IRX) from Yahoo Finance.

    data=[]
    for maturity in maturities:
        exp_date=datetime.strptime(maturity, "%Y-%m-%d")
        days_to_maturity=(exp_date - today).days
        T=days_to_maturity/365.25

        opt_chain=ticker.option_chain(maturity)
        calls=opt_chain.calls
        calls=calls[calls['volume'] > 10] # We ignore illiquid options
        calls=calls[(calls['strike'] > S0 * 0.9) & (calls['strike'] < S0 * 1.1)] # We ignore deep ITM and deep OTM calls

        for index, row in calls.iterrows():
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
        d=np.sqrt((rho*sigma*phi*1j-kappa)**2 + sigma**2*(phi*1j+phi**2))
        g=(kappa-rsi-d)/(kappa-rsi+d)
        
        exp1=np.exp(r*phi*1j*T)
        term1=S0**(phi*1j)*((1-g*np.exp(-d*T))/(1-g))**(-2 * kappa * theta / sigma**2)
        term2=np.exp((kappa*theta / sigma**2) * (kappa - rsi - d) * T)
        term3=np.exp((v0/sigma**2)*(kappa-rsi-d)*(1-np.exp(-d*T))/(1-g*np.exp(-d*T)))
        
        return exp1 * term1 * term2 * term3
    
    def integral_price(self, S0, K, T, r, kappa, theta, sigma, rho, v0):
        def integrand_1(phi):
            num=self.heston_characteristic_function(phi - 1j, S0, K, T, r, kappa, theta, sigma, rho, v0)
            denominator=1j * phi * S0 * np.exp(r * T)
            return (np.exp(-1j * phi * np.log(K)) * num / denominator).real

        def integrand_2(phi):
            num=self.heston_characteristic_function(phi, S0, K, T, r, kappa, theta, sigma, rho, v0)
            denominator=1j*phi
            return (np.exp(-1j*phi*np.log(K))*num / denominator).real
        
        limit_max = 100
        int_1, _=quad(integrand_1, 1e-8, limit_max)
        int_2, _=quad(integrand_2, 1e-8, limit_max)

        #Gil-Pelaez Formula
        P1=0.5+(1/np.pi)*int_1
        P2=0.5+(1/np.pi)*int_2

        price=S0*P1-K*np.exp(-r*T)*P2
            
        return price
    
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
        bnds = ((0.01, 15.0), (0.001, 1.0), (0.01, 2.0), (-0.99, 0.99), (0.001, 1.0))
        
        result = minimize(
            self.cost_function, 
            x0, 
            method='L-BFGS-B', 
            bounds=bnds,
            options={'disp': True, 'maxiter': 100}
        )
        
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

"""