import numpy as np

class HestonModel:
    def __init__(self,kappa,theta,sigma,rho,v0,S0,r,T):
        self.kappa=kappa
        self.theta=theta
        self.sigma=sigma
        self.rho=rho
        self.v0=v0
        self.r=r
        self.T=T
        self.S0=S0
    def simulate_paths(self,num_paths,num_steps):
        dt=self.T/num_steps
        v,x=np.zeros((num_steps+1,num_paths)),np.zeros((num_steps+1,num_paths))
        v[0],x[0]=self.v0,np.log(self.S0) #x beeing the log normal distribution for the spot
        for i in range(num_steps):
            z1,z2=np.random.randn(num_paths),np.random.randn(num_paths)
            Z1=z1
            Z2=self.rho*z1+np.sqrt(1-self.rho**2)*z2 #The brownian motions are correlated

            positive_v=np.maximum(0,v[i])

            v[i+1]=v[i]+self.kappa*(self.theta-positive_v)*dt+self.sigma*np.sqrt(positive_v*dt)*Z1
            x[i+1]=x[i]+(self.r-0.5*positive_v)*dt+np.sqrt(dt*positive_v)*Z2
        S=np.exp(x)
        return S

