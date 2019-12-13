import numpy as np
from scipy.stats import binom

class VanillaOption:
    def __init__(self, strike, expiry, payoff):
        self.__strike = strike
        self.__expiry = expiry
        self.__payoff = payoff

    @property
    def strike(self):
        return self.__strike

    @strike.setter
    def strike(self, new_strike):
        self.__strike = new_strike


    @property
    def expiry(self):
        return self.__expiry

    @expiry.setter
    def expiry(self, new_expiry):
        self.__expiry = new_expiry

    def payoff(self, spot):
        return self.__payoff(self, spot)

def call_payoff(option, spot):
    return np.maximum(spot - option.strike, 0.0)

def put_payoff(option, spot):
    return np.maximum(option.strike - spot, 0.0)


def single_period_model(option, spot, rate, u, d):
    Cu = option.payoff(u * spot)
    Cd = option.payoff(d * spot)
    delta = (Cu - Cd) / (spot * (u - d))
    bond = np.exp(-rate * option.expiry) * ((u * Cd - d * Cu) / (u-d))
    
    return delta, bond

def european_binomial(option, spot, rate, vol, div, steps):
    strike = option.strike
    expiry = option.expiry
    call_t = 0.0
    spot_t = 0.0
    h = expiry / steps
    num_nodes = steps + 1
    u = np.exp((rate - div) * h + vol * np.sqrt(h))
    d = np.exp((rate - div) * h - vol * np.sqrt(h))
    pstar = (np.exp(rate * h) - d) / ( u - d)
    
    for i in range(num_nodes):
        spot_t = spot * (u ** (steps - i)) * (d ** (i))
        call_t += option.payoff(spot_t) * binom.pmf(steps - i, steps, pstar)

    call_t *= np.exp(-rate * expiry)
    
    return call_t

def european_binomial_three(option, spot, rate, vol, div, steps, u, d):
    
    strike = option.strike
    expiry = option.expiry
    
    #Establishes needed numbers
    call_t = 0.0
    spot_t = 0.0
    h = expiry / steps
    num_nodes = steps + 1
    disc = np.exp(-rate * h) 
    
    #Creating the buckets/arrays for spot, premium, delta, bond
    spot_t = np.zeros((num_nodes, num_nodes))
    prc_t = np.zeros((num_nodes, num_nodes))
    del_t = np.zeros((num_nodes, num_nodes))
    bond_t = np.zeros((num_nodes, num_nodes))
    
    for j in range(num_nodes):
        spot_t[j,-1] = spot * (u ** (steps - j)) * (d ** (j))
        prc_t[j, -1] = option.payoff(spot_t[j, -1])
        

    for i in range((steps - 1), -1, -1):
        for j in range(i+1):
            spot_t[j,i] = spot_t[j,i+1] / u
            del_t[j,i] = (prc_t[j,i+1] - prc_t[j+1, i+1]) / (spot_t[j,i] * (u - d))
            bond_t[j,i] = disc * (u * prc_t[j+1,i+1] - d * prc_t[j,i+1])/ (u - d)
            prc_t[j,i] = del_t[j,i] * spot_t[j,i] + bond_t[j,i]
            
    print(f"The premium at each node is displayed in the arary below \n",prc_t)
    print(f"The delta at each node is displayed in the array below \n", del_t)
    print(f"The bond at each node is displayed in the array below \n",bond_t)
    
    return()

def european_binomial_four(option, spot, rate, vol, div, steps):
    strike = option.strike
    expiry = option.expiry
    call_t = 0.0
    spot_t = 0.0
    h = expiry / steps
    num_nodes = steps + 1
    #u = np.exp((rate - div) * h + vol * np.sqrt(h))
    #d = np.exp((rate - div) * h - vol * np.sqrt(h))
    u = 1.3
    d = 0.8
    pstar = (np.exp(rate * h) - d) / ( u - d)
    disc = np.exp(-rate * h) 
    spot_t = np.zeros((num_nodes, num_nodes))
    prc_t = np.zeros((num_nodes, num_nodes))
    del_t = np.zeros((num_nodes, num_nodes))
    bond_t = np.zeros((num_nodes, num_nodes))
    
    for j in range(num_nodes):
        spot_t[j,-1] = spot * (u ** (steps - j)) * (d ** (j))
        prc_t[j, -1] = option.payoff(spot_t[j, -1])

    for i in range((steps - 1), -1, -1):
        for j in range(i+1):
            spot_t[j,i] = spot_t[j,i+1] / u
            del_t[j,i] = (prc_t[j,i+1] - prc_t[j+1, i+1]) / (spot_t[j,i] * (u - d))
            bond_t[j,i] = disc * (u * prc_t[j+1,i+1] - d * prc_t[j,i+1])/ (u - d)
            prc_t[j,i] = del_t[j,i] * spot_t[j,i] + bond_t[j,i]
            
    return (prc_t[0,0], del_t[0,0])


def american_binomial_set(option, spot, rate, vol, div, steps, u, d):
    strike = option.strike
    expiry = option.expiry
    call_t = 0.0
    spot_t = 0.0
    h = expiry / steps
    num_nodes = steps + 1
#     u = np.exp((rate - div) * h + vol * np.sqrt(h))
#     d = np.exp((rate - div) * h - vol * np.sqrt(h))
    pstar = (np.exp(rate * h) - d) / ( u - d)
    disc = np.exp(-rate * h) 
    spot_t = np.zeros(num_nodes)
    prc_t = np.zeros(num_nodes)
    early_exercise = -1
    
    for i in range(num_nodes):
        spot_t[i] = spot * (u ** (steps - i)) * (d ** (i))
        prc_t[i] = option.payoff(spot_t[i])


    for i in range((steps - 1), -1, -1):
        for j in range(i+1):
            prc_t[j] = disc * (pstar * prc_t[j] + (1 - pstar) * prc_t[j+1])
            spot_t[j] = spot_t[j] / u
            
            if prc_t[j] > option.payoff(spot_t[j]):
                prc_t[j] = prc_t[j]
                
            else:
                prc_t[j] = option.payoff(spot_t[j])
                early_exercise += 1
                    
    return prc_t[0], early_exercise

def american_binomial(option, spot, rate, vol, div, steps):
    strike = option.strike
    expiry = option.expiry
    call_t = 0.0
    spot_t = 0.0
    h = expiry / steps
    num_nodes = steps + 1
    u = np.exp((rate - div) * h + vol * np.sqrt(h))
    d = np.exp((rate - div) * h - vol * np.sqrt(h))
    pstar = (np.exp(rate * h) - d) / ( u - d)
    disc = np.exp(-rate * h) 
    spot_t = np.zeros(num_nodes)
    prc_t = np.zeros(num_nodes)
    early_exercise = 0
    
    for i in range(num_nodes):
        spot_t[i] = spot * (u ** (steps - i)) * (d ** (i))
        prc_t[i] = option.payoff(spot_t[i])


    for i in range((steps - 1), -1, -1):
        for j in range(i+1):
            prc_t[j] = disc * (pstar * prc_t[j] + (1 - pstar) * prc_t[j+1])
            spot_t[j] = spot_t[j] / u
            
            if prc_t[j] > option.payoff(spot_t[j]):
                prc_t[j] = prc_t[j]
                
            else:
                prc_t[j] = option.payoff(spot_t[j])
                early_exercise += 1
                    
    return prc_t[0]


if __name__ == "__main__":
    print("This is a module. Not intended to be run standalone.")
