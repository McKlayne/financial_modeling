import numpy as np
import scipy.stats as stats
from scipy.stats import binom
from scipy.stats import norm
from collections import namedtuple

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
    
    for i in range(num_nodes):
        spot_t[i] = spot * (u ** (steps - i)) * (d ** (i))
        prc_t[i] = option.payoff(spot_t[i])


    for i in range((steps - 1), -1, -1):
        for j in range(i+1):
            prc_t[j] = disc * (pstar * prc_t[j] + (1 - pstar) * prc_t[j+1])
            spot_t[j] = spot_t[j] / u
            prc_t[j] = np.maximum(prc_t[j], option.payoff(spot_t[j]))
                    
    return prc_t[0]

PricerResult = namedtuple('PricerResult', ['price', 'stderr'])

def naive_monte_carlo_pricer(option, spot, rate, vol, div, reps, steps):
    strike = option.strike
    expiry = option.expiry
    dt = expiry / steps
    nudt = (rate - div - 0.5 * vol * vol) * dt
    sigdt = vol * np.sqrt(dt)
    z = np.random.normal(size=reps)
    paths = spot * np.exp(nudt + sigdt  * z)
    payoffs = option.payoff(paths) 
    price = np.mean(payoffs) * np.exp(-rate * expiry)
    stderr = np.std(payoffs, ddof=1) / np.sqrt(reps-1)
        
    return PricerResult(price, stderr)


def antithetic_monte_carlo_pricer(option, spot, rate, vol, div, reps, steps):
    strike = option.strike
    expiry = option.expiry
    dt = expiry / steps
    nudt = (rate - div - 0.5 * vol * vol) * dt
    sigdt = vol * np.sqrt(dt)
    lnS = np.log(spot)
    
    sum_CT = 0
    sum_CT2 = 0
    
    for j in range(reps):
        lnS_1 = lnS
        lnS_2 = lnS
        z = np.random.normal(size=reps)

        for i in range(int(steps)):
            lnS_1 = lnS_1 + nudt + sigdt * (z[i])
            lnS_2 = lnS_2 + nudt + sigdt * (-z[i])
            
            
        Spot_1 = np.exp(lnS_1)
        Spot_2 = np.exp(lnS_2)
        CT = 0.5 * (option.payoff(Spot_1) + option.payoff(Spot_2))
        sum_CT = sum_CT + CT
        sum_CT2 = sum_CT2 + CT*CT
    
    price = sum_CT/reps*np.exp(-rate * expiry)
    SD = np.sqrt((sum_CT2 - sum_CT*sum_CT/reps)*np.exp(-2*rate*expiry)/(reps - 1))
    stderr = SD/np.sqrt(reps)
        
    return PricerResult(price, stderr)

def black_scholes_delta(spot, t, strike, expiry, vol, rate, div):
    tau = expiry - t
    d1 = (np.log(spot/strike) + (rate - div + 0.5 * vol * vol) * tau) / (vol * np.sqrt(tau))
    delta = np.exp(-div * tau) * norm.cdf(d1) 
    return delta

def black_scholes_gamma(spot, t, strike, expiry, vol, rate, div):
    tau = expiry - t
    d1 = (np.log(spot/strike) + (rate - div + 0.5 * vol * vol) * tau) / (vol * np.sqrt(tau))
    gamma = np.exp(-div * tau) * norm.pdf(d1) / (spot * vol * np.sqrt(tau))
    return gamma

def black_scholes_delta_anti_control_pricer(option, spot, rate, vol, div, reps, steps):
    expiry = option.expiry
    strike = option.strike
    dt = expiry / steps
    nudt = (rate - div - 0.5 * vol * vol) * dt
    sigsdt = vol * np.sqrt(dt)
    erddt = np.exp((rate - div) * dt)    
    beta = -1.0
    cash_flow_t = np.zeros(reps)
    price = 0.0

    for j in range(reps):
        spot_1 = spot
        spot_2 = spot
        convar_1 = 0.0
        convar_2 = 0.0
        
        z = np.random.normal(size=reps)

        for i in range(int(steps)):
            t = i * dt
            delta_1 = black_scholes_delta(spot_1, t, strike, expiry, vol, rate, div)
            delta_2 = black_scholes_delta(spot_2, t, strike, expiry, vol, rate, div)
            spot_tn_1 = spot_1 * np.exp(nudt + sigsdt * z[i])
            spot_tn_2 = spot_2 * np.exp(nudt + sigsdt * -z[i])
            convar_1 = convar_1 + delta_1 * (spot_tn_1 - spot_1 * erddt)
            convar_2 = convar_2 + delta_2 * (spot_tn_2 - spot_2 * erddt)
            spot_1 = spot_tn_1
            spot_2 = spot_tn_2

        cash_flow_t[j] = option.payoff(spot_1) + beta * convar_1 + option.payoff(spot_2) + beta * convar_2

    price = np.exp(-rate * expiry) * cash_flow_t.mean() / 2
    stderr = cash_flow_t.std(ddof = 1) / np.sqrt(reps-1)
    
    return PricerResult(price, stderr)

def black_scholes_delta_gamma_anti_control_pricer(option, spot, rate, vol, div, reps, steps):
    expiry = option.expiry
    strike = option.strike
    dt = expiry / steps
    nudt = (rate - div - 0.5 * vol * vol) * dt
    sigsdt = vol * np.sqrt(dt)
    erddt = np.exp((rate - div) * dt)    
    egamma = np.exp((2*(rate - div) + vol**2) * dt)-2*erddt+1
    beta_1 = -1.0
    beta_2 = -0.5
    cash_flow_t = np.zeros(reps)
    price = 0.0

    for j in range(reps):
        spot_1 = spot
        spot_2 = spot
        convar_1 = 0.0
        convar_2 = 0.0
        
        z = np.random.normal(size=reps)

        for i in range(int(steps)):
            t = i * dt
            delta_1 = black_scholes_delta(spot_1, t, strike, expiry, vol, rate, div)
            delta_2 = black_scholes_delta(spot_2, t, strike, expiry, vol, rate, div)
            gamma_1 = black_scholes_gamma(spot_1, t, strike, expiry, vol, rate, div)
            gamma_2 = black_scholes_gamma(spot_2, t, strike, expiry, vol, rate, div)
            spot_tn_1 = spot_1 * np.exp(nudt + sigsdt * z[i])
            spot_tn_2 = spot_2 * np.exp(nudt + sigsdt * -z[i])
            convar_1 = convar_1 + delta_1 * (spot_tn_1 - spot_1 * erddt) + delta_2 * (spot_tn_2 - spot_2 * erddt)
            convar_2 = convar_2 + gamma_1 * ((spot_tn_1 - spot_1)**2-spot_1**2*egamma) + gamma_2 * ((spot_tn_2 - spot_2)**2-spot_2**2*egamma)
            spot_1 = spot_tn_1
            spot_2 = spot_tn_2

        cash_flow_t[j] = option.payoff(spot_1) + option.payoff(spot_2) + beta_1 * convar_1 + beta_2 * convar_2

    price = np.exp(-rate * expiry) * cash_flow_t.mean() / 2
    stderr = cash_flow_t.std(ddof = 1) / np.sqrt(reps-1)
    
    return PricerResult(price, stderr)



def stratified_monte_carlo_pricer(option, spot, rate, vol, div, reps):
    strike = option.strike
    expiry = option.expiry
    dt = expiry
    strat = np.random.uniform(size=reps)
    uhat = np.zeros(reps)
    for i in range(reps):
        uhat[i] = (i + strat[i])/reps
    
    z = stats.norm.ppf(uhat)
    nudt = (rate - div - 0.5 * vol * vol) * dt
    sigdt = vol * np.sqrt(dt)
    paths = spot * np.exp(nudt + sigdt  * z)
    payoffs = option.payoff(paths) 
    price = np.mean(payoffs) * np.exp(-rate * expiry)
    stderr = np.std(payoffs, ddof=1) / np.sqrt(reps-1)
    
    return PricerResult(price, stderr)


if __name__ == "__main__":
    print("This is a module. Not intended to be run standalone.")
