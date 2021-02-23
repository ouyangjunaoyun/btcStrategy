from scipy.stats import norm
from scipy import optimize
import numpy as np
from BTCTools.logger import get_logger
from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
iv_logger = get_logger('iv calculator', output_path=f'./log/ivcalculator-{current_time}.log')

class BSMethods():

    global iv_logger

    def __init__(self, S, K, r, T, vol, call_put='call'):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.vol = vol
        self.call_put = call_put

        iv_logger.info(f'--- initialize BS calculator of {self.call_put} with S={self.S}, K={self.K}, r={self.r}, TTM={self.T} and vol={self.vol}')
    
    def overrideVol(self, vol):
        
        iv_logger.info(f'override vol in BS calculator from {self.vol} to {vol}')
        
        self.vol = vol
    
    def computeBS(self, vol):
        local_vol = vol
        if self.vol != 0.0:
            local_vol = self.vol
        
        d1 = (np.log(self.S/self.K)+(self.r+0.5*local_vol**2)*self.T)/(local_vol*np.sqrt(self.T))
        d2 = d1 - local_vol*np.sqrt(self.T)
        iv_logger.debug(f'compute BS price of {self.call_put} with vol: {local_vol} and d_1: {d1}')

        if self.call_put == 'call':
            return self.S*norm.cdf(d1) - self.K*norm.cdf(d2)
        else:
            return self.K*norm.cdf(-d2) - self.S*norm.cdf(-d1)

    def computeDelta(self, vol):
        local_vol = vol
        if self.vol != 0.0:
            local_vol = self.vol
        d1 = (np.log(self.S/self.K)+(self.r+0.5*local_vol**2)*self.T)/(local_vol*np.sqrt(self.T))
        iv_logger.debug(f'compute delta of {self.call_put} with vol: {local_vol} and d_1: {d1}')
        if self.call_put == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1)-1

    def computeGamma(self, vol):
        local_vol = vol
        if self.vol != 0.0:
            local_vol = self.vol
        d1 = (np.log(self.S/self.K)+(self.r+0.5*local_vol**2)*self.T)/(local_vol*np.sqrt(self.T))
        iv_logger.debug(f'compute gamma of {self.call_put} with vol: {local_vol} and d_1: {d1}')
        return 1/(self.S*local_vol*np.sqrt(self.T))*norm.pdf(d1)

    def computeVega(self, vol):
        local_vol = vol
        if self.vol != 0.0:
            local_vol = self.vol
        d1 = (np.log(self.S/self.K)+(self.r+0.5*local_vol**2)*self.T)/(local_vol*np.sqrt(self.T))
        iv_logger.debug(f'compute vega of {self.call_put} with vol: {local_vol} and d_1=: {d1}')
        return self.S*np.sqrt(self.T)*norm.pdf(d1)

    def computeTheta(self, vol):
        local_vol = vol
        if self.vol != 0.0:
            local_vol = self.vol
        d1 = (np.log(self.S/self.K)+(self.r+0.5*local_vol**2)*self.T)/(local_vol*np.sqrt(self.T))
        d2 = d1 - local_vol*np.sqrt(self.T)
        iv_logger.debug(f'compute theta of {self.call_put} with vol: {local_vol} and d_1: {d1}')
        if self.call_put == 'call':
            return -(self.S*local_vol/2/np.sqrt(self.T) * norm.pdf(d1)) - self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        else:
            return -(self.S*local_vol/2/np.sqrt(self.T) * norm.pdf(d1)) + self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(-d2)

class IVSolver(BSMethods):
    
    global iv_logger

    def __init__(self, S, K, r, T, option_price, call_put='call'):
        BSMethods.__init__(self, S, K, r, T, 0.0, call_put)
        self.option_price = option_price
        iv_logger.info(f'--- initialize iv calculator with option price: {self.option_price}')
    
    def init_iv(self):
        return np.sqrt(2 * np.pi / self.T) * self.option_price / self.S
    
    def computeIV(self):
        try:
            initial_iv = self.init_iv()
            iv_logger.info(f'--- initialize iv in iv calculator to be {initial_iv}')
            f = lambda sigma : self.computeBS(sigma) - self.option_price
            fp = lambda sigma : self.computeVega(sigma)
            sigma = optimize.newton(f, initial_iv, fprime=fp)
            iv_logger.debug(f'solved iv: {sigma} with option price: {self.option_price}')
            return sigma
        except:
            iv_logger.error(f'unable to compute implied vol from option price {self.option_price}')
            raise ValueError