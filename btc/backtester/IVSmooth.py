import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import norm
from scipy import optimize

from os import listdir
from os.path import isfile, join
import re
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from qpsolvers import solve_qp
import qpsolvers
from functools import reduce
from scipy import interpolate
from functools import partial

def NCDF(d):
    return norm.cdf(d)
def NPDF(d):
    return norm.pdf(d)
def TAU(t, T):
    def TAU_(t, T):
        return ((T - t).total_seconds() / (24*3600) / 365.25)
    f = np.vectorize(TAU_)
    return f(t, T)

class IVCalculator:
    def __init__(self):
        self.sigmas = []
    def initSigma(self, S,K,tau,r,q, price):
        return np.nan
    def target(self,S,K,tau,r,q,sigma, price):
        return np.nan
    def vega(self,S,K,tau,r,q,sigma):
        return np.nan
    
    def calc(self,S,K,tau,r,q, price, max_sigma = 10.0):
        sigma_0 = self.initSigma(S,K,tau,r,q, price)
        f = lambda sigma : self.target(S,K,tau,r,q,sigma, price)
        fp = lambda sigma : self.vega(S,K,tau,r,q,sigma)
        sigma = sigma_0
        result = optimize.newton(f, x0 = np.arange(0.1,10,0.1)*sigma_0,  fprime=fp, full_output=True)
        root, converged, zero_der = result
        sigma = root[converged & (~zero_der)]
        if (len(sigma) > 0):
            return sigma[0]
        else:
            print("!!! Exception: ", "sigma_0", sigma_0, "result: ", result, "S= ", S,"K= ", K,"tau= ", tau,"r= ", r,"q= ",q, "price= ", price)
            return np.nan
    

class IVCalculatorBSCall(IVCalculator):
    def __init__(self):
        IVCalculator.__init__(self)

    def initSigma(self, S,K,tau,r,q, price):
        return price/S *np.sqrt(2*np.pi/tau)
    def d1(self, S,K,tau,r,q,sigma):
        return (np.log(S/K) + (r-q+np.power(sigma,2)/2)*tau)/(sigma*np.sqrt(tau))
    def d2(self, S,K,tau,r,q,sigma):
        return self.d1(S,K,tau,r,q,sigma) - sigma*np.sqrt(tau)
    def vega(self,S,K,tau,r,q,sigma):
        d1 = self.d1(S,K,tau,r,q,sigma)
        v = S*np.exp(-q*tau)*np.sqrt(tau)*NPDF(d1)
        return v
    def bsPrice(self,S,K,tau,r,q,sigma):
        d1 = self.d1(S,K,tau,r,q,sigma)
        d2 = self.d2(S,K,tau,r,q,sigma)
        price = S*np.exp(-q*tau)*NCDF(d1) - K*np.exp(-r*tau)*NCDF(d2)
        return price
    def target(self,S,K,tau,r,q,sigma, price):
        self.sigmas.append(sigma)
        bsPrice = self.bsPrice(S,K,tau,r,q,sigma)
        return bsPrice - price
    
def ivCall(S, K, tau, r, q, price):
    c = IVCalculatorBSCall()
    return c.calc(S,K,tau, r,q, price)

def priceCall(S,K,tau,r,q,sigma):
    c = IVCalculatorBSCall()
    return c.bsPrice(S,K,tau, r,q, sigma)

def ivCallDf(option_price_df, future_price_df, t, r, q):

    mats = option_price_df.columns
    mats_option_ivs = {}
    mats_option_total_vars = {}
    for mat in mats:
        strikes_options_ivs = {}
        strikes_options_total_vars = {}
        option_df_selected_mat = option_price_df.loc[:,mat]
        future_df_selected_mat = future_price_df.loc[:,mat]
        strikes = option_df_selected_mat.index.unique().astype(float).values
        if (len(future_df_selected_mat) > 0):
            S = future_df_selected_mat.iloc[0]
            for strike in strikes:
                price = option_df_selected_mat.to_dict().get(strike, np.nan)
                if (~np.isnan(price)):
                    K = float(strike)
                    T = mat
                    tau = TAU(t, T)
                    iv = ivCall(S, K, tau, r, q, price)
                    strikes_options_ivs[strike] = iv
                    strikes_options_total_vars[strike] = np.power(iv, 2.0) * tau
            mats_option_ivs[mat] = strikes_options_ivs
            mats_option_total_vars[mat] = strikes_options_total_vars
    option_iv_df = pd.DataFrame(mats_option_ivs).sort_index()
    option_total_var_df = pd.DataFrame(mats_option_total_vars).sort_index()
    return option_iv_df, option_total_var_df

def priceCallDf(option_iv_df, future_price_df, t, r, q):

    mats = option_iv_df.columns
    mats_option_prices = {}
    for mat in mats:
        strikes_options_prices = {}
        
        option_df_selected_mat = option_iv_df.loc[:,mat]
        future_df_selected_mat = future_price_df.loc[:,mat]
        strikes = option_df_selected_mat.index.unique().astype(float).values

        if (len(future_df_selected_mat) > 0):
            S = future_df_selected_mat.iloc[0]
            for strike in strikes:
                iv = option_df_selected_mat.to_dict().get(strike, np.nan)
                if (~np.isnan(iv)):
                    K = float(strike)
                    T = mat
                    tau = TAU(t, T)
                    price = priceCall(S, K, tau, r, q, iv)
                    strikes_options_prices[strike] = price
            mats_option_prices[mat] = strikes_options_prices
    option_price_df = pd.DataFrame(mats_option_prices).sort_index()
    return option_price_df


def indexKappaToK(dfs_kappa, forward):
    dfs_K = []
    for i in range(len(dfs_kappa)):
        df_kappa = dfs_kappa[i]
        df_K = df_kappa
        df_K.index = df_K.index / forward.T[i]
        dfs_K.append(df_K)
    return dfs_K


def indexKToKappa(dfs_K, forward):
    dfs_kappa = []
    for i in range(len(dfs_K)):
        df_K = dfs_K[i]
        df_kappa = df_K
        df_kappa.index = df_kappa.index * forward.T[i]
        dfs_kappa.append(df_kappa)
    return dfs_kappa


def pre_smooth_i(example_option_total_var_df, i, forward, kappa):
#     i = 0
    T = example_option_total_var_df.columns[0]
    example_option_total_var_df_col = example_option_total_var_df.copy()
    example_option_total_var_df_col.index = example_option_total_var_df_col.index / forward.T[i]
#     print(example_option_total_var_df_col)
    min_valid_idx = example_option_total_var_df_col[~example_option_total_var_df_col.isnull().values].index.min()
    max_valid_idx = example_option_total_var_df_col[~example_option_total_var_df_col.isnull().values].index.max()
    kappa_valid = kappa[(kappa >= min_valid_idx)&(kappa <= max_valid_idx) ]
#     print(kappa_valid)
    example_option_total_var_df_col_kappa = example_option_total_var_df_col.append(pd.DataFrame({T: np.repeat(np.nan, len(kappa_valid))}, index = kappa_valid)).sort_index()
    # f_spline = interp1d(example_option_total_var_df_col.index, example_option_total_var_df_col.values.reshape(-1), kind='cubic',fill_value="extrapolate")
    # example_option_total_var_df_col_interpolate = f_spline(example_option_total_var_df_col_kappa.index)

    example_option_total_var_df_col_interpolate = example_option_total_var_df_col_kappa.interpolate(method = 'spline', order=3,limit_direction = 'both')

    example_option_total_var_df_col_interpolate_filter = example_option_total_var_df_col_interpolate.loc[kappa_valid,:]
    example_option_total_var_df_col_interpolate_filter = np.maximum(example_option_total_var_df_col_interpolate_filter,0) ## need to fix later
    return example_option_total_var_df_col_interpolate_filter

def pre_smooth(example_option_total_var_df, forward, kappa):
#     i = 0
    example_option_total_var_df_col_interpolate_filters = []
    for T in range(len(example_option_total_var_df.columns)):
        example_option_total_var_df_col_interpolate_filter = pre_smooth_i(example_option_total_var_df, i, forward, kappa)
        example_option_total_var_df_col_interpolate_filters.append(example_option_total_var_df_col_interpolate_filter)
    return example_option_total_var_df_col_interpolate_filters

def fitSpline(v, u, g, gamma):
    def fitSpline_(v, u, g, gamma):
#         print(v)
        # u, g length is n, gamma length is n-2
        # v is one strike point
        # u must be sorted, g, gamma are according to the order
        n = len(u)
        h = u[1:] - u[:-1]

        # add gamma1=0 and gamman=0
        gamma = np.concatenate([[0], gamma , [0]])

        if v <= u[0]:
            gp1 = (g[1] - g[0])/h[0] - (h[0]*gamma[1]) / 6.0
            return g[0] - (u[0] - v)*gp1

        elif v >= u[-1]:
            #gpn = (g[n-1] - g[n-2])/h[n-2] + (h[n-2] *gamma[n-3]) / 6.0
            gpn = (g[-1] - g[-2])/h[-1] + (h[-1] *gamma[-2]) / 6.0
            return g[-1]-(v - u[-1])*gpn
        else:
            idx = len(u[u <= v]) -1
            u_i = u[idx]
            u_i_1 = u[idx+1]
            g_i = g[idx]
            g_i_1 = g[idx+1]
            h_i = h[idx]
            gamma_i = gamma[idx]
            gamma_i_1 = gamma[idx+1]
            return ((v - u_i) * g_i_1 + (u_i_1-v)*g_i) / h_i - 1/ 6.0 * (v-u_i)*(u_i_1-v)*((1 + (v-u_i)/h_i)*gamma_i_1 + (1+ (u_i_1-v)/h_i)*gamma_i)
    fn = np.vectorize(fitSpline_, excluded=['u', 'g', 'gamma'])
    return fn(v=v, u=u, g=g, gamma=gamma)

def evaluateSpline(f_splines, option_df):
    option_smooth_df = option_df.copy()
    for T in option_df.columns:
        f_spline = f_splines[T].values
        if len(f_spline)> 0:
            f_spline = f_spline[0]
            option_smooth_df.loc[:,T] = f_spline(option_smooth_df.index)
        else:
            option_smooth_df.loc[:,T] = np.nan
    return option_smooth_df


# code 2

# t =  pd.to_datetime("2021-01-29 07:44:10+00:00", utc = True)

# example_option_df = result_df[result_df['datetime'] == t]['option_df'].values[0]
# example_future_df = result_df[result_df['datetime'] == t]['future_df'].values[0]

def smooth(example_option_df, example_future_df, t, r, q):

# setting for all maturity
# r = 0.001
# q = r

    Ts = example_option_df.columns
    taus = TAU(t, Ts)
    Ks = (example_option_df.index).values.reshape(-1,1)
    Fs = example_future_df.values

    # n = len(example_option_df_T)
    forward = Fs*np.exp((r-q)*taus); # 1* 3
    moneyness = Ks/forward; # 21 * 3
    kappa = np.arange((np.floor(np.min(moneyness*10))/10),(np.ceil(np.max(moneyness*10))/10),0.2)
    example_option_iv_df,example_option_total_var_df = ivCallDf(example_option_df, example_future_df, t, r, q)


    # loop from last maturity to first
    example_option_price_df_interpolate_filter_kappa_alls = []
    # example_option_price_df_interpolate_filter_K_T_alls = []
    f_splines = []
    for i in reversed(range(len(Ts))):
        # i = 2

        # pre smoothing
        T = example_option_iv_df.columns[i]
        example_option_total_var_df_T = example_option_total_var_df[[T]]
        forward_T = forward.T[i]
        min_valid_idx = example_option_total_var_df_T[~example_option_total_var_df_T.isnull().values].index.min()
        max_valid_idx = example_option_total_var_df_T[~example_option_total_var_df_T.isnull().values].index.max()
        example_option_total_var_df_T_valid = example_option_total_var_df_T.loc[(example_option_total_var_df_T.index >= min_valid_idx)&(example_option_total_var_df_T.index <= max_valid_idx) ]
        example_option_total_var_df_col_interpolate_filter = pre_smooth_i(example_option_total_var_df_T_valid, i, forward, kappa)


        # convert back to price + K index
        tau = TAU(t, T)
        example_option_iv_df_interpolate_filter = (example_option_total_var_df_col_interpolate_filter / tau).apply(np.sqrt)
        example_option_iv_df_interpolate_filter_K = example_option_iv_df_interpolate_filter.copy()
        example_option_iv_df_interpolate_filter_K.index = example_option_iv_df_interpolate_filter_K.index * forward_T

        example_option_price_df_interpolate_filter_K = priceCallDf(example_option_iv_df_interpolate_filter_K, example_future_df[[T]], t, r, q)
        example_option_price_df_interpolate_filter_K = np.maximum(example_option_price_df_interpolate_filter_K,0) # need to fix later

        # main smoothing
        example_option_price_df_interpolate_filter_K_T = example_option_price_df_interpolate_filter_K[T]
        example_future_df_T = example_future_df[T]


        kappa_T = example_option_total_var_df_col_interpolate_filter.index
        K = (example_option_price_df_interpolate_filter_K_T.index)
        F = example_option_price_df_interpolate_filter_K_T.iloc[0]
        n = len(example_option_price_df_interpolate_filter_K_T)
        u = K
        h = u[1:] - u[:-1]
        y = np.concatenate((example_option_price_df_interpolate_filter_K_T.values, np.repeat(0, n-2)))

        # get Q
        Q = np.zeros((n, n-2), float)
        q_higher = 1/h[1:]
        q_lower = 1/h[:-1]
        q_mid = -1/h[:-1] - 1/h[1:]
        np.fill_diagonal(Q, q_higher)
        rng = np.arange(n-2)
        Q[rng+1, rng] = q_mid
        Q[rng+2, rng] = q_lower

        # get R
        R = np.zeros((n-2, n-2), float)
        r_higher = h[1:-1]/6.0
        r_lower = r_higher
        r_mid = (h[:-1] + h[1:])/3.0
        np.fill_diagonal(R, r_mid)
        rng = np.arange(n-3)
        R[rng+1, rng] = r_lower
        R[rng, rng+1] = r_higher

        # get A
        A = np.append(Q, -1.0*R.T, axis = 0)

        # get B
        B = np.zeros((2*n-2, 2*n-2), float)
        rng = np.arange(n)
        rngmesh = tuple(np.meshgrid(rng,rng))
        B[rngmesh] = np.eye(n)
            # set lambda
        lamb = 0.01
        rng = n+np.arange(n-2)
        rngmesh = tuple(np.meshgrid(rng,rng))
        B[rngmesh] = lamb * R

        # Conditions: 
        # condition 7
        C7= np.zeros((1, 2*n-2), float)
        C7[0,0] = -1.0 / h[0]
        C7[0,1] = 1.0 / h[0]
        C7[0,n] = -h[0] / 6.0
        D7 =  np.array([-np.exp(-r * tau)])

        # condition 8
        C8= np.zeros((1, 2*n-2), float)
        C8[0,n-2] = 1.0 / h[n-2]
        C8[0,n-1] = -1.0 / h[n-2]
        C8[0,n] = -h[n-2] / 6.0
        D8 =  np.array([-np.exp(-r * tau)])

        C = np.concatenate((C7, C8))
        D = np.concatenate((D7, D8))


        # Lb and ub
        lb = np.zeros((2*n-2), float)
        lb[:n] = np.maximum(F*np.exp(-q*tau)-u*np.exp(-r*tau), 0).values.reshape(-1)


        if (i == (len(Ts)-1)):
            ub = np.zeros((2*n-2), float)
            ub[:n] = F*np.exp(-q*tau)
            ub[n:] = 1e12 # set inf 
        else:
            ub = np.zeros((2*n-2), float)
            gs = pd.concat(example_option_price_df_interpolate_filter_kappa_alls, axis = 1).loc[kappa_T, Ts[i+1]].values
            ub[:n] = np.exp(q*(taus[i+1]-taus[i]))*gs
            ub[n:] = 1e12




        ub = np.zeros((2*n-2), float)
        ub[:n] = F*np.exp(-q*tau)
        ub[n:] = 1e12 # set inf 


        # solve
        P = B # quick way to build a symmetric matrix
        q_ = -y
        G = -C
        h_ = -D
        A_ = A.T
        b = np.zeros((n-2), float)
        X = solve_qp(P, q_, G, h_, A_, b, lb, ub, solver= 'cvxopt')
        print("QP solution:", X)
        g = X[:n]
        gamma = X[n:]
        X_all = fitSpline(kappa*forward_T, u, g, gamma)
        # A partial function with b = 1 and c = 2 
        f_spline = partial(fitSpline, u = u, g = g, gamma = gamma) 
        f_splines.append(pd.DataFrame([f_spline], columns = [T]))
        
        example_option_price_df_interpolate_filter_kappa_T_all = pd.DataFrame(X_all, index = kappa, columns = [T])
        example_option_price_df_interpolate_filter_kappa_alls.append(example_option_price_df_interpolate_filter_kappa_T_all)
    #     example_option_price_df_interpolate_filter_kappa_alls = [pd.concat(example_option_price_df_interpolate_filter_kappa_alls, axis = 1)]
        
    #     X_all_K = fitSpline(Ks, u, g, gamma)
    #     example_option_price_df_interpolate_filter_K_T_all = pd.DataFrame(X_all_K, index = Ks.reshape(-1), columns = [T])
    #     example_option_price_df_interpolate_filter_K_T_alls.append(example_option_price_df_interpolate_filter_K_T_all)
    return pd.concat(f_splines, axis = 1)
