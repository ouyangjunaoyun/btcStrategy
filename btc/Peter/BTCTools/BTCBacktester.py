from enum import Enum
import pandas as pd
import glob
import matplotlib.pyplot as plt
from datetime import datetime, date
from BTCTools import BSTools
import pytz
import numpy as np
from BTCTools.logger import get_logger

class DataIndex(Enum):
    '''
    
    Indices for data stored in raw_data/Options/*
    
    '''
    Time = 0
    TimeStamp = 1
    LastPrice = 2
    Volume = 3
    AskPrice = 4
    AskVolume = 5
    BidPrice = 6
    BidVolume = 7

current_time = datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
backtester_logger = get_logger('backtester', output_path=f'./log/backtester-{current_time}.log')
trader_logger = get_logger('trader', output_path=f'./log/trader-{current_time}.log')
OPTION_HEADER = ['time', 'timestamp', 'last', 'volume', 'ask_p', 'ask_v', 'bid_p', 'bid_v']
INDEX_HEADER = ['time', 'price']

class Trader:
    '''
    
    Trader class to initialize position and rebalance
    
    Transaction Cost Definition:
        Maker - liquidity maker: BTC/ETH Futures/Perpetual 0.00%
                                 BTC/ETH Options 0.03% of the underlying or 0.0003 BTC/ETH per
                                    option contract (no more than 12.5% of the options price)
        Taker - liquidity taker: BTC/ETH Futures/Perpetual 0.05%
                                 BTC/ETH Options 0.03% of the underlying or 0.0003 BTC/ETH per
                                    option contract (no more than 12.5% of the options price)
    '''
    global trader_logger

    def __init__(self, initial_capital=1e6, option_notional=5, include_transaction_cost=False):

        self.initial_capital = initial_capital
        self.cash = self.initial_capital
        self.option_notional = option_notional
        self.include_transaction_cost = include_transaction_cost
        self.no_of_option = 0
        self.no_of_underlying = 0
        self.trading_threshold = abs(self.option_notional/1e2)
        trader_logger.info(f'--- initialize trader with initial capital: {initial_capital}, notionals of option to trade: {option_notional} with transaction cost: {include_transaction_cost}')
    
    def getInitialParameters(self):
        return self.initial_capital, self.option_notional
    
    def getCashAmount(self):
        return self.cash
    
    def outputPNL(self, option_price, underlying_price):
        trader_logger.debug(f'trader cash level: {self.cash}, option value: {self.no_of_option * option_price}, undelying asset value: {self.no_of_underlying * underlying_price}')
        return self.cash + self.no_of_option * option_price + self.no_of_underlying * underlying_price
    
    def underlyingTransactionCost(self, no_of_underlying, underlying_price):
        # Taker
        if self.include_transaction_cost:
            return 0.05/100 * no_of_underlying * underlying_price
        else:
            return 0.0
    
    def optionTransactionCost(self, no_of_option, option_price, underlying_price, index_price):
        # Taker
        if self.include_transaction_cost:
            return min(max(0.03/100 * underlying_price, 0.0003 * index_price) * no_of_option, 12.5/100 * option_price)
        else:
            return 0.0
    
    def rebalance(self, delta, underlying_price, current_time):
        notional_to_trade = delta - self.no_of_underlying # buy +, sell -
        if abs(notional_to_trade) > self.trading_threshold:
            transaction_cost = self.underlyingTransactionCost(abs(notional_to_trade), underlying_price)
            trader_logger.debug(f'notionals to trade: {notional_to_trade} at time {current_time} with cost: {transaction_cost}')
            self.cash -= notional_to_trade * underlying_price + transaction_cost
            self.no_of_underlying += notional_to_trade
    
    def tradeOption(self, option_price, underlying_price, index_price, optional_notional_of_option = 0):
        
        no_of_option_to_trade = optional_notional_of_option if optional_notional_of_option != 0 else self.option_notional
        transaction_cost = self.optionTransactionCost(abs(no_of_option_to_trade), option_price, underlying_price, index_price)
        self.cash += no_of_option_to_trade * option_price - transaction_cost
        self.no_of_option -= no_of_option_to_trade

        trader_logger.debug(f'sell option notional {no_of_option_to_trade} with cost: {transaction_cost}, current cash level: {self.cash} and current # of options: {self.no_of_option}')

class BackTester:
    
    '''
    
    Simulates options market making using raw_data/Options/*
    
    '''
    global backtester_logger

    def __init__(self, option_url, underlying_url, index_url, include_transaction_cost=False):
        
        global OPTION_HEADER, INDEX_HEADER
        self.option_pattern = option_url
        self.underlying_pattern = underlying_url
        self.btc_pattern = index_url
        self.include_transaction_cost = include_transaction_cost
        backtester_logger.info(f'--- initialize backtester with option file: {option_url}, underlying file: {underlying_url}, index file: {index_url} with transaction cost: {include_transaction_cost}')
        
        # local variables transformed from url's
        year = int(option_url.split('-')[0].split('.')[-1])
        month = int(option_url.split('-')[1])
        day = int(option_url.split('-')[2].split('.')[0])
        strike = int(option_url.split('-')[2].split('.')[1])
        without_timezone = datetime(year, month, day, 8, 0)
        timezone = pytz.timezone("UTC")
        with_timezone = timezone.localize(without_timezone)
        self.expiry = with_timezone
        self.strike = strike
        
        # for index and underlying
        self.underlying_data = pd.read_csv(self.underlying_pattern, names=OPTION_HEADER)
        self.underlying_data['timestamp'] = pd.to_datetime(self.underlying_data['time'], utc=True)
        self.underlying_data['timestamp'] = self.underlying_data['timestamp'].apply(lambda x: pd.Timestamp(x.year, x.month, x.day, x.hour, x.minute))
        self.btc_data = pd.read_csv(self.btc_pattern, names=INDEX_HEADER)
        self.btc_data['timestamp'] = self.btc_data['time'].apply(lambda x: pd.Timestamp(x))
        
        # for plotting
        self.timestamp = []
        self.pnl = []
        self.cash_level = []
        self.option_price_history = []
        self.delta_history = []
    
    def getUnderlyingPrice(self, current_time):
        offset = 0
        target = pd.Timestamp(year=current_time.year, month=current_time.month, day=current_time.day, hour=current_time.hour, minute=current_time.minute)
        target_data = self.underlying_data[self.underlying_data['timestamp'] == target]
        try:
            offset_minute = current_time.minute
            offset_hour = current_time.hour
            while target_data.empty:
                offset += 1
                if offset_minute-offset < 0:
                    offset_minute = 59
                    offset_hour -= 1
                    offset = 0
                target = pd.Timestamp(year=current_time.year, month=current_time.month, day=current_time.day, hour=current_time.hour, minute=offset_minute-offset)
                target_data = self.underlying_data[self.underlying_data['timestamp'] == target]
            return target_data['last'].mean()
        except:
            backtester_logger.error(f'error time: {current_time}, error data: {target_data}')
    
    def getBTCPrice(self, current_time):
        offset = 0
        target = pd.Timestamp(year=current_time.year, month=current_time.month, day=current_time.day, hour=current_time.hour, minute=current_time.minute)
        target_data = self.btc_data[self.btc_data['timestamp'] == target]
        try:
            while target_data.empty:
                offset += 1
                target = pd.Timestamp(year=current_time.year, month=current_time.month, day=current_time.day, hour=current_time.hour, minute=current_time.minute-offset)
                target_data = self.btc_data[self.btc_data['timestamp'] == target]
            return target_data.price.item()
        except:
            backtester_logger.error(f'error time: {current_time}, error data: {target_data}')
    
    def updatePNL(self, trader, current_time, option_price, underlying_price, delta):
        self.timestamp.append(current_time)
        self.pnl.append(trader.outputPNL(option_price, underlying_price))
        self.cash_level.append(trader.getCashAmount())
        self.option_price_history.append(option_price)
        self.delta_history.append(delta)
        backtester_logger.debug(f'current pnl is {self.pnl[-1]} with cash: {self.cash_level[-1]}')
        return
    
    def simulateData(self):
        
        global OPTION_HEADER

        for file in glob.glob(self.option_pattern):
            
            # initialize a trader with some notionals of options to sell
            backtester_logger.info(f'now starting to run option data: {file}')
            trader = Trader(initial_capital=1e6, option_notional=50, include_transaction_cost=self.include_transaction_cost)
            call_put = file.split('.')[-2]
            capital, notional = trader.getInitialParameters()
            data = pd.read_csv(file, names=OPTION_HEADER)
            data['timestamp'] = data['timestamp'].apply(lambda x: pd.Timestamp(x/1000, unit='s', tz='UTC'))
            
            for i, row in data.iterrows():
                
                current_time = row[DataIndex.TimeStamp.value]
                underlying_price = self.getUnderlyingPrice(current_time)
                index_price = self.getBTCPrice(current_time)
                option_price = (row[DataIndex.BidPrice.value] + row[DataIndex.AskPrice.value])/2 * index_price
                backtester_logger.debug(f'running at {current_time}, underlying price: {underlying_price}, index price: {index_price}, option price: {option_price}')

                # sell the option at middle price level at start, and rebalance constantly
                if i == 0:
                    trader.tradeOption(option_price, underlying_price, index_price)
                
                # trader rebalance at some specific threshold
                ttm = (self.expiry - current_time).total_seconds() / (365 * 24 * 60 * 60)
                try:
                    iv_calculator = BSTools.IVSolver(underlying_price, self.strike, 0.0, ttm, option_price, call_put)
                    iv = iv_calculator.computeIV()
                    delta = iv_calculator.computeDelta(iv)

                    # calculate the # of options to rebalance
                    delta_to_trade = delta * notional
                    trader.rebalance(delta_to_trade, underlying_price, current_time)

                    # record pnl, option price, cash level and delta history
                    self.updatePNL(trader, current_time, option_price, underlying_price, delta)

                except:
                    backtester_logger.error(f'unable to compute delta at time {current_time}, underlying price: {underlying_price}, option price: {option_price}, ttm: {ttm}')
            self.plotPNL()
    
    def plotPNL(self):
        fig, _ = plt.subplots(figsize=(12, 10))
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        
        ax1.set_title('PnL (Dollar Amount)')
        ax1.plot(self.timestamp, self.pnl, color='tab:red', label='PnL')
        ax1.set_ylabel('PnL ($)', color='tab:red')
        ax1.legend()
        #ax_copy = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        #ax_copy.plot(self.timestamp, self.option_price_history, color='tab:blue', label='Option Price')
        #ax_copy.set_ylabel('Option History', color='tab:blue')
        #ax_copy.legend()
        
        ax2.set_title('Option Price History')
        ax2.plot(self.timestamp, self.option_price_history, label='Option Price')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        
        ax3.set_title('Cash Level')
        ax3.plot(self.timestamp, self.cash_level, label='Cash Amount')
        ax3.legend()
        
        ax4.set_title('Delta Exposure (1 Option Contract)')
        ax4.plot(self.timestamp, self.delta_history, label='Delta')
        ax4.legend()
        
        plt.show()