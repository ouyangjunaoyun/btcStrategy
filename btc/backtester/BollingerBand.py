import sys
import os

module_path_list = ["/v/global/user/j/ju/junouyan/Jun/pylib/Backtester/src"]

for module_path in module_path_list:
    if module_path not in sys.path:
        sys.path.append(module_path)

from Backtester import *

# import imp
# imp.reload(backtester);
# imp.reload(backtester.Position);
# imp.reload(backtester.Decision);
# imp.reload(backtester.Trader);
# imp.reload(backtester.Bactester);


print(" In BollingerBandTrader Module ")


class BollingerBandTrader(Trader):
    # Boilinger Band
    '''
    # decision_data is in format:
    # [{date:date, sym : sym, indicator1 : oas}]
    # market_data is in format: (price and yield)
    # [{date:date, sym: sym,  price :price, yield: yield ... }]
    '''
    def __init__(self, cost=0, valueType = 'price'):
        Trader.__init__(self, cost, valueType)
        self.N = 20 #  day ema and estd of oas
        self.M = 1.5
        self.tradeSize = 100

    def make_decision(self,date, decision_data, market_data, syms):
        '''
         Force to overwrite

        # decision_data is in format:
        # [{date:date, sym: sym, value1 :value1, ... }]
        # market_data is in format: (price and yield)
        # [{date:date, sym: sym,  price :price, yield: yield ... }]
        # default
        '''
        # 1. calculate newAct based on decision_data and cache
        decs = []
        for sym in syms:
            #             print(self.cache['decision_data'].tail(1))
            this_decision_data = self.cache['decision_data'].query("(sym == @sym)").tail(1).to_dict('record')[0]
            # if (date == dt.date(2018,1,26)):
            #     print (this_decision_data)
            if this_decision_data['oas'] < this_decision_data['OASBOLD']:
                # if (date == dt.date(2018,1,26)):
                #     print (1)
                dec = Decision(date, sym, self.tradeSize)
                decs.append(dec)
            elif  this_decision_data['oas'] > this_decision_data['OASBOLU']:
                dec = Decision(date, sym, -self.tradeSize)
                decs.append(dec)
                # if (date == dt.date(2018,1,26)):
                #     print (2)
            else:
                # if (date == dt.date(2018,1,26)):
                #     print (3)
                pass
        # 2. execute newAct and update position
        #         decs = []
        if decs != []:
            self._exec_decision(date, market_data, decs)
        return decs

    def update_cache(self, date, decision_data, market_data, syms):
        '''
         Force to overwrite

        # decision_data is in format:
        # {date:date, value1 :value1, ... }
        # market_data is in format: (price and yield)
        # [{date:date, sym: sym,  price :price, yield: yield ... }]
        # default
        '''
        if not (self.cache['decision_data'] is None):
            new_decision_datas = []
            for sym in syms:
                old_decision_data = self.cache['decision_data'].query("(date < @date) and (sym == @sym)").tail(2*self.N)
                new_decision_data = old_decision_data.append(decision_data.query("sym == @sym"))
                alpha = 2/(self.N+1)
                new_decision_data.loc[:,'OASEma'] = new_decision_data['oas'].ewm(span = 20).mean()
                new_decision_data.loc[:,'OASEmstd'] = new_decision_data['oas'].ewm(span = 20).std()
                new_decision_data.loc[:,'OASBOLU'] = new_decision_data['OASEma'] + self.M*new_decision_data['OASEmstd']
                new_decision_data.loc[:,'OASBOLD'] = new_decision_data['OASEma'] - self.M*new_decision_data['OASEmstd']
                new_decision_datas.append(new_decision_data.tail(1))
            new_decision_datas=pd.concat(new_decision_datas)
            self.cache['decision_data'] = self.cache['decision_data'].append(new_decision_datas)
        else:
            new_decision_datas = []
            for sym in syms:
                new_decision_data = decision_data.query("sym == @sym")
                alpha = 2/(self.N+1)
                new_decision_data.loc[:,'OASEma'] = new_decision_data['oas'].ewm(span = 20).mean()
                new_decision_data.loc[:,'OASEmstd'] = new_decision_data['oas'].ewm(span = 20).std()
                new_decision_data.loc[:,'OASBOLU'] = new_decision_data['OASEma'] + self.M*new_decision_data['OASEmstd']
                new_decision_data.loc[:,'OASBOLD'] = new_decision_data['OASEma'] - self.M*new_decision_data['OASEmstd']
                new_decision_datas.append(new_decision_data.tail(1))
            new_decision_datas=pd.concat(new_decision_datas)
            self.cache['decision_data'] = new_decision_datas

