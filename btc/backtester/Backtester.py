import sys
import numpy as np
import pandas as pd
import datetime as dt
import copy


module_path_list = ["/v/global/user/j/ju/junouyan/Jun/pylib/DateFunctions"]
for module_path in module_path_list:
    if module_path not in sys.path:
        sys.path.append(module_path)

from DateFunctions.DateFunctions import DateFunctions


# bokeh
import bokeh.plotting as bp
# import bokeh.layouts as bl
from bokeh.palettes import Paired as palette
# from bokeh.plotting import output_file, figure, show, save
# from bokeh.layouts import gridplot, column, row
from bokeh.models import LinearAxis, Range1d, Triangle, InvertedTriangle


class Position:
    def __init__(self, cash = 0, pos = None):
        if pos is None:
            pos = {}
        self.cash = cash
        self.pos = pos # sym : pos
    #         print(self.cash, str(self.pos))

    def newAct(self, sym, size, value, cost = 0):
        self.cash -= size * value
        self.cash -= abs(size) * cost # transaction cost
        if sym in self.pos.keys():
            self.pos[sym] += size
        else:
            self.pos[sym] = size

    def calcValue(self, market_data, valueType = 'price'):
        # market_data is in format: (price and yield)
        # [{date:date, sym: sym,  price :price, yield: yield ... }]
        value = self.cash
        for sym in self.pos.keys():
            values = market_data[market_data['sym'] == sym][valueType].values
            if len(values) > 0:
                value += self.pos[sym] * values[0]
            else:
                return np.nan
        return value

    def __str__(self):
        return "cash : " + str(round(self.cash,2)) + " position: " + str(self.pos)

    def copy(self):
        return Position(self.cash, self.pos.copy())

    def __add__(self, other):
        new_cash = self.cash + other.cash
        new_pos = {}
        for sym in list(set(self.pos.keys() + other.pos.keys())):
            size = self.pos.get(sym, 0) + other.pos.get(sym, 0)
            new_pos[sym] = size
        return Position(new_cash, new_pos.copy())

    def __sub__(self, other):
        new_cash = self.cash - other.cash
        new_pos = {}
        for sym in list(set(self.pos.keys() + other.pos.keys())):
            size = self.pos.get(sym, 0) - other.pos.get(sym, 0)
            new_pos[sym] = size
        return Position(new_cash, new_pos.copy())

    def isEmptyPos(self):
        r = []
        for sym, size in self.pos.items():
            r.append(size == 0)
        return all(r)

class Decision:
    '''
    Make decision per  instrument per time. Need date, sym, and how much to trade.
    '''
    def __init__(self, date, sym, size):
        self.date = date
        self.sym = sym
        self.size = size

    def __str__(self):
        return "Decision(date : " + self.date.strftime("%Y-%m-%d") + " sym: " + self.sym + " size: " + str(self.size) + ")"

    def copy(self):
        return Decision(self.date, self.sym, self.size)

class Trader:

    def __init__(self, cost = 0, valueType = 'price'):
        self.log = pd.DataFrame(columns = ['date', 'pos_SOD', 'decisions', 'pos_EOD', 'pnl'])
        self.position = Position()
        self.cache = {'decision_data' : None, 'market_data' : None}
        self.valueType = valueType
        self.cost = cost

    def make_decision(self,date, decision_data, market_data, syms):
        '''
         Force to overwrite

        # decision_data is in format:
        # [{date:date, sym: sym, value1 :value1, ... }]
        # market_data is in format: (price and yield)
        # [{date:date, sym: sym,  price :price, yield: yield ... }]
        # default
        '''
        # 1. calculate newAct based on decision_data
        decs = []
        self._exec_decision(date, market_data, decs)
        return decs

    def _exec_decision(self, date, market_data, decs = []):
        # 2. execute newAct and update position

        if not (decs is []):
            for dec in decs:
                values = market_data[market_data['sym'] == dec.sym][self.valueType].values
                if len(values) > 0:
                    value = values[0]
                    self.position.newAct(dec.sym, dec.size, value, self.cost)
                else:
                    print ("execution failed: &s, %s" % (date.strftime("%Y.%m.%d"), dec.sym))


    def update_cache(self, date, decision_data, market_data, syms):
        '''
         Force to overwrite

        # decision_data is in format:
        # [{date:date, sym: sym, value1 :value1, ... }]
        # market_data is in format: (price and yield)
        # [{date:date, sym: sym,  price :price, yield: yield ... }]
        # default
        '''
        pass

    def _update_log(self, date, decision_data, market_data, syms):
        # 1. update cache data for next day decision making
        #         print(decision_data)
        self.update_cache(date, decision_data, market_data, syms)
        #         print(self.cache['decision_data'])
        # 2. first calculate the decision_data and market data you want to pass in
        pos_SOD = self.position.copy()
        decs = self.make_decision(date,decision_data, market_data, syms)
        # 3. update logs
        pos_EOD = self.position.copy()
        pnl =  pos_EOD.calcValue(market_data, self.valueType)
        new_log = pd.DataFrame({'date' : date, 'pos_SOD' : [pos_SOD] , 'decisions' : [decs] , 'pos_EOD' : [pos_EOD], 'pnl' : [pnl]})
        self.log = self.log.append(new_log).reset_index(drop = True)

class Bactester:
    def __init__(self, trader, all_market_data, all_decision_data, syms):
        self.trader = trader
        self.all_market_data = all_market_data
        self.all_decision_data = all_decision_data
        self.finish = False
        self.syms = syms
        self.result = {}

    def prepareData(self):
        '''
         Force to overwrite

        # first need to init and pass in all_market_data and all_decision_data
        # then use all_market_data and all_decision_data to compute the new decision data need to pass in
        '''
        pass

    def calcStat(self):
        '''
        # first need to wait for backtesting to finish first
        # Then can calculate the result of the strategy
        '''
        if self.finish:
            emptyDays = self.trader.log['pos_SOD'].map(lambda x: x.isEmptyPos())
            empty_days_ratio = "%s / %s" % (sum(emptyDays), sum(~emptyDays))

            pnls = self.trader.log['pnl'][~emptyDays].dropna()
            if len(pnls > 0):
                # 1. total pnl
                total_pnl = pnls.values[-1]
                # 2. max pnl
                max_pnl = pnls.max()
                # 3. min pnl
                min_pnl = pnls.min()
                # 4. win_days_ratio
                gain = pnls.diff(1)
                num_win = sum(gain > 0)
                num_loss = sum(gain < 0)
                avg_win = np.mean(gain[gain > 0])
                avg_loss = np.mean(gain[gain < 0])
                win_days_ratio = "%s / %s" % (num_win , num_loss)
                # 5. win_avg_pnl_ratio
                win_avg_pnl_ratio = "%s / %s" % (avg_win , avg_loss)
                # 6. sharpe
                sharpe = np.sqrt(252) *np.mean(gain) / np.std(gain)
                # 7. num of decisions
                num_decisions = {}
                for value in self.trader.log['decisions'].tolist():
                    for dec in value:
                        if dec.sym in num_decisions.keys():
                            num_decisions[dec.sym] += 1
                        else:
                            num_decisions[dec.sym] = 1

                self.result = {"total_pnl": total_pnl, "max_pnl": max_pnl, "min_pnl": min_pnl,  "empty_days_ratio": empty_days_ratio,
                               "win_days_ratio":win_days_ratio, "win_avg_pnl_ratio": win_avg_pnl_ratio,"sharpe": sharpe,
                               "num of decisions": num_decisions}
                return self.result

        print("Backtesting not Done. Please run backtest() first")
        return None

    def plotPnlResult(self, syms = None, p = None):
        '''
        # first need to wait for backtesting to finish first
        # Then can plot the result
        '''
        if p is None:
            p = bp.figure(plot_width=1200,plot_height=400, x_axis_type='datetime',
                      toolbar_location="above", x_axis_label="Date",
                      y_axis_label="Pnl ($)", title="Pnl Result")
        if self.finish:
            p.line(self.trader.log['date'],self.trader.log['pnl'], line_color = 'black', line_width = 2, legend = 'pnl')


            if syms is None:
                syms = self.syms
            sym_color = {}
            for i in range(len(self.syms)):
                sym_color[self.syms[i]] = palette[12][i % 12]

            decision_dots = []
            for index, value in self.trader.log.iterrows():
                for dec in value['decisions']:
                    if dec.sym in syms:
                        decision_dots.append(pd.DataFrame({"date": [dec.date],"sym":[dec.sym], "size": [dec.size], "pnl": [value['pnl']]}))
            decision_dots = pd.concat(decision_dots)
            if len(syms) <= 5:
                for sym in syms:
                    decision_dots_selected = decision_dots.query("sym == @sym")
                    decision_dots_selected_pos = decision_dots_selected.query("size >= 0")
                    glyph_pos = Triangle(x=decision_dots_selected_pos['date'], y=decision_dots_selected_pos['pnl'], \
                                        size=1, line_color=sym_color[sym], line_width=2, fill_color=sym_color[sym], legend = sym)
                    p.add_glyph(glyph_pos)
                    decision_dots_selected_neg = decision_dots_selected.query("size < 0")
                    glyph_neg = InvertedTriangle(x=decision_dots_selected_neg['date'], y=decision_dots_selected_neg['pnl'], \
                                        size=1, line_color=sym_color[sym], line_width=2, fill_color=sym_color[sym], legend = sym)
                    p.add_glyph(glyph_neg)
            else:
                print("Too many syms to plot error!")
            p.legend.location = 'bottom_right'
            p.legend.click_policy="hide"
            return p
        else:
            print("Backtesting not Done. Please run backtest() first")
            return p

    def plotPositionResult(self, p = None):
        '''
        # first need to wait for backtesting to finish first
        # Then can plot the result
        '''
        if p is None:
            p = bp.figure(plot_width=1200,plot_height=400, x_axis_type='datetime',
                  toolbar_location="above",x_axis_label="Date",
                  y_axis_label="Position", title="Position Result")
        if self.finish:
            cash = self.trader.log['pos_EOD'].map(lambda x: x.cash)
            cash_positive_index = cash > 0
            # Setting the second y axis range name and range
            p.extra_y_ranges = {"cash": Range1d(start=min(cash)-1*max(abs(cash)), end=max(cash)+1*max(abs(cash)))}
            # Adding the second axis to the plot.
            p.add_layout(LinearAxis(y_range_name="cash", axis_label="Cash Position ($)"), 'right')
            p.rect(self.trader.log['date'][cash_positive_index], cash[cash_positive_index]/2, 1, cash[cash_positive_index],
                   fill_color="#D5E1DD", color="green", alpha = 0.4,  y_range_name="cash", legend = 'cash pos')
            p.rect(self.trader.log['date'][~cash_positive_index], cash[~cash_positive_index]/2, 1, cash[~cash_positive_index],
                   fill_color="#D5E1DD", color="red", alpha = 0.4,  y_range_name="cash", legend = 'cash pos')

            i = 0
            plots = []
            for sym in self.syms:
                plot=p.line(self.trader.log['date'],self.trader.log['pos_EOD'].map(lambda x: x.pos.get(sym, 0)),
                            line_color = palette[12][i % 12], line_width = 2, line_dash = 'dashed', legend = sym +' pos')
                plots.append(plot)
                i+=1
            p.legend.location = 'bottom_right'
            p.legend.click_policy="hide"
            p.y_range.renderers = plots
            return p
        else:
            print("Backtesting not Done. Please run backtest() first")
            return p

    def backtest(self, startDate = dt.date(2017, 1, 31), endDate = dt.date(2019, 8, 31)):
        '''
         # this is the main enter function of the code
         '''
        self.finish = False
        # 1. prepare data
        self.prepareData()

        # 2. start backtesting

        currentLoopDate = startDate
        while currentLoopDate < endDate:
            if (DateFunctions.isBusinessDay(currentLoopDate)):
                sys.stdout.write("\r Countdown: %s" % currentLoopDate.strftime("%Y-%m-%d"))
                sys.stdout.flush()

                decision_data_selected = self.all_decision_data.query("date == @currentLoopDate")
                # print(decision_data_selected)
                market_data_selected = self.all_market_data.query("date == @currentLoopDate")

                self.trader._update_log(currentLoopDate, decision_data_selected, market_data_selected, self.syms)

            currentLoopDate+= dt.timedelta(days =1)
        self.finish = True
        # 3. calc result
        self.calcStat()
        return self.trader, self.result


