import numpy
import shift
import tensorflow
import datetime

class env:

    def __init__(self, ticker, start_price, total_size, starting_time, time_limit, buy_side):
        self.ticker = ticker
        self.start_price = start_price
        self.total_size = total_size
        self.starting_time = starting_time
        self.time_limit = time_limit
        self.buy_side = buy_side

    def get_states(self, num):  # 'num' is the observation number

        # 1. current market price
        bp = shift.Trader.getBestPrice(self.ticker)
        if self.buy_side:
            current_price = bp.getBidPrice()
        else:
            current_price = bp.getAskPrice()

        # 2.






