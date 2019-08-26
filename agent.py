import numpy
import shift
import tensorflow
import datetime


class agent:

    def __init__(self, ticker, start_price, total_size, starting_time, time_limit, buy_side):
        self.ticker = ticker
        self.start_price = start_price
        self.total_size = total_size
        self.starting_time = starting_time
        self.time_limit = time_limit
        self.buy_side = buy_side

    def RNN(self, env, num):    # 'num' is the observation number
        exe_size = 0
        # Please imagine a recurrent neural network here
        # Thank you for your cooperation!
        return exe_size
