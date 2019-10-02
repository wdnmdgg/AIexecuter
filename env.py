import shift
import collections
import time
from threading import Thread, Lock
import numpy as np
import copy

class CirList:
    def __init__(self, length):
        self.size = length
        self._table = [None]*length
        self.idx = 0
        self._counter = 0

    def insertData(self, data):
        self._counter += 1
        self._table[self.idx] = data
        self.idx = (self.idx+1) % self.size

    def getData(self):
        tail = self._table[0:self.idx]
        head = self._table[self.idx:]
        ret = head+tail
        return copy.deepcopy(ret)

    def isFull(self):
        return self._counter >= self.size

    def __repr__(self):
        return str(self.getData())


class Env:

    def __init__(self,
                 trader,
                 symbol,
                 commission,
                 action_space):
        """
        :param trader:
        :param symbol:
        :param commission:
        action_space: int: amount of actions we can have e.g. 5
        """

        self.symbol = symbol
        self.trader = trader
        self.commission = commission
        self.action_space = int(action_space)
        self.action_level = np.linspace(0,1,self.action_space)

        self.timeInterval = None
        self.nTimeStep = None
        self.isBuy = None
        self.total_share = None
        self.remained_share = None
        self.currentPos = None
        self.remained_steps = None
        self.objPrice = None
        self.base_price = None
        self.record = None
        self.time_total = None
        self.order = None

        self.dataThread = Thread(target=self._link)
        self.thread_alive = None
        self.ordertable = CirList(length=10)
        self.executedtable = CirList(length=10)
        self.mutex = Lock()


    def _link(self):
        while self.trader.isConnected() and self.thread_alive:
            if self.isBuy:
                arg = shift.OrderBookType.GLOBAL_ASK
            else:
                arg = shift.OrderBookType.GLOBAL_BID
            orders = self.trader.getOrderBookWithDestination(self.symbol, arg)
            if self.order:
                last_submitted_order = self.trader.get_executed_orders(self.order.id)
            else:
                last_submitted_order = np.nan
            self.mutex.acquire()
            # print(88)
            self.ordertable.insertData(orders)
            self.executedtable.insertData(last_submitted_order)
            # print(tmp)
            self.mutex.release()

            time.sleep(self.timeInterval)
        print('Data Thread stopped.')

    def kill_thread(self):
        self.thread_alive = False

    def set_objective(self, share, time_total,time_steps, objPrice):
        self.isBuy = True if share > 0 else False
        self.total_share = abs(share)
        self.remained_share = abs(share)
        self.time_total = time_total
        self.nTimeStep = time_steps
        self.timeInterval = self.time_total / self.nTimeStep
        self.objPrice = objPrice

        self.currentPos = self.getCurrentPosition()
        self.remained_steps = time_steps

        self.record = collections.defaultdict(list)
        self.thread_alive = True
        self.dataThread.start()

    def step(self, action):# action is shares we want to execute (or level of the ratio of the remained shares)
        #self.base_price = self.getClosePrice(action)
        orderType = shift.Order.MARKET_BUY if self.isBuy else shift.Order.MARKET_SELL
        #signBuy = 1 if self.isBuy else -1
        if self.remained_steps>0:
            shares_to_be_executed = np.floor(self.action_level[int(action)]*self.remained_share/100)
            self.order = shift.Order(orderType,self.symbol,shares_to_be_executed) # action should be size (1 size = 100 shares)
        else:
            self.order = shift.Order(orderType,self.symbol,self.remained_share)
        self.trader.submit_order(self.order)    #self.trader.submitOrder(order)
        """rest for a period of time"""
        if self.remained_steps > 0:
            time.sleep(self.timeInterval)
        else:
            time.sleep(1)

        # tmp_share = self.remained_share
        # self.currentPos = self.getCurrentPosition()
        # self.remained_share = self.total_share-abs(self.currentPos)
        # exec_share = tmp_share - self.remained_share
        # print(f"executed shares: {exec_share}")
        self.remained_steps -= 1

        if (int(self.remained_share)==0) or self.remained_steps<0:
            done = 1
        else:
            done = 0
        close_price, exec_size =self.getClosePrice(self.order.id)
        exec_share = exec_size*100
        if self.isBuy:
            reward = (exec_share * (close_price - self.objPrice)) + self.commission
        else:
            reward = (exec_share * (-close_price + self.objPrice)) + self.commission

        next_obs = self.get_obs()
        next_obs['reward'] = reward
        next_obs['isdone'] = done
        #self.add_features(self.record,next_obs,5)
        """reward, remained_time, remained_share, order_book, isdone"""
        return next_obs

    def getClosePrice(self,id_):
        last_submitted_order = self.trader.get_executed_orders(id_)
        add_price = 0
        executed_size = 0
        for order in last_submitted_order:
            add_price+=order.price*order.executed_size
            executed_size+=order.executed_size
        close_price = add_price/executed_size
        return close_price, executed_size  #self.trader.getClosePrice(self.symbol,self.isBuy,abs(share))

    def getCurrentPosition(self):# with sign
        return self.trader.get_portfolio_item(self.symbol).get_shares()     # self.trader.getPortfolioItem(self.symbol).getShares()

    def get_obs(self):
        allcloseprice = self.getClosePriceAll(99)########need modification!!!!!!!#########
        allcloseprice = np.asarray(allcloseprice)/self.objPrice
        rs_rate = self.remained_share/self.total_share
        rt_rate = self.remained_steps/self.nTimeStep
        states = np.hstack((allcloseprice,np.asarray([rs_rate,rt_rate ])))
        res = {'states':states}
        return res.copy()

    def get_record(self):
        return self.record

    def reset(self):
        next_obs = self.get_obs()
        next_obs['reward'] = np.nan
        next_obs['isdone'] = 1
        return next_obs.copy()

    # def getAllClosePrice(self,order_type,unit):
    #     if order_type == shift.Order.Type.MARKET_BUY:
    #         arg = shift.OrderBookType.GLOBAL_ASK
    #     else:
    #         arg = shift.OrderBookType.GLOBAL_BID
    #     self.orderbook = self.trader.getOrderBookWithDestination(self.symbol, arg)
    #     share_sum = 0
    #     price_sum = 0
    #     sizesum = 0
    #     queue = []
    #     res = []
    #     for order in self.orderbook:
    #         queue.append((order.price,order.size))
    #         sizesum += order.size
    #         while (sizesum >= unit) and queue:
    #             init = (queue[-1][1],int(sizesum-unit))
    #             queue[-1][1] -= init[1]
    #             price_sum += sum([j[1]*j[0] for j in queue])
    #             share_sum += unit
    #             res.append(price_sum / share_sum)
    #             if init[1]>0:
    #                 queue=[init]
    #                 sizesum = init[1]
    #             else:
    #                 queue = []
    #                 sizesum = 0
    #     if queue:
    #         price_sum+=sum([i[1]*i[0] for i in queue])
    #         share_sum += sum([i[1] for i in queue])
    #         res.append(price_sum / share_sum)
    #     return res
    #
    # def add_features(self,state_dict,features,limit):
    #     """features need to fit the key of the state_dictionary"""
    #     for f in features:
    #         state_dict[f].append(features[f])
    #         if len(state_dict[f])==limit:
    #             state_dict[f]=state_dict[f][1:]

    def getClosePriceAll(self, volumns:"maximum amount of close prices")->list:
        self.mutex.acquire()
        tabData = self.ordertable.getData()
        self.mutex.release()
        share_sum = 0
        price_sum = 0
        res = []
        for order in tabData[-1]:
            for i in range(1,order.size):
                share_sum+=1
                price_sum+=order.price
                res.append(price_sum/share_sum)
                if share_sum>=volumns:
                    return res
        return res







