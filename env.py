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
        return ret

    def isFull(self):
        return self._counter >= self.size

    def __repr__(self):
        return str(self.getData())


class Env:

    def __init__(self,
                 trader,
                 symbol,
                 commission,
                 action_space,
                 share,
                 time_total,
                 time_steps,
                 objPrice,
                 close_price_volumn):
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
        self.close_price_volumn = None
        self.set_objective(share,time_total,time_steps,objPrice,close_price_volumn)

        self.dataThread = Thread(target=self._link)
        self.ordertable = CirList(length=10)
        self.executedtable = CirList(length=10)
        self.mutex = Lock()
        self.thread_alive = True
        print('Waiting for connection', end='')
        for _ in range(5):
            time.sleep(1)
            print('.', end='')
        print()
        self.dataThread.start()

    def _link(self):
        while self.trader.is_connected() and self.thread_alive:
            self.arg1 = shift.OrderBookType.GLOBAL_ASK
            self.arg2 = shift.OrderBookType.GLOBAL_BID
            orders1 = self.trader.get_order_book_with_destination(self.symbol, self.arg1)
            orders2 = self.trader.get_order_book_with_destination(self.symbol, self.arg2)
            # if self.order:
            #     self.last_submitted_order = self.trader.get_executed_orders(self.order.id)
            # else:
            #     self.last_submitted_order = np.nan
            self.mutex.acquire()
            # print(88)
            self.ordertable.insertData((orders1, orders2))
            # self.executedtable.insertData(self.last_submitted_order)
            # print(tmp)
            self.mutex.release()
            #print(self.ordertable.getData()[-1][0].price)
            #print(self.timeInterval)
            time.sleep(int(self.timeInterval))
        print('Data Thread stopped.')

    def kill_thread(self):
        self.thread_alive = False

    def set_objective(self, share, time_total,time_steps, objPrice,close_price_volumn):
        self.isBuy = True if share > 0 else False
        self.total_share = abs(share)
        self.remained_share = abs(share)
        self.time_total = time_total
        self.nTimeStep = time_steps
        self.timeInterval = self.time_total / self.nTimeStep
        self.objPrice = objPrice

        self.currentPos = self.getCurrentPosition()
        self.remained_steps = time_steps
        self.close_price_volumn = close_price_volumn

        self.record = collections.defaultdict(list)


    def step(self, action):# action is shares we want to execute (or level of the ratio of the remained shares)
        #self.base_price = self.getClosePrice(action)
        orderType = shift.Order.Type.MARKET_BUY if self.isBuy else shift.Order.Type.MARKET_SELL
        #signBuy = 1 if self.isBuy else -1
        if self.action_level[int(action)] * self.remained_share<100:
            sizes_to_be_executed = 1
        else:
            sizes_to_be_executed = int(np.floor(self.action_level[int(action)] * self.remained_share / 100))
        if self.remained_steps>0:
            self.order = shift.Order(orderType,self.symbol,sizes_to_be_executed) # action should be size (1 size = 100 shares)
        else:
            self.order = shift.Order(orderType,self.symbol,self.remained_share)
        self.trader.submit_order(self.order)    #self.trader.submitOrder(o)rder)
        print(f"order id:{self.order.id}")
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
        close_price, exec_size =self.getClosePrice(self.order.id)
        exec_share = exec_size*100
        self.remained_share -= exec_share
        if (int(self.remained_share)==0) or self.remained_steps<0:
            done = 1
        else:
            done = 0
        if self.remained_steps == 0 and self.remained_share>0:
            penalty = -10000000
        else:
            penalty = 0
        if self.isBuy:
            reward = (exec_share * (-close_price + self.objPrice)) - self.commission + penalty
        else:
            reward = (exec_share * (close_price - self.objPrice)) - self.commission + penalty

        next_obs = self.get_obs(self.close_price_volumn)
        next_obs['reward'] = reward
        next_obs['isdone'] = done
        #self.add_features(self.record,next_obs,5)
        """close prices, remained_time, remained_share, reward, isdone"""
        return next_obs

    def getClosePrice(self,id_):
        last_submitted_order = self.trader.get_executed_orders(id_)
        add_price = 0
        executed_size = 0
        for order in last_submitted_order:
            add_price+=order.executed_price*order.executed_size
            #print(f'add_price: {add_price}')
            executed_size+=order.executed_size
        print(f'sum of executes size of last order: {add_price}')
        close_price = add_price/executed_size
        return close_price, executed_size  #self.trader.getClosePrice(self.symbol,self.isBuy,abs(share))

    def getCurrentPosition(self):# with sign
        return self.trader.get_portfolio_item(self.symbol).get_shares()     # self.trader.getPortfolioItem(self.symbol).getShares()

    def get_obs(self,close_price_volumn):
        allcloseprice = self.getClosePriceAll(self.isBuy,close_price_volumn)########need modification!!!!!!!#########
        allcloseprice = np.asarray(allcloseprice)/self.objPrice
        rs_rate = self.remained_share/self.total_share
        rt_rate = self.remained_steps/self.nTimeStep
        states = np.hstack((allcloseprice,np.asarray([rs_rate,rt_rate ])))
        res = {'states':states}
        return res.copy()

    def get_record(self):
        return self.record

    def reset(self):
        next_obs = self.get_obs(self.close_price_volumn)
        next_obs['reward'] = np.nan
        next_obs['isdone'] = 0
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

    def getClosePriceAll(self, isbuy,volumns:int):
        self.mutex.acquire()
        tabData = self.ordertable.getData()
        self.mutex.release()
        data = tabData[-1][0] if isbuy else tabData[-1][1]
        share_sum = 0
        price_sum = 0
        res = []
        #print(data)
        # for order in data:
        #     print(
        #         "%7.2f\t\t%4d\t%6s\t\t%19s"
        #         % (order.price, order.size, order.destination, order.time))
        for order in data:
            for i in range(1,order.size):
                share_sum+=1
                price_sum+=order.price
                res.append(price_sum/share_sum)
                if share_sum>=volumns:
                    return res
        if len(res)<volumns:
            res+=[0]*(volumns-len(res))
        return res

#if __name__ == "__main__":

# trader = shift.Trader("test001")
# try:
#     trader.connect("initiator.cfg", "password")
#     trader.sub_all_order_book()
# except shift.IncorrectPasswordError as e:
#     print(e)
# except shift.ConnectionTimeoutError as e:
#     print(e)
# env_test = Env(trader=trader,symbol='AAPL',commission=0,action_space=11,share=10000, time_total=60,time_steps=30, objPrice=100,close_price_volumn=10)
#env_test.set_objective(share=10000, time_total=60,time_steps=30, objPrice=100,close_price_volumn=10)
# while True:
#     env_test.ordertable
# # env_test.reset()
# #     env_test.step(3)
# data = env_test.ordertable.getData()
# for order in data[-1][0]:
#     print(
#         "%7.2f\t\t%4d\t%6s\t\t%19s"
#         % (order.price, order.size, order.destination, order.time))


