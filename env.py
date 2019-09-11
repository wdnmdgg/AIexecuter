import shift
import numpy as np
import time


class env:

    def __init__(self,
                 trader,
                 time_interval,
                 time_steps,
                 symbol,
                 commission):
        """
        :param trader:
        :param time_interval:
        :param time_steps:
        :param symbol:
        :param commission:
        """
        self.timeInterval = time_interval
        self.symbol = symbol
        self.nTimeStep = time_steps
        self.trader = trader
        self.commission = commission

        self.isBuy = None
        self.total_share = None
        self.remained_share = None
        self.currentPos = None
        self.remained_time = None
        self.total_time = None
        self.objPrice = None
        self.record = []

    def set_objective(self, share, remained_time, objPrice):
        self.isBuy = True if share > 0 else False
        self.total_share = abs(share)
        self.remained_share = abs(share)
        self.currentPos = self._getCurrentPosition()
        self.remained_time = remained_time
        self.total_time = remained_time
        self.objPrice = objPrice

    def step(self, action):# action is shares we want to execute
        self.base_price = self.getClosePrice(action)
        orderType = shift.Order.MARKET_BUY if self.isBuy else shift.Order.MARKET_SELL
        #signBuy = 1 if self.isBuy else -1
        if self.remained_time>0:
            order = shift.Order(orderType,self.symbol,action)
        else:
            order = shift.Order(orderType,self.symbol,self.remained_share)
        self.trader.submitOrder(order)
        tmp_share = self.remained_share
        self.remained_share = self.total_share-abs(self.base_price-self.currentPos)
        self.currentPos = self._getCurrentPosition()
        exec_share = tmp_share - self.remained_share
        done = False
        reward = (exec_share * abs(self.getClosePrice(action)- self.objPrice) * 100 )+ self.commission
        if int(self._getCurrentPosition()*100)==self.total_share:
            done = True
        self.remained_time -= 1
        if self.remained_time > 0:
            time.sleep(self.timeInterval)
        else:
            time.sleep(1)
        next_obs = self.get_obs()
        self.record.append(next_obs)
        return next_obs,reward,done

    def getClosePrice(self,share):
        return self.trader.getClosePrice(self.symbol,self.isBuy,abs(share))

    def _getCurrentPosition(self):
        return int(self.trader.getPortfolioItem(self.symbol).getShares() / 100)

    def get_obs(self):
        allcloseprice = self.getAllClosePrice(self.isBuy,5)
        return [self.remained_time,self.remained_share,allcloseprice]

    def get_record(self):
        return self.record

    def getAllClosePrice(self,order_type,unit):
        if order_type:
            arg = shift.OrderBookType.GLOBAL_BID
        else:
            arg = shift.OrderBookType.GLOBAL_ASK
        self.orderbook = self.trader.getOrderBookWithDestination(self.symbol, arg)
        share_sum = 0
        price_sum = 0
        sizesum = 0
        queue = []
        res = []
        for order in self.orderbook:
            queue.append((order.price,order.size))
            sizesum += order.size
            while (sizesum >= unit) and queue:
                init = (queue[-1][1],int(sizesum-unit))
                queue[-1][1] -= init[1]
                price_sum += sum([j[1]*j[0] for j in queue])
                share_sum += unit
                res.append(price_sum / share_sum)
                if init[1]>0:
                    queue=[init]
                    sizesum = init[1]
                else:
                    queue = []
        if queue:
            price_sum+=sum([i[1]*i[0] for i in queue])
            share_sum += sum([i[1] for i in queue])
            res.append(price_sum / share_sum)
        return res





