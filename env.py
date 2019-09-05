import shift
import numpy as np
import time
class env:
    def __init__(self,
                 trader,
                 t,
                 nTimeStep,
                 symbol,
                 commission):
        self.timeInterval = t
        self.symbol = symbol
        self.nTimeStep = nTimeStep
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
    def step(self,action):
        self.base_price = self.getClosePrice(action)
        orderType = shift.Order.MARKET_BUY if self.isBuy else shift.Order.MARKET_SELL
        #signBuy = 1 if self.isBuy else -1
        if self.remained_time>0:
            order = shift.Order(orderType,self.symbol,action)
        else:
            order = shift.Order(orderType,self.symbol,self.remained_share)
        self.trader.submitOrder(order)
        tmp_share = self.remained_sahre
        self.remained_share = self.total_share-abs(self.base_price-self.currentPos)
        self.currentPos = self._getCurrentPosition()
        exec_share = tmp_share - self.remained_share
        done = False
        self.trader.getOder()
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
        orderbook = self.trader.getOrderBookWithDestination(self.symbol,self.isBuy)
        return np.asarray([self.remained_time,self.remained_share,orderbook])
    def get_record(self):
        return self.record







