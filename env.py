import shift
import collections
import time


class Env:

    def __init__(self,
                 trader,
                 symbol,
                 commission):
        """
        :param trader:
        :param symbol:
        :param commission:
        """

        self.symbol = symbol
        self.trader = trader
        self.commission = commission

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

    def set_objective(self, share, time_total, time_steps, objPrice, feature_key):
        self.timeInterval = time_total / time_steps
        self.nTimeStep = time_steps
        self.isBuy = True if share > 0 else False
        self.total_share = abs(share)
        self.remained_share = abs(share)
        self.currentPos = self.getCurrentPosition()
        self.remained_steps = time_steps
        self.objPrice = objPrice
        self.record = collections.defaultdict(list)

    def step(self, action):     # action is shares we want to execute
        # self.base_price = self.getClosePrice(action)
        orderType = shift.Order.MARKET_BUY if self.isBuy else shift.Order.MARKET_SELL
        # signBuy = 1 if self.isBuy else -1
        if self.remained_steps > 0:
            order = shift.Order(orderType, self.symbol, action)
        else:
            order = shift.Order(orderType, self.symbol, self.remained_share)
        self.trader.submitOrder(order)
        """rest for a period of time"""
        if self.remained_steps > 0:
            time.sleep(self.timeInterval)
        else:
            time.sleep(1)

        tmp_share = self.remained_share
        self.currentPos = self.getCurrentPosition()
        self.remained_share = self.total_share-abs(self.currentPos)
        self.remained_steps -= 1

        exec_share = tmp_share - self.remained_share
        if (int(self.getCurrentPosition() * 100) == self.total_share) or self.remained_steps < 0:
            done = True
        else:
            done = False

        if self.isBuy:
            reward = (exec_share * (self.getClosePrice(action) - self.objPrice)) + self.commission
        else:
            reward = (exec_share * (-self.getClosePrice(action) + self.objPrice)) + self.commission

        next_obs = self.get_obs()
        next_obs['reward'] = reward
        next_obs['isdone'] = done
        self.add_features(self.record, next_obs, 5)
        """reward, remained_time, remained_share, order_book, isdone"""
        return self.record

    def getClosePrice(self, share):
        return self.trader.getClosePrice(self.symbol, self.isBuy, abs(share))

    def getCurrentPosition(self):   # with sign
        return self.trader.getPortfolioItem(self.symbol).getShares()

    def get_obs(self):
        allcloseprice = self.getAllClosePrice(self.isBuy, 5)
        res = {'remained_time': self.remained_steps, 'remained_shares': self.remained_share, 'order_book':allcloseprice}
        return res

    def get_record(self):
        return self.record

    def getAllClosePrice(self, order_type, unit):
        if order_type == shift.Order.Type.MARKET_BUY:
            arg = shift.OrderBookType.GLOBAL_ASK
        else:
            arg = shift.OrderBookType.GLOBAL_BID
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
                if init[1] > 0:
                    queue = [init]
                    sizesum = init[1]
                else:
                    queue = []
                    sizesum = 0
        if queue:
            price_sum+=sum([i[1]*i[0] for i in queue])
            share_sum += sum([i[1] for i in queue])
            res.append(price_sum / share_sum)
        return res

    def add_features(self, state_dict, features, limit):
        """features need to fit the key of the state_dictionary"""
        for f in features:
            state_dict[f].append(features[f])
            if len(state_dict[f]) == limit:
                state_dict[f] = state_dict[f][1:]

    def getClosePriceAll(self, order_type, volumns):
        if order_type == shift.Order.Type.MARKET_BUY:
            arg = shift.OrderBookType.GLOBAL_ASK
        else:
            arg = shift.OrderBookType.GLOBAL_BID
        self.orderbook = self.trader.getOrderBookWithDestination(self.symbol, arg)
        share_sum = 0
        price_sum = 0
        res = []
        for order in self.orderbook:
            for i in range(1,order.size+1):
                share_sum+=1
                price_sum+=order.price
                res.append(price_sum/share_sum)
                if share_sum>volumns:
                    return res
        return res







