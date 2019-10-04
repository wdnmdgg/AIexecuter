import numpy as np


class SimpleReplayPool:
    """
    Each element in pool is a tuple (obs, act, rwd, obs_plus1)
    """

    def __init__(self, max_pool_size, pop_size):
        self.max_pool_size = max_pool_size
        self.pop_size = pop_size
        self.size = 0
        self.pool = []

    def add_sample(self, obs, act, rwd, obs_plus1, terminal):
        self.size += 1
        if len(self.pool) > self.max_pool_size:
            self.size -= self.pop_size
            self.pool = self.pool[self.pop_size:]
        sample = (obs, act, rwd, obs_plus1, terminal)
        self.pool.append(sample)

    def random_batch(self, batch_size):
        """
        :param batch_size: may excess the current pool size!!
        :return: 5 variables [batch size, feature size]
        """
        assert batch_size < self.size
        index = np.random.choice(range(self.size), size=batch_size, replace=False)
        s0 = []
        act = []
        r = []
        s1 = []
        ter = []
        for j in index:
            s0.append(self.pool[j][0])
            act.append(self.pool[j][1])
            r.append([self.pool[j][2]])     # 1 dim
            s1.append(self.pool[j][3])
            ter.append([self.pool[j][4]])   # 1 dim

        s0 = np.array(s0)
        act = np.array(act)
        r = np.array(r)
        s1 = np.array(s1)
        ter = np.array(ter)

        return s0, act, r, s1, ter

    def current_size(self):
        return self.size


# a = SimpleReplayPool(100, 10)
# for i in range(112):
#     a.add_sample([i], [1-i/2, 2*i], 7, [2, 3, 4], 1)
#
# print(a.random_batch(1), a.current_size())
