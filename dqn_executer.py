from env import Env
from agent import LSTMAgent
from pool import SimpleReplayPool
import shift
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import Q_function as Qf
import time

GAMMA = 0.95
EPSILON = 0.1
EPISODES = 1000

trader = shift.Trader("democlient")
try:
    trader.connect("initiator.cfg", "password")
    trader.sub_all_order_book()
except shift.IncorrectPasswordError as e:
    print(e)
except shift.ConnectionTimeoutError as e:
    print(e)

symbol = "AAPL"
commission = 0
sess = tf.Session()
execute_time = 300     # total execution time (seconds)
exe_times = 150        # steps
exe_interval = execute_time / exe_times
action_space = 11
exe_shares = 100*10     # shares
exe_price = 100         # dollar
episode_list = []       # to be continued


#def main():
env = Env(trader=trader,
          symbol=symbol,
          commission=commission,
          action_space=action_space)
agent = LSTMAgent(sess_=sess,
                  observations_dim=12,
                  action_space=action_space,
                  batch_size=5,
                  Q_function=Qf.ann,
                  optimizer=tf.train.AdamOptimizer,
                  GAMMA=GAMMA,
                  EPSILON=EPSILON,
                  learning_rate=0.001)
pool = SimpleReplayPool(max_pool_size=1000,
                        pop_size=100)

for i in range(EPISODES):
    # Deal with the initialization for each episode
    print(f'The number {i} episode \n\n')
    if i//2 == 1:
        env.set_objective(share=-exe_shares,
                          time_total=execute_time,
                          time_steps=exe_times,
                          objPrice=exe_price,
                          close_price_volumn=10)
        print(f'This time sell {exe_shares} shares\n\n')
    else:
        env.set_objective(share=exe_shares,
                          time_total=execute_time,
                          time_steps=exe_times,
                          objPrice=exe_price,
                          close_price_volumn=10)
        print(f'This time buy {exe_shares} shares\n\n')
    ob = env.reset()
    act = agent.get_action(ob['states'])
    print(act)
    ob = env.step(act)
    terminal = ob['isdone']
    print(f'observation is {ob}\nremained shares: {env.remained_share}\n')
    while terminal==0:
        agent.save_buffer([ob['reward'], ob['states'], ob['isdone']],
                          False)
        pool.add_sample(agent.tmp_buffer[0],
                        agent.tmp_buffer[1],
                        agent.tmp_buffer[2],
                        agent.tmp_buffer[3],
                        agent.tmp_buffer[4])
        act = agent.get_action(ob['states'])
        print(act)
        ob = env.step(act)
        terminal = ob['isdone']
        print(f'observation is {ob}\nremained shares: {env.remained_share}\n')
    if i >5:
        s0, acts, r, s1, ter = pool.random_batch(1)
        agent.train(s0, acts, r, s1, ter)


# if __name__ == '__main__':
#     #main()
