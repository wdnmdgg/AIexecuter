from env import Env
from agent import LSTMAgent
from pool import SimpleReplayPool
import os
from matplotlib import pyplot as plt
import shift
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import Q_function as Qf
import time

GAMMA = 0.95
EPSILON = 0.1
EPISODES = 500

# Process the loading files
model_list = os.listdir('saved_models')
trained_model_num = '0'
if model_list:
    model_list.remove('checkpoint')
    for name in model_list:
        trained_model_num = max(name.split('.')[0], trained_model_num)
    load = True
else:
    load = False
# Trader connection
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
execute_time = 60     # total execution time (seconds)
exe_times = 30        # steps
exe_interval = execute_time / exe_times
action_space = 51
exe_shares = 100*30     # shares
exe_price = 166         # objPrice dollar
episode_list = []       # to be continued
batch_size = 10
num_states = 80
loss_list = []
reward_list = []


#def main():
env = Env(trader=trader,
          symbol=symbol,
          commission=commission,
          action_space=action_space,
          share=exe_shares,
          time_total=execute_time,
          time_steps=exe_times,
          objPrice=exe_price,
          close_price_volumn=num_states)
agent = LSTMAgent(sess_=sess,
                  observations_dim=num_states+2,

                  action_space=action_space,
                  batch_size=batch_size,
                  Q_function=Qf.ann,
                  optimizer=tf.train.AdamOptimizer,
                  GAMMA=GAMMA,
                  EPSILON=EPSILON,
                  LOAD=load,
                  learning_rate=0.001)
pool = SimpleReplayPool(max_pool_size=1000,
                        pop_size=100)

for i in range(EPISODES):
    # Deal with the initialization for each episode
    print("*"*100)
    print(f'THE {i+1} EPISODE \n\n')
    sum_rew4epi = 0
    if i % 2 == 1:
        bp = trader.get_best_price(symbol)
        exe_price = bp.get_bid_price()
        env.set_objective(share=-exe_shares,
                          time_total=execute_time,
                          time_steps=exe_times,
                          objPrice=exe_price,
                          close_price_volumn=num_states)
        print(f'SELL {exe_shares} SHARES\n\n')
    else:
        bp = trader.get_best_price(symbol)
        exe_price = bp.get_ask_price()
        env.set_objective(share=exe_shares,
                          time_total=execute_time,
                          time_steps=exe_times,
                          objPrice=exe_price,
                          close_price_volumn=num_states)
        print(f'BUY {exe_shares} SHARES\n\n')
    print("="*30)
    ob = env.reset()
    act = agent.get_action(ob['states'])
    #print(act)
    ob = env.step(act)
    terminal = ob['isdone']
    print(f'observation is {ob}\nremained shares: {env.remained_share}\n')
    sum_rew4epi += ob['reward']
    while True:
        print('='*30)
        agent.save_buffer([ob['reward'], ob['states'], ob['isdone']],
                          False)
        pool.add_sample(agent.tmp_buffer[0],
                        agent.tmp_buffer[1],
                        agent.tmp_buffer[2],
                        agent.tmp_buffer[3],
                        agent.tmp_buffer[4])
        if terminal == 1:
            print(f"Total Reward For This Episode: {sum_rew4epi}")
            reward_list.append(sum_rew4epi)
            break
        act = agent.get_action(ob['states'])
        #print(act)
        ob = env.step(act)
        terminal = ob['isdone']
        print(f'observation is {ob}\nremained shares: {env.remained_share}\n')
        sum_rew4epi += ob['reward']
    if batch_size < pool.size:
        s0, acts, r, s1, ter = pool.random_batch(batch_size)
        agent.train(s0, acts, r, s1, ter)
        loss_list.append(agent.lossrc)

plt.subplot(2, 1, 1)
plt.plot(reward_list)
plt.title('Reward and Loss')
plt.ylabel('Reward')

plt.subplot(2, 1, 2)
plt.plot(loss_list)
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.show()

agent.saver.save(sess, f'./saved_models/{int(trained_model_num)+1}')


# if __name__ == '__main__':
#     main()
