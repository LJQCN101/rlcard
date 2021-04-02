''' A toy example of playing against pretrianed AI on Leduc Hold'em
'''
import tensorflow as tf
import os

import rlcard
from rlcard.agents import DQNAgent, NFSPAgent
from rlcard.agents import NolimitholdemHumanAgent as HumanAgent
from rlcard.utils import print_card

# Make environment
# Set 'record_action' to True because we need it to print results
env = rlcard.make('no-limit-holdem', config={'record_action': True})

human_agent = HumanAgent(env.action_num)

with tf.Session() as sess:
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agent = DQNAgent(sess,
                     scope='dqn_1',
                     action_num=env.action_num,
                     replay_memory_init_size=1000,
                     train_every=1,
                     state_shape=env.state_shape,
                     mlp_layers=[512, 512])

    agent2 = NFSPAgent(sess,
                       scope='nfsp',
                       action_num=env.action_num,
                       state_shape=env.state_shape,
                       hidden_layers_sizes=[512, 512],
                       anticipatory_param=0.1,
                       min_buffer_size_to_learn=1000,
                       q_replay_memory_init_size=1000,
                       train_every=64,
                       q_train_every=64,
                       q_mlp_layers=[512, 512])

    saver = tf.train.Saver()
    save_dir = 'models/nolimit_holdem_dqn'
    saver.restore(sess, os.path.join(save_dir, 'model'))

    env.set_agents([human_agent, agent])


    while (True):
        print(">> Start a new game")

        trajectories, payoffs = env.run(is_training=False)
        # If the human does not take the final action, we need to
        # print other players action
        if len(trajectories[0]) == 0:
            print('game over!')
            continue

        final_state = trajectories[0][-1][-2]
        action_record = final_state['action_record']
        state = final_state['raw_obs']
        _action_list = []
        for i in range(1, len(action_record)+1):
            if action_record[-i][0] == state['current_player']:
                break
            _action_list.insert(0, action_record[-i])
        for pair in _action_list:
            print('>> Player', pair[0], 'chooses', pair[1])

        # Let's take a look at what the agent card is
        print('===============     Cards all Players    ===============')
        for hands in env.get_perfect_information()['hand_cards']:
            print_card(hands)

        print('===============     Result     ===============')
        if payoffs[0] > 0:
            print('You win {} chips!'.format(payoffs[0]))
        elif payoffs[0] == 0:
            print('It is a tie.')
        else:
            print('You lose {} chips!'.format(-payoffs[0]))
        print('')

        input("Press any key to continue...")
