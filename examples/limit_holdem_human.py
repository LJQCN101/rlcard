''' A toy example of playing against a random agent on Limit Hold'em
'''

import rlcard
from rlcard.agents import LimitholdemHumanAgent as HumanAgent
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils.utils import print_card

import tensorflow as tf
import os

# Make environment and enable human mode
# Set 'record_action' to True because we need it to print results
env = rlcard.make('limit-holdem', config={'record_action': True})
human_agent = HumanAgent(env.action_num)

with tf.Session() as sess:
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    agent = DQNAgent(sess,
                     scope='dqn',
                     action_num=env.action_num,
                     replay_memory_init_size=1000,
                     train_every=1,
                     state_shape=env.state_shape,
                     mlp_layers=[512, 512])

    saver = tf.train.Saver()
    save_dir = 'models/limit_holdem_dqn'
    saver.restore(sess, os.path.join(save_dir, 'model'))

    env.set_agents([human_agent, agent])

    print(">> Limit Hold'em random agent")

    while (True):
        print(">> Start a new game")

        trajectories, payoffs = env.run(is_training=False)
        # If the human does not take the final action, we need to
        # print other players action
        if len(trajectories[0]) != 0:
            final_state = trajectories[0][-1][-2]
            action_record = final_state['action_record']
            state = final_state['raw_obs']
            _action_list = []
            for i in range(1, len(action_record)+1):
                """
                if action_record[-i][0] == state['current_player']:
                    break
                """
                _action_list.insert(0, action_record[-i])
            for pair in _action_list:
                print('>> Player', pair[0], 'chooses', pair[1])

        # Let's take a look at what the agent card is
        print('=============     Random Agent    ============')
        print_card(env.get_perfect_information()['hand_cards'][1])

        print('===============     Result     ===============')
        if payoffs[0] > 0:
            print('You win {} chips!'.format(payoffs[0]))
        elif payoffs[0] == 0:
            print('It is a tie.')
        else:
            print('You lose {} chips!'.format(-payoffs[0]))
        print('')

        input("Press any key to continue...")
