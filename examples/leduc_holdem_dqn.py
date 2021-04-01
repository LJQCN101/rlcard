''' An example of learning a Deep-Q Agent on Leduc Hold'em
'''

import tensorflow as tf
import os

import rlcard
from rlcard import models
from rlcard.agents import DQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

def main():
    # Make environment
    env = rlcard.make('leduc-holdem', config={'seed': 0, 'env_num': 4})
    eval_env = rlcard.make('leduc-holdem', config={'seed': 0, 'env_num': 4})

    # Set the iterations numbers and how frequently we evaluate the performance
    evaluate_every = 100
    evaluate_num = 10000
    episode_num = 800000

    # The intial memory size
    memory_init_size = 1000

    # Train the agent every X steps
    train_every = 1

    _reward_max = -0.5

    # The paths for saving the logs and learning curves
    log_dir = './experiments/leduc_holdem_dqn_result/'

    # Set a global seed
    set_global_seed(0)

    with tf.Session() as sess:

        # Initialize a global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Set up the agents
        agent = DQNAgent(sess,
                         scope='dqn',
                         action_num=env.action_num,
                         replay_memory_init_size=memory_init_size,
                         train_every=train_every,
                         state_shape=env.state_shape,
                         mlp_layers=[128,128])
        # random_agent = RandomAgent(action_num=eval_env.action_num)
        cfr_agent = models.load('leduc-holdem-cfr').agents[0]
        env.set_agents([agent, agent])
        eval_env.set_agents([agent, cfr_agent])

        # Initialize global variables
        sess.run(tf.global_variables_initializer())

        # Init a Logger to plot the learning curve
        logger = Logger(log_dir)

        saver = tf.train.Saver()
        save_dir = 'models/leduc_holdem_dqn'
        saver.restore(sess, os.path.join(save_dir, 'model'))

        for episode in range(episode_num):

            # Generate data from the environment
            trajectories, _ = env.run(is_training=True)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                _reward = tournament(eval_env, evaluate_num)[0]
                logger.log_performance(episode, _reward)
                if _reward > _reward_max:
                    # Save model
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    saver.save(sess, os.path.join(save_dir, 'model'))
                    _reward_max = _reward

        # Close files in the logger
        logger.close_files()

        # Plot the learning curve
        logger.plot('DQN')

if __name__ == '__main__':
    main()