''' An example of learning a Deep-Q Agent on Texas No-Limit Holdem
'''

import tensorflow as tf
import os

import rlcard
from rlcard.agents import DQNAgent, NFSPAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

def main():
    # Make environment
    env = rlcard.make('no-limit-holdem', config={'seed': 0, 'env_num': 16, 'game_player_num': 4})
    eval_env = rlcard.make('no-limit-holdem', config={'seed': 0, 'env_num': 16})

    # Set the iterations numbers and how frequently we evaluate the performance
    evaluate_every = 100
    evaluate_num = 1000
    episode_num = 200000

    # The intial memory size
    memory_init_size = 1000

    # Train the agent every X steps
    train_every = 1

    _reward_max = -0.8

    # The paths for saving the logs and learning curves
    log_dir = './experiments/nolimit_holdem_dqn_result/'

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
                         mlp_layers=[512, 512])

        agent2 = NFSPAgent(sess,
                          scope='nfsp',
                          action_num=env.action_num,
                          state_shape=env.state_shape,
                          hidden_layers_sizes=[512,512],
                          anticipatory_param=0.1,
                          min_buffer_size_to_learn=memory_init_size,
                          q_replay_memory_init_size=memory_init_size,
                          train_every = 64,
                          q_train_every=64,
                          q_mlp_layers=[512,512])

        # Initialize global variables
        sess.run(tf.global_variables_initializer())

        save_dir = 'models/nolimit_holdem_dqn'
        saver = tf.train.Saver()
        #saver.restore(sess, os.path.join(save_dir, 'model'))

        random_agent = RandomAgent(action_num=eval_env.action_num)
        env.set_agents([agent, agent, agent2, random_agent])
        eval_env.set_agents([agent, agent2])

        # Init a Logger to plot the learning curve
        logger = Logger(log_dir)

        for episode in range(episode_num):
            agent2.sample_episode_policy()
            # Generate data from the environment
            trajectories, _ = env.run(is_training=True)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent.feed(ts)

            for ts in trajectories[2]:
                agent2.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                _reward = tournament(eval_env, evaluate_num)[0]
                logger.log_performance(episode, _reward)
                if _reward > _reward_max:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    saver.save(sess, os.path.join(save_dir, 'model'))
                    _reward_max = _reward

        # Close files in the logger
        logger.close_files()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saver.save(sess, os.path.join(save_dir, 'model_final'))
    
if __name__ == '__main__':
    main()