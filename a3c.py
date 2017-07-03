import better_exceptions
import tensorflow as tf
import numpy as np
import gym
from tqdm import tqdm

from network import *
from async_agent import *

# A3C algorithm
def main(config,
         RANDOM_SEED,
         LOG_DIR,
         SAVE_PERIOD,
         SUMMARY_PERIOD,
         GAME,
         DISCOUNT_FACTOR,
         DEVICE,
         LEARNING_RATE,
         DECAY,
         GRAD_CLIP,
         ENTROPY_BETA,
         NUM_THREADS,
         AGENT_PER_THREADS,
         UNROLL_STEP,
         MAX_ITERATION,
         **kwargs):
    # Initialize Seed
    tf.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.reset_default_graph()

    sample_env = gym.make(GAME)
    nA = sample_env.action_space.n

    # define actor critic networks and environments
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.polynomial_decay(LEARNING_RATE,global_step,MAX_ITERATION//2,
                                              LEARNING_RATE*0.1)
    master_ac = ActorCritic(nA,device_name=DEVICE,
                            learning_rate=learning_rate,decay=DECAY,grad_clip=GRAD_CLIP,
                            entropy_beta=ENTROPY_BETA)
    group_agents = [
        A3CGroupAgent([gym.make(GAME) for _ in range(AGENT_PER_THREADS)],
                       ActorCritic(nA,master=master_ac,device_name=DEVICE,scope_name='Thread%02d'%i,
                                   learning_rate=learning_rate,decay=DECAY,grad_clip=GRAD_CLIP,
                                   entropy_beta=ENTROPY_BETA),
                       unroll_step=UNROLL_STEP,
                       discount_factor=DISCOUNT_FACTOR,
                       seed=i)
        for i in range(NUM_THREADS)]

    queue = tf.FIFOQueue(capacity=NUM_THREADS*10,
                            dtypes=[tf.float32,tf.float32,tf.float32],)
    qr = tf.train.QueueRunner(queue, [g_agent.enqueue_op(queue) for g_agent in group_agents])
    tf.train.queue_runner.add_queue_runner(qr)
    loss = queue.dequeue()

    # Miscellaneous(init op, summaries, etc.)
    increase_step = global_step.assign(global_step + 1)
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    def _train_info():
        total_eps = sum([g_agent.num_episodes() for g_agent in group_agents])
        avg_r = sum([g_agent.reward_info()[0] for g_agent in group_agents]) / len(group_agents)
        max_r = max([g_agent.reward_info()[1] for g_agent in group_agents])
        return total_eps,avg_r,max_r
    train_info = tf.py_func(_train_info,[],[tf.int64,tf.float64,tf.float64],stateful=True)
    pl, el, vl = loss
    total_eps, avg_r, max_r = train_info

    tf.summary.scalar('learning_rate',learning_rate)
    tf.summary.scalar('policy_loss',pl)
    tf.summary.scalar('entropy_loss',el)
    tf.summary.scalar('value_loss',vl)
    tf.summary.scalar('total_episodes',total_eps)
    tf.summary.scalar('average_rewards',avg_r)
    tf.summary.scalar('maximum_rewards',max_r)
    summary_op = tf.summary.merge_all()
    config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(config.as_matrix()), collections=[])

    # saver and sessions
    saver = tf.train.Saver(var_list=master_ac.train_vars,max_to_keep = 3)

    sess = tf.Session()
    sess.graph.finalize()

    sess.run(init_op)
    master_ac.initialize(sess)
    for agent in group_agents :
        agent.ac.initialize(sess)
    print('Initialize Complete...')

    try:
        summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
        summary_writer_eps = tf.summary.FileWriter(os.path.join(LOG_DIR,'per-eps'))
        summary_writer.add_summary(sess.run(config_summary))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        for step in tqdm(xrange(MAX_ITERATION)) :
            if coord.should_stop() :
                break

            (pl,el,vl), summary_str, (total_eps,avg_r,max_r),_ = sess.run([loss, summary_op, train_info, increase_step])
            if( step % SUMMARY_PERIOD == 0 ):
                summary_writer.add_summary(summary_str,step)
                summary_writer_eps.add_summary(summary_str,total_eps)
                tqdm.write('step(%7d) policy_loss:%1.5f,entropy_loss:%1.5f,value_loss:%1.5f, te:%5d avg_r:%2.1f max_r:%2.1f'%
                        (step,pl,el,vl,total_eps,avg_r,max_r))

            if( (step+1) % SAVE_PERIOD == 0 ):
                saver.save(sess,LOG_DIR+'/model.ckpt',global_step=step+1)
    except Exception, e:
        coord.request_stop(e)
    finally :
        coord.request_stop()
        coord.join(threads)

        saver.save(sess,LOG_DIR+'/last.ckpt')
        sess.close()
        #queue.close() #where should it go?

def get_default_param():
    return {
        'GAME' : 'Breakout-v0',
        'DISCOUNT_FACTOR':0.99,
        'DEVICE' : '/gpu:0',

        'SAVE_PERIOD':20000,
        'SUMMARY_PERIOD':100,

        'LEARNING_RATE':0.00025,
        'DECAY':0.99,
        'GRAD_CLIP':0.1,
        'ENTROPY_BETA':0.01,

        'NUM_THREADS':4,
        'AGENT_PER_THREADS':64,
        'UNROLL_STEP':5,
        'MAX_ITERATION':1000000
    }

if __name__ == "__main__":
    class MyConfig(dict):
        pass
    params = get_default_param()
    params.update({
        'LOG_DIR':'log/'+params['GAME'],
        'RANDOM_SEED':113,
    })
    config = MyConfig(params)
    def as_matrix() :
        return [[k, str(w)] for k, w in config.items()]
    config.as_matrix = as_matrix

    main(config=config,**config)
