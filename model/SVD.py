import numpy as np
import tensorflow as tf


class SVD(object):
    def __init__(self, param):
        self.num_user = param['num_user']
        self.num_item = param['num_item']
        self.embedding_size = param['embedding_size']
        self.reg = param['reg']
        self.initial_learning_rate = param['initial_learning_rate']
        self.decay_steps = param['decay_steps']
        self.decay_rate = param['decay_rate']
        self.global_step = param['global_step']

        # build model
        self.graph = tf.Graph()
        self.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.config.gpu_options.allow_growth = True
        # session
        self.sess = tf.Session(graph=self.graph, config=self.config)
        with self.graph.as_default():
            self._build_model()

    def __del__(self):
        self.sess.close()

    def _build_model(self):
        tf.set_random_seed(123)
        with tf.name_scope("inputs"):
            self.user_batch = tf.placeholder(tf.int32, shape=[None], name="user_id")
            self.item_batch = tf.placeholder(tf.int32, shape=[None], name="item_id")
            self.rating_batch = tf.placeholder(tf.float32, shape=[None], name="true_rating")

            # # factor weights of user item weights and bias
            # self.u_w_weights = tf.placeholder(
            #     tf.float32, shape=[None, self.embedding_size], name='u_w_weights')
            # self.i_w_weights = tf.placeholder(
            #     tf.float32, shape=[None, self.embedding_size], name='i_w_weights')
            #
            # self.u_b_weights = tf.placeholder(
            #     tf.float32, shape=[None], name='u_b_weights')
            # self.i_b_weights = tf.placeholder(
            #     tf.float32, shape=[None], name='i_b_weights')

        with tf.variable_scope("variables"):
            global_bias = tf.get_variable(name="global_bias", shape=[])
            self.user_bias_lookup = tf.get_variable(name="user_bias_lookup",
                                                    initializer=tf.truncated_normal(shape=[self.num_user], stddev=0.02))
            self.item_bias_lookup = tf.get_variable(name="item_bias_lookup",
                                                    initializer=tf.truncated_normal(shape=[self.num_item], stddev=0.02))
            self.user_bias = tf.nn.embedding_lookup(self.user_bias_lookup, self.user_batch, name="user_bias")
            self.item_bias = tf.nn.embedding_lookup(self.item_bias_lookup, self.item_batch, name="item_bias")
            self.user_weights_lookup = tf.get_variable(name="user_embedding_lookup",
                                                       initializer=tf.truncated_normal(
                                                           shape=[self.num_user, self.embedding_size], stddev=0.02))
            self.item_weights_lookup = tf.get_variable(name="item_embedding_lookup",
                                                       initializer=tf.truncated_normal(
                                                           shape=[self.num_item, self.embedding_size], stddev=0.02))
            self.user_weights = tf.nn.embedding_lookup(self.user_weights_lookup, self.user_batch,
                                                       name="user_embedding")

            self.item_weights = tf.nn.embedding_lookup(self.item_weights_lookup, self.item_batch,
                                                       name="item_embedding")

        with tf.name_scope("predict"):
            # infer = tf.reduce_mean(tf.matmul(self.user_weights, tf.transpose(self.item_weights)), axis=1)
            # shape: [None]
            # u_w = tf.multiply(self.user_weights, self.u_w_weights)
            # i_w = tf.multiply(self.item_weights, self.i_w_weights)
            # u_b = tf.multiply(self.user_bias, self.u_b_weights)
            # i_b = tf.multiply(self.item_bias, self.i_b_weights)

            infer = tf.reduce_mean(tf.multiply(self.user_weights, self.item_weights), axis=1, keep_dims=False)
            infer = tf.add(infer, global_bias)
            infer = tf.add(infer, self.user_bias)
            self.infer = tf.add(infer, self.item_bias, name="predict")

            user_l2 = tf.sqrt(tf.nn.l2_loss(self.user_weights))
            item_l2 = tf.sqrt(tf.nn.l2_loss(self.item_weights))
            l2_sum = tf.add(user_l2, item_l2)
            user_bias_sq = tf.square(self.user_bias)
            item_bias_sq = tf.square(self.item_bias)
            sq_bias_sum = tf.add(user_bias_sq, item_bias_sq)
            self.regularizer = tf.add(sq_bias_sum, l2_sum, name="regularizer")

        with tf.name_scope("optimization"):
            # mse_cost = tf.losses.mean_squared_error(labels=self.rating_batch, predictions=self.infer)
            self.se = tf.square(self.rating_batch - self.infer)
            mse_cost = tf.reduce_mean(self.se)
            self.rmse_cost = tf.sqrt(mse_cost)
            reg_lambda = tf.constant(self.reg, shape=[], name="lambda")
            self.cost = tf.add(mse_cost, tf.multiply(reg_lambda, self.regularizer))

            _global_step = tf.Variable(self.global_step, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, _global_step,
                                                            self.decay_steps,
                                                            self.decay_rate, staircase=True)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            self.gradients = optimizer.compute_gradients(self.cost)
            # capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if
            #                     grad is not None]

            self.grad_descent = optimizer.apply_gradients(self.gradients, global_step=_global_step)

        # init
        self.sess.run(
            [tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])

        # saver
        self.saver = tf.train.Saver(max_to_keep=None)

    def get_total_weights(self):
        return self.sess.run([self.user_weights_lookup, self.item_weights_lookup])

    def get_total_weights_bias(self):
        return self.sess.run([self.user_weights_lookup, self.item_weights_lookup,
                              self.user_bias_lookup, self.item_bias_lookup])

    def get_weights_bias(self, **inputs):
        return self.sess.run([self.user_weights, self.item_weights, self.user_bias, self.item_bias],
                             feed_dict={self.user_batch: inputs['user'], self.item_batch: inputs['item']})

    def predict_by_perturb_weights(self, **inputs):
        return self.sess.run(self.infer,
                             feed_dict={self.user_weights: inputs['user_weights'],
                                        self.item_weights: inputs['item_weights'],
                                        self.user_bias: inputs['user_bias'],
                                        self.item_bias: inputs['item_bias']})

    def loss_by_perturb_weights(self, **inputs):
        return self.sess.run(self.se,
                             feed_dict={self.user_weights: inputs['user_weights'],
                                        self.item_weights: inputs['item_weights'],
                                        self.user_bias: inputs['user_bias'],
                                        self.item_bias: inputs['item_bias'],
                                        self.rating_batch: inputs['rating']})

    def train_by_modified_embeddings(self, **inputs):
        return self.sess.run([self.grad_descent, self.rmse_cost],
                             feed_dict={self.user_batch: inputs['user'], self.item_batch: inputs['item'],
                                        self.user_weights: inputs['user_weights'],
                                        self.item_weights: inputs['item_weights'],
                                        self.user_bias: inputs['user_bias'], self.item_bias: inputs['item_bias'],
                                        self.rating_batch: inputs['rating']})

    def train(self, **inputs):
        return self.sess.run([self.grad_descent, self.rmse_cost],
                             feed_dict={self.user_batch: inputs['user'],
                                        self.item_batch: inputs['item'],
                                        self.rating_batch: inputs['rating']})

    def score(self, **inputs):
        return self.sess.run(self.rmse_cost,
                             feed_dict={self.user_batch: inputs['user'],
                                        self.item_batch: inputs['item'],
                                        self.rating_batch: inputs['rating']})

    def predict(self, **inputs):
        return self.sess.run(self.infer,
                             feed_dict={self.user_batch: inputs['user'],
                                        self.item_batch: inputs['item']})

    def save(self, save_path):
        self.saver.save(sess=self.sess, save_path=save_path)

    def restore(self, save_path):
        self.saver.restore(sess=self.sess, save_path=save_path)
