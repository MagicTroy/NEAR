"""
DeepCoNN model
@author:
Sixun Ouyang (sixun.ouyang@insight-centre.org)
"""

import tensorflow as tf


class DeepCoNN(object):
    def __init__(self, param):
        self.user_length = param['user_review_length']
        self.item_length = param['item_review_length']
        self.user_vocab_size = param['user_vocab_size']
        self.item_vocab_size = param['item_vocab_size']
        self.fm_k = param['fm_k']
        self.n_latent = param['n_latent']
        self.embedding_size = param['embedding_size']
        self.filter_sizes = param['filter_sizes']
        self.num_filters = param['num_filters']

        self.learning_rate = param['learning_rate']
        self.l2_reg_lambda = 0.0
        self.l2_reg_V = 0.0

        # session
        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            self._build_model()

    def __del__(self):
        self.sess.close()

    def _build_model(self):
        tf.set_random_seed(123)
        # with tf.Graph().as_default():
        self.input_u = tf.placeholder(tf.int32, [None, self.user_length], name="input_u")
        self.input_i = tf.placeholder(tf.int32, [None, self.item_length], name="input_i")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        """
        plus one in case un-know words from generated review
        """
        with tf.name_scope("user_embedding"):
            self.user_weights_lookup = tf.Variable(
                tf.random_uniform([self.user_vocab_size, self.embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_users = tf.nn.embedding_lookup(self.user_weights_lookup, self.input_u)
            print(self.embedded_users)

            expand_embedded_users = tf.expand_dims(self.embedded_users, -1)
            # print(self.embedded_users)

        with tf.name_scope("item_embedding"):
            self.item_weights_lookup = tf.Variable(
                tf.random_uniform([self.item_vocab_size, self.embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_items = tf.nn.embedding_lookup(self.item_weights_lookup, self.input_i)
            expand_embedded_items = tf.expand_dims(self.embedded_items, -1)

        self.h_list = []
        self.pooled_outputs_u = []

        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("user_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    expand_embedded_users,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                print('conv', conv)

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                self.h_list.append(h)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.user_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")

                print('pool', pooled)
                self.pooled_outputs_u.append(pooled)
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool_u = tf.concat(self.pooled_outputs_u, 3)
        self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, num_filters_total])

        pooled_outputs_i = []

        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("item_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    expand_embedded_items,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.item_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_i.append(pooled)
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool_i = tf.concat(pooled_outputs_i, 3)
        self.h_pool_flat_i = tf.reshape(self.h_pool_i, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.h_pool_flat_u, 1.0)
            self.h_drop_i = tf.nn.dropout(self.h_pool_flat_i, 1.0)
        with tf.name_scope("get_fea"):
            self.Wu = tf.get_variable(
                "Wu",
                shape=[num_filters_total, self.n_latent],
                initializer=tf.contrib.layers.xavier_initializer())
            bu = tf.Variable(tf.constant(0.1, shape=[self.n_latent]), name="bu")
            self.u_fea = tf.matmul(self.h_drop_u, self.Wu) + bu
            # self.u_fea = tf.nn.dropout(self.u_fea,self.dropout_keep_prob)
            self.Wi = tf.get_variable(
                "Wi",
                shape=[num_filters_total, self.n_latent],
                initializer=tf.contrib.layers.xavier_initializer())
            bi = tf.Variable(tf.constant(0.1, shape=[self.n_latent]), name="bi")
            self.i_fea = tf.matmul(self.h_drop_i, self.Wi) + bi
            # self.i_fea=tf.nn.dropout(self.i_fea,self.dropout_keep_prob)

        with tf.name_scope('fm'):
            self.z = tf.nn.relu(tf.concat([self.u_fea, self.i_fea], 1))

            # self.z=tf.nn.dropout(self.z,self.dropout_keep_prob)

            WF1 = tf.Variable(
                tf.random_uniform([self.n_latent * 2, 1], -0.1, 0.1), name='fm1')
            Wf2 = tf.Variable(
                tf.random_uniform([self.n_latent * 2, self.fm_k], -0.1, 0.1), name='fm2')
            one = tf.matmul(self.z, WF1)

            inte1 = tf.matmul(self.z, Wf2)
            inte2 = tf.matmul(tf.square(self.z), tf.square(Wf2))

            inter = (tf.square(inte1) - inte2) * 0.5

            inter = tf.nn.dropout(inter, self.dropout_keep_prob)

            inter = tf.reduce_sum(inter, 1, keep_dims=True)
            # print inter
            b = tf.Variable(tf.constant(0.1), name='bias')

            self.predictions = one + inter + b

            # print self.predictions

        with tf.name_scope("loss"):
            # losses = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + self.l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.se = tf.square(tf.subtract(self.predictions, self.input_y))
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.rmse_cost = tf.sqrt(tf.losses.mean_squared_error(predictions=self.predictions, labels=self.input_y))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))

        # optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # init
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
        # saver
        self.saver = tf.train.Saver(max_to_keep=None)

    def get_total_feature(self):
        return self.sess.run([self.Wu, self.Wi])

    def get_feature(self, **inputs):
        user_feature, item_feature = self.sess.run([self.u_fea, self.i_fea],
                                                   feed_dict={self.input_u: inputs['user'],
                                                              self.input_i: inputs['item']})
        return user_feature, item_feature, None, None

    def get_total_weights(self):
        return self.sess.run([self.user_weights_lookup, self.item_weights_lookup])

    def get_user_weights(self, **inputs):
        user_embedding = self.sess.run(self.embedded_users,
                                       feed_dict={self.input_u: inputs['user']})
        return user_embedding

    def get_weights_bias(self, **inputs):
        user_embedding, item_embedding = self.sess.run([self.embedded_users, self.embedded_items],
                                                       feed_dict={self.input_u: inputs['user'],
                                                                  self.input_i: inputs['item']})
        return user_embedding, item_embedding, None, None

    def predict_by_weights(self, **inputs):
        return self.sess.run(self.predictions,
                             feed_dict={self.embedded_users: inputs['user_weights'],
                                        self.embedded_items: inputs['item_weights'],
                                        self.dropout_keep_prob: 1.})

    def predict_by_perturb_weights(self, **inputs):
        return self.sess.run(self.predictions,
                             feed_dict={self.u_fea: inputs['user_weights'],
                                        self.i_fea: inputs['item_weights'],
                                        self.dropout_keep_prob: 1.})

    def get_loss_by_weights(self, **inputs):
        return self.sess.run(self.se,
                             feed_dict={self.embedded_users: inputs['user_weights'],
                                        self.embedded_items: inputs['item_weights'],
                                        self.input_y: inputs['rating'], self.dropout_keep_prob: 1.})

    def train_by_modified_embeddings(self, **inputs):
        return self.sess.run([self.optimizer, self.rmse_cost],
                             feed_dict={self.input_u: inputs['user'], self.input_i: inputs['item'],
                                        self.embedded_users: inputs['user_weights'],
                                        self.embedded_items: inputs['item_weights'],
                                        self.input_y: inputs['rating'],
                                        self.dropout_keep_prob: inputs['drop_out']})

    def train_by_modified_feature(self, **inputs):
        return self.sess.run([self.optimizer, self.rmse_cost],
                             feed_dict={self.input_u: inputs['user'], self.input_i: inputs['item'],
                                        self.u_fea: inputs['user_weights'],
                                        self.i_fea: inputs['item_weights'],
                                        self.input_y: inputs['rating'],
                                        self.dropout_keep_prob: inputs['drop_out']})

    def train(self, **inputs):
        return self.sess.run(
            [self.optimizer, self.loss, self.rmse_cost, self.accuracy],
            feed_dict={self.input_u: inputs['user'], self.input_i: inputs['item'],
                       self.input_y: inputs['rating'], self.dropout_keep_prob: inputs['drop_out']})

    def score(self, **inputs):
        return self.sess.run(
            [self.loss, self.mae, self.accuracy],
            feed_dict={self.input_u: inputs['user'], self.input_i: inputs['item'], self.input_y: inputs['rating'],
                       self.dropout_keep_prob: 1.})

    def predict(self, **inputs):
        return self.sess.run(
            self.predictions,
            feed_dict={self.input_u: inputs['user'], self.input_i: inputs['item'], self.dropout_keep_prob: 1.})

    def save(self, save_path):
        self.saver.save(sess=self.sess, save_path=save_path)

    def restore(self, save_path):
        self.saver.restore(sess=self.sess, save_path=save_path)
