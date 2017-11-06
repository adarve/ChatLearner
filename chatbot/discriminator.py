import tensorflow as tf
import chatbot.modelhelper as model_helper


EPS = 1e-12


class Discriminator(object):
    def __init__(self, generator, sample_id):
        self.global_step = generator.global_step
        self.training = generator.training
        self.batch_input = generator.batch_input
        self.vocab_size = generator.vocab_size
        self.hparams = generator.hparams
        self.time_major = self.hparams.time_major
        self.sample_id = tf.transpose(sample_id)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                self.predict_real = self._build_disc(self.batch_input.original_source,
                                                     self.batch_input.original_target)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                self.predict_fake = self._build_disc(self.batch_input.original_source, self.sample_id)

        with tf.name_scope("discriminator_loss"):
            zeros = tf.zeros(tf.shape(self.predict_real[:,0]), dtype=tf.int32)
            loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_real, labels=zeros))
            self.accuracy_real = tf.metrics.accuracy(zeros, tf.argmax(self.predict_real, 1))

            ones = tf.ones(tf.shape(self.predict_fake[:,0]), dtype=tf.int32)
            loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_fake, labels=ones))
            self.accuracy_fake = tf.metrics.accuracy(ones, tf.argmax(self.predict_fake, 1))

            self.loss = loss_real + loss_fake
            self.gan_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_fake, labels=zeros))

        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer()
            discrim_grads_and_vars = discrim_optim.compute_gradients(self.loss,
                                                                     var_list=discrim_tvars)
            self.update = discrim_optim.apply_gradients(discrim_grads_and_vars)

    def metrics(self):
        accuracy_real = tf.summary.scalar("disc_accuracy_real", self.accuracy_real[1]),
        accuracy_fake = tf.summary.scalar("disc_accuracy_fake", self.accuracy_fake[1]),
        #loss_real = tf.summary.scalar("disc_loss_real", self.loss_real),
        #loss_fake = tf.summary.scalar("disc_loss_fake", self.loss_fake),
        loss = tf.summary.scalar("disc_loss", self.loss),
        return [ accuracy_real, accuracy_fake, loss ]

    def _build_disc(self, source, target):
        embedding = model_helper.create_embbeding(vocab_size=self.vocab_size,
                                                  embed_size=512)

        with tf.variable_scope("encoder"):
            _, encoder_source = self._build_encoder(embedding, source)
        with tf.variable_scope("encoder", reuse=True):
            _, encoder_target = self._build_encoder(embedding, target)
        encoder = tf.concat([encoder_source, encoder_target], 1)
        #layer = tf.layers.dense(encoder, 100, activation=tf.nn.relu)
        #return tf.layers.dense(layer, 2, activation=tf.nn.softmax)
        #return tf.layers.dense(encoder, 2, activation=tf.nn.softmax)
        return tf.layers.dense(encoder, 2)


    def _build_encoder(self, embedding, input):
        if self.time_major:
            input = tf.transpose(input)
        encoder_emb_inp = tf.nn.embedding_lookup(embedding, input)
        cell = model_helper.create_rnn_cell(512, 1, 0.9)
        return tf.nn.dynamic_rnn(
            cell,
            encoder_emb_inp,
            dtype=tf.float32,
            time_major=self.time_major)


