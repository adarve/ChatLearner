import tensorflow as tf
import chatbot.modelhelper as model_helper


class DoubleEncoderDiscriminator(object):
    def __init__(self, generator, sample_id):
        self.global_step = generator.global_step
        self.training = generator.training
        self.batch_input = generator.batch_input
        self.vocab_size = generator.vocab_size
        self.hparams = generator.hparams
        self.time_major = self.hparams.time_major
        self.sample_id = tf.transpose(sample_id)
        self.reverse_vocab_table = generator.reverse_vocab_table

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
            self.gan_loss = -tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_fake, labels=ones))

        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer()
            gradients = tf.gradients(self.loss, discrim_tvars)
            clipped_gradients, _ = model_helper.gradient_clip(gradients,
                                                              max_gradient_norm=self.hparams.max_gradient_norm)

            self.update = discrim_optim.apply_gradients(zip(clipped_gradients, discrim_tvars))

    def metrics(self):
        accuracy_real = tf.summary.scalar("disc_accuracy_real", self.accuracy_real[1]),
        accuracy_fake = tf.summary.scalar("disc_accuracy_fake", self.accuracy_fake[1]),
        #loss_real = tf.summary.scalar("disc_loss_real", self.loss_real),
        #loss_fake = tf.summary.scalar("disc_loss_fake", self.loss_fake),
        loss = tf.summary.scalar("disc_loss", self.loss),
        source_words = self.reverse_vocab_table.lookup(tf.to_int64(self.batch_input.original_source[:5]))
        source_words = tf.summary.text("source", source_words)
        target_words = self.reverse_vocab_table.lookup(tf.to_int64(self.batch_input.original_target[:5]))
        target_words = tf.summary.text("target", target_words)
        predicted_words = self.reverse_vocab_table.lookup(tf.to_int64(self.sample_id[:5]))
        predicted_words = tf.summary.text("predicted", predicted_words)
        return [ accuracy_real, accuracy_fake, loss, source_words, target_words, predicted_words ]

    def get_initializer(self):
        return tf.no_op()

    def _build_disc(self, source, target):
        embedding = model_helper.create_embedding(vocab_size=self.vocab_size,
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


class AvgEmbDiscriminator(object):
    def __init__(self, generator, sample_id):
        self.global_step = generator.global_step
        self.training = generator.training
        self.batch_input = generator.batch_input
        self.vocab_list = generator.vocab_list
        self.vocab_size = generator.vocab_size
        self.hparams = generator.hparams
        self.time_major = self.hparams.time_major
        self.sample_id = tf.transpose(sample_id)
        self.reverse_vocab_table = generator.reverse_vocab_table
        #self.embedding = generator.embedding

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
            self.gan_loss = -loss_fake

        with tf.name_scope("discriminator_train"):
            #discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer()
            #gradients = tf.gradients(self.loss, discrim_tvars)
            #clipped_gradients, _ = model_helper.gradient_clip(gradients,
            #                                                  max_gradient_norm=self.hparams.max_gradient_norm)
            #self.update = discrim_optim.apply_gradients(zip(gradients, discrim_tvars), global_step=self.global_step)
            self.update = discrim_optim.minimize(self.loss, global_step=self.global_step)

    def metrics(self):
        accuracy_real = tf.summary.scalar("disc_accuracy_real", self.accuracy_real[1]),
        accuracy_fake = tf.summary.scalar("disc_accuracy_fake", self.accuracy_fake[1]),
        #loss_real = tf.summary.scalar("disc_loss_real", self.loss_real),
        #loss_fake = tf.summary.scalar("disc_loss_fake", self.loss_fake),
        loss = tf.summary.scalar("disc_loss", self.loss),
        source_words = self.reverse_vocab_table.lookup(tf.to_int64(self.batch_input.original_source[:5]))
        source_words = tf.summary.text("source", source_words)
        target_words = self.reverse_vocab_table.lookup(tf.to_int64(self.batch_input.original_target[:5]))
        target_words = tf.summary.text("target", target_words)
        predicted_words = self.reverse_vocab_table.lookup(tf.to_int64(self.sample_id[:5]))
        predicted_words = tf.summary.text("predicted", predicted_words)
        return [ accuracy_real, accuracy_fake, loss, source_words, target_words, predicted_words ]

    def _build_disc(self, source, target):
        self._load_embedding()
        source_emb = tf.nn.embedding_lookup(self.embedding, source)
        source_emb = tf.reduce_max(source_emb, 1)
        target_emb = tf.nn.embedding_lookup(self.embedding, target)
        target_emb = tf.reduce_max(target_emb, 1)
        #final = source_emb - target_emb
        final = tf.concat([source_emb, target_emb], 1)
        return tf.layers.dense(final, 2)

    def _load_embedding(self):
        self.embedding = model_helper.create_embedding(self.vocab_size, 300, trainable=False)
        pass

    def get_initializer(self):
        from settings import PROJECT_ROOT
        import os
        pretrained_embeddings_file = os.path.join(PROJECT_ROOT, 'Data', 'Corpus', 'lexvec.enwiki+newscrawl.300d.W.pos.vectors')
        return model_helper.populate_embedding(self.embedding, self.vocab_list, pretrained_embeddings_file)
        #return tf.no_op()

