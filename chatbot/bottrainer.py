# Copyright 2017 Bo Shao. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
import os
import time
import sys
from datetime import datetime

import tensorflow as tf
from chatbot.tokenizeddata import TokenizedData
from chatbot.modelcreator import ModelCreator
from chatbot.hparams import HParams

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BotTrainer(object):
    def __init__(self, corpus_dir, hparams_dir=None):
        hparams = HParams(hparams_dir).hparams if hparams_dir else None
        self.graph = tf.Graph()
        with self.graph.as_default():
            tokenized_data = TokenizedData(corpus_dir=corpus_dir, hparams=hparams)

            self.hparams = tokenized_data.hparams
            self.train_batch = tokenized_data.get_training_batch()
            self.model = ModelCreator(training=True, tokenized_data=tokenized_data,
                                      batch_input=self.train_batch)


    def train(self, result_dir, target=""):
        """Train a seq2seq model."""
        # Summary writer
        summary_name = "train_log_{}".format(datetime.utcnow())
        summary_writer = tf.summary.FileWriter(os.path.join(result_dir, summary_name), self.graph)

        log_device_placement = self.hparams.log_device_placement
        num_epochs = self.hparams.num_epochs

        config_proto = tf.ConfigProto(log_device_placement=log_device_placement,
                                      allow_soft_placement=True)
        config_proto.gpu_options.allow_growth = True

        with tf.Session(target=target, config=config_proto, graph=self.graph) as sess:
            self.model.create_or_load_model(result_dir, sess)
            global_step = self.model.global_step.eval(session=sess)
            print("# Global step = {:5d}".format(global_step))

            # Initialize all of the iterators
            sess.run(self.train_batch.initializer)

            # Initialize the statistic variables
            ckpt_loss, disc_loss, ckpt_predict_count = 0.0, 0.0, 0.0
            train_perp = 2000.0
            train_epoch = 0

            print("# Training loop started @ {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            epoch_start_time = time.time()
            while train_epoch < num_epochs:
                # Each run of this while loop is a training step, multiple time/steps will trigger
                # the train_epoch to be increased.
                learning_rate = self._get_learning_rate(train_perp)

                try:
                    #batch = sess.run(self.model.batch_input)
                    #sample_id, source, target = sess.run([self.model.sample_id,
                    #            self.model.batch_input.original_source,
                    #            self.model.batch_input.original_target])

                    step_result = self.model.train_step(sess, learning_rate=learning_rate,
                                                    train_discriminator=(train_epoch % 5 == 0))
                                                    #train_discriminator=True)
                    (_, step_disc_loss, step_loss, step_predict_count, step_summary, global_step,
                     step_word_count, batch_size) = step_result

                    #print("# Gen loss = {}".format(step_loss))
                    #print("# Disc loss = {}".format(disc_loss))
                    #sample_id = sess.run(self.model.sample_id)

                    #print(sess.run(self.model.disc.predict_real))
                    #import ipdb; ipdb.set_trace()
                    #print(sess.run(self.model.disc.predict_fake))
                    #import ipdb; ipdb.set_trace()

                    # Write step summary.
                    summary_writer.add_summary(step_summary, global_step)

                    # update statistics
                    ckpt_loss += (step_loss * batch_size)
                    disc_loss += step_disc_loss
                    ckpt_predict_count += step_predict_count

                except tf.errors.OutOfRangeError:
                    # Finished going through the training dataset. Go to next epoch.
                    train_epoch += 1

                    mean_loss = ckpt_loss / ckpt_predict_count
                    train_perp = math.exp(float(mean_loss)) if mean_loss < 300 else math.inf

                    epoch_dur = time.time() - epoch_start_time
                    print("# Finished epoch {:2d} @ step {:5d} @ {}. In the epoch, learning rate = {:.6f}, "
                          "mean loss = {:.4f}, perplexity = {:8.4f}, and {:.2f} seconds elapsed."
                          .format(train_epoch, global_step, time.strftime("%Y-%m-%d %H:%M:%S"),
                                  learning_rate, mean_loss, train_perp, round(epoch_dur, 2)))
                    print("# disc loss = {:.4f}".format(disc_loss))
                    epoch_start_time = time.time()  # The start time of the next epoch

                    summary = tf.Summary(value=[tf.Summary.Value(tag="train_perp", simple_value=train_perp)])
                    summary_writer.add_summary(summary, global_step)

                    # Save checkpoint
                    self.model.saver.save(sess, os.path.join(result_dir, "basic"), global_step=global_step)

                    ckpt_loss, disc_loss, ckpt_predict_count = 0.0, 0.0, 0.0

                    sess.run(self.model.batch_input.initializer)
                    continue

            # Done training
            self.model.saver.save(sess, os.path.join(result_dir, "basic"), global_step=global_step)
            summary_writer.close()

    @staticmethod
    def _get_learning_rate(perplexity):
        if perplexity <= 1.48:
            return 9.6e-5
        elif perplexity <= 1.64:
            return 1e-4
        elif perplexity <= 2.0:
            return 1.2e-4
        elif perplexity <= 2.4:
            return 1.6e-4
        elif perplexity <= 3.2:
            return 2e-4
        elif perplexity <= 4.8:
            return 2.4e-4
        elif perplexity <= 8.0:
            return 3.2e-4
        elif perplexity <= 16.0:
            return 4e-4
        elif perplexity <= 32.0:
            return 6e-4
        else:
            return 8e-4

if __name__ == "__main__":
    from settings import PROJECT_ROOT

    corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')
    res_dir = os.path.join(PROJECT_ROOT, 'Data', sys.argv[1])
    bt = BotTrainer(corpus_dir=corp_dir, hparams_dir=res_dir)
    bt.train(res_dir)
