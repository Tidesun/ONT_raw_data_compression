import tensorflow as tf
import tensorflow_compression as tfc
# architecture of the neural network
def make_analysis_transform(latent_dims):
    return tf.keras.Sequential([
      tf.keras.layers.Conv1D(
          16, 3, 
          activation="leaky_relu", name="conv_1_1",padding='same'),
      tf.keras.layers.Conv1D(
          16, 3, 
          activation="leaky_relu", name="conv_1_2",padding='same'),
      tf.keras.layers.MaxPooling1D(pool_size=2),
      tf.keras.layers.Conv1D(
          32, 3, activation="leaky_relu", name="conv_2_1",padding='same'),
      tf.keras.layers.Conv1D(
          32, 3, activation="leaky_relu", name="conv_2_2",padding='same'),
      tf.keras.layers.MaxPooling1D(pool_size=2),
      tf.keras.layers.Conv1D(
          64, 3, 
          activation="leaky_relu", name="conv_3_1",padding='same'),
      tf.keras.layers.Conv1D(
          64, 3, 
          activation="leaky_relu", name="conv_3_2",padding='same'),
      tf.keras.layers.MaxPooling1D(pool_size=2),
      tf.keras.layers.Conv1D(
          128, 3, activation="leaky_relu", name="conv_4_1",padding='same'),
      tf.keras.layers.Conv1D(
          128, 3, activation="leaky_relu", name="conv_4_2",padding='same'),
      tf.keras.layers.MaxPooling1D(pool_size=2),
      tf.keras.layers.Conv1D(
          256, 3, activation="leaky_relu", name="conv_5_1",padding='same'),
      tf.keras.layers.Conv1D(
          256, 3, activation="leaky_relu", name="conv_5_2",padding='same'),
      tf.keras.layers.MaxPooling1D(pool_size=2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          512, use_bias=True, activation="leaky_relu", name="fc_1"),
      tf.keras.layers.Dense(
          latent_dims, use_bias=True, activation=None, name="fc_2"), # 128
  ], name="analysis_transform")

def make_synthesis_transform(latent_dims):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(
          512, use_bias=True, activation="leaky_relu", name="fc_1"),
      tf.keras.layers.Dense(
          window_size * latent_dims, use_bias=True, activation="leaky_relu", name="fc_2"),
      tf.keras.layers.Reshape((window_size,latent_dims)),
#       tf.keras.layers.Conv1DTranspose(
#           256, 3,activation="leaky_relu", name="deconv_1_1"),
#       tf.keras.layers.Conv1DTranspose(
#           256, 3,activation="leaky_relu", name="deconv_1_2"),
#       tf.keras.layers.UpSampling1D(2),
      tf.keras.layers.Conv1DTranspose(
          128, 3,activation="leaky_relu", name="deconv_2_1",padding='same'),
      tf.keras.layers.Conv1DTranspose(
          128, 3,activation="leaky_relu", name="deconv_2_2",padding='same'),
#       tf.keras.layers.UpSampling1D(2),
      tf.keras.layers.Conv1DTranspose(
          64, 3,activation="leaky_relu", name="deconv_3_1",padding='same'),
      tf.keras.layers.Conv1DTranspose(
          64, 3,activation="leaky_relu", name="deconv_3_2",padding='same'),
#       tf.keras.layers.UpSampling1D(2),
      tf.keras.layers.Conv1DTranspose(
          32, 3,activation="leaky_relu", name="deconv_4_1",padding='same'),
      tf.keras.layers.Conv1DTranspose(
          32, 3,activation="leaky_relu", name="deconv_4_2",padding='same'),
#       tf.keras.layers.UpSampling1D(2),
      tf.keras.layers.Conv1DTranspose(
          1, 3,activation="leaky_relu", name="deconv_5_1",padding='same'),
      tf.keras.layers.Conv1DTranspose(
          1, 3,activation="leaky_relu", name="deconv_5_2",padding='same'),
  ], name="synthesis_transform")

def make_classifier():
    return tf.keras.Sequential([
      tf.keras.layers.Dense(
          512, use_bias=True, activation="leaky_relu", name="fc_1"),
      tf.keras.layers.Dense(
          64*5, activation='linear', name="fc_2"),
      tf.keras.layers.Reshape((64,5)),
  ], name="classifier")
def pass_through_loss(_, x):
    return x
def ctc_loss(y_true,y_pred):
    return tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        logit_length=tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1]),
#         ignore_longer_outputs_than_inputs=True,
        logits_time_major=False,
        blank_index=0,
        label_length=None)
def get_distortion_loss(x,x_tilde):
    distortion = tf.reduce_mean(abs(x - x_tilde))
    return distortion
def get_rate_loss(rate):
    return tf.reduce_mean(rate)

# for training
class ONTRawSignalsCompressionTrainer(tf.keras.Model):
    def __init__(self, latent_dims):
        super().__init__()
        self.analysis_transform = make_analysis_transform(latent_dims)
        self.synthesis_transform = make_synthesis_transform(latent_dims)
        self.prior_log_scales = tf.Variable(tf.zeros((latent_dims,)))
        self.classifier = make_classifier()
      @property
      def prior(self):
        return tfc.NoisyLogistic(loc=0., scale=tf.exp(self.prior_log_scales))

      def call(self, x, training):
        """Computes rate and distortion losses."""
        y = self.analysis_transform(x)
        entropy_model = tfc.ContinuousBatchedEntropyModel(
            self.prior, coding_rank=1, compression=False)
        y_tilde, rate = entropy_model(y, training=training)
        x_tilde = self.synthesis_transform(y_tilde)
        # Average number of bits per MNIST digit.
        rate = tf.reduce_mean(rate)
        # Mean absolute difference across pixels.
        distortion = tf.reduce_mean(abs(x - x_tilde))
        base_probability = self.classifier(y)
        return dict(rate=rate, distortion=distortion,base_probability=base_probability)
def train(training_dataset,validation_dataset):
    latent_dims = 64
    lmbda = 2000
    num_epochs = 1000
    trainer = ONTRawSignalsCompressionTrainer(latent_dims)
    trainer.compile(
        optimizer=tf.keras.optimizers.Adadelta(),
        # Just pass through rate and distortion as losses/metrics.
        loss=dict(rate=pass_through_loss, distortion=pass_through_loss,base_probability=ctc_loss),
        # metrics=dict(rate=pass_through_loss, distortion=pass_through_loss,base_probability=ctc_loss),
        loss_weights=dict(rate=1., distortion=lmbda,base_probability=5),
    )
    history = \
    trainer.fit(training_dataset,validation_data=validation_dataset,steps_per_epoch=10000,epochs=num_epochs)
