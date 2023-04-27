import tensorflow as tf
import tensorflow_compression as tfc
# loss
def pass_through_loss(_, x):
    return x
def ctc_loss(y_true,y_pred):
    return tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        logit_length=tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1]),
#         ignore_longer_outputs_than_inputs=True,
        logits_time_major=False,
        blank_index=-1,
        label_length=None)
# for evaluation
class ONTRawSignalsCompressor(tf.keras.Model):

    def __init__(self, analysis_transform, entropy_model):
        super().__init__()
        self.analysis_transform = analysis_transform
        self.entropy_model = entropy_model

    def call(self, x):
        y = self.analysis_transform(x)
        _, bits = self.entropy_model(y, training=False)
        return self.entropy_model.compress(y), bits
class ONTRawSignalsDecompressor(tf.keras.Model):

    def __init__(self, entropy_model, synthesis_transform):
        super().__init__()
        self.entropy_model = entropy_model
        self.synthesis_transform = synthesis_transform

    def call(self, string):
        y_hat = self.entropy_model.decompress(string, ())
        x_hat = self.synthesis_transform(y_hat)
        return x_hat
class ONTRawSignlsClassifier(tf.keras.Model):
    def __init__(self, classifier, analysis_transform):
        super().__init__()
        self.classifier = classifier
        self.analysis_transform = analysis_transform

    def call(self, x):
        y = self.analysis_transform(x)
        base_probability = self.classifier(y)
        return base_probability
def make_raw_signals_codec(trainer):
    entropy_model = tfc.ContinuousBatchedEntropyModel(trainer.prior, coding_rank=1, compression=True)
    compressor = ONTRawSignalsCompressor(trainer.analysis_transform, entropy_model)
    decompressor = ONTRawSignalsDecompressor(entropy_model, trainer.synthesis_transform)
    classifier = ONTRawSignlsClassifier(trainer.classifier,trainer.analysis_transform)
    return compressor, decompressor,classifier
def predict(test_case_X,test_case_y,pre_trained_model_path='pretrained/lmbda_2000/'):
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
    trainer.load_weights(pre_trained_model_path)
    
    compressor, decompressor,classifier = make_raw_signals_codec(trainer)
    strings, entropies = compressor(test_case_X)
    reconstructions = decompressor(strings)
    rate = get_rate_loss(entropies).numpy()
    pred = classifier(test_case_X)