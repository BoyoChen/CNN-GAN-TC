import tensorflow as tf


class CompoundModel():
    def __init__(self, generator, discriminator, regressor):
        self.generator = generator
        self.discriminator = discriminator
        self.regressor = regressor
        self.use_generated_image = 'both'
        self.freeze_VIS_generator = False
        self.freeze_PMW_generator = False

    def set_use_generated_image(self, new_val):
        self.use_generated_image = new_val

    def set_freeze_VIS_generator(self, new_val):
        self.freeze_VIS_generator = new_val

    def set_freeze_PMW_generator(self, new_val):
        self.freeze_PMW_generator = new_val

    def save_weights(self, saving_path, save_format):
        self.generator.save_weights(saving_path + '/' + 'generator', save_format=save_format)
        self.discriminator.save_weights(saving_path + '/' + 'discriminator', save_format=save_format)
        self.regressor.save_weights(saving_path + '/' + 'regressor', save_format=save_format)

    def generate_noon_image(self, image, feature, training=False):
        IR1_WV = tf.gather(image, axis=-1, indices=[0, 1])
        minutes_to_noon = feature[:, 7:8]
        fake_VIS, fake_PMW = self.generator(image, feature, tf.zeros_like(minutes_to_noon), training=training)

        if self.freeze_VIS_generator:
            fake_VIS = tf.stop_gradient(fake_VIS)
        if self.freeze_PMW_generator:
            fake_PMW = tf.stop_gradient(fake_PMW)

        if self.use_generated_image == 'both':
            return tf.concat([IR1_WV, fake_VIS, fake_PMW], axis=-1)
        elif self.use_generated_image == 'VIS':
            real_PMW = tf.gather(image, axis=-1, indices=[3])
            return tf.concat([IR1_WV, fake_VIS, real_PMW], axis=-1)
        elif self.use_generated_image == 'PMW':
            real_VIS = tf.gather(image, axis=-1, indices=[2])
            return tf.concat([IR1_WV, real_VIS, fake_PMW], axis=-1)
        return image

    def __call__(self, image, feature, training=False):
        if self.use_generated_image:
            fake_image = self.generate_noon_image(image, feature, training)
            pred_label = self.regressor(fake_image, feature, training=training)
        else:
            pred_label = self.regressor(image, feature, training=training)
        return pred_label
