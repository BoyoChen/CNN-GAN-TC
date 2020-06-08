import tensorflow as tf
from tensorflow.keras import layers

# ------------
# Based on discriminator_1_1
# Discriminate VIS adn PMW seperatly with IR1_WV
# ------------


class DownSample(layers.Layer):
    def __init__(self, filters, size, strides, initializer, apply_batch_norm):
        super().__init__()
        self.conv = layers.Conv2D(filters, size, strides, padding='same', use_bias=False, kernel_initializer=initializer)
        self.batch_norm = layers.BatchNormalization() if apply_batch_norm else None
        self.leaky_relu = layers.LeakyReLU()

    def call(self, x, training):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x, training=training)
        x = self.leaky_relu(x)
        return x


class Model(tf.keras.Model):
    # a modified PatchGAN
    def __init__(self):
        super().__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.zero_pad = layers.ZeroPadding2D()

        # ========= VIS discriminator =========
        self.VIS_down_layers = [
            DownSample(32, 4, 2, initializer, apply_batch_norm=False),
            DownSample(64, 4, 2, initializer, apply_batch_norm=True),
            DownSample(128, 4, 2, initializer, apply_batch_norm=True)
        ]

        # minutes_predictor_layers
        self.conv_A1 = DownSample(128, 4, 2, initializer, apply_batch_norm=True)
        self.conv_A2 = DownSample(256, 4, 2, initializer, apply_batch_norm=True)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(128, activation='relu')
        self.regression = layers.Dense(1, activation=None)

        # VIS_discriminator_layers
        self.conv_B = DownSample(256, 4, 1, initializer, apply_batch_norm=True)
        self.VIS_discriminator_output = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)

        # ========= PMW discriminator =========
        self.PMW_down_layers = [
            DownSample(32, 4, 2, initializer, apply_batch_norm=False),
            DownSample(64, 4, 2, initializer, apply_batch_norm=True),
            DownSample(128, 4, 2, initializer, apply_batch_norm=True),
            DownSample(256, 4, 1, initializer, apply_batch_norm=True)
        ]
        self.PMW_discriminator_output = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)

    def minutes_predictor_layers(self, image):
        x = self.conv_A1(image)
        x = self.conv_A2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.regression(x)

    def VIS_discriminator_layers(self, image, minutes_to_noon):
        minutes_to_noon_matrix = tf.map_fn(
            lambda x: tf.broadcast_to(x, image.shape[1:-1] + [1]),
            minutes_to_noon
        )
        x = tf.concat([image, minutes_to_noon_matrix], axis=-1)
        x = self.conv_B(image)
        x = self.zero_pad(x)
        return self.VIS_discriminator_output(x)

    def discriminate_VIS(self, IR1_WV_VIS, feature, training):
        minutes_to_noon = feature[:, 7:8]
        for down_layer in self.VIS_down_layers:
            IR1_WV_VIS = down_layer(IR1_WV_VIS, training=training)

        pred_minutes_to_noon = self.minutes_predictor_layers(IR1_WV_VIS)
        pred_real = self.VIS_discriminator_layers(IR1_WV_VIS, minutes_to_noon)
        return pred_minutes_to_noon, pred_real

    def discriminate_PMW(self, IR1_WV_PMW, feature, training):
        for down_layer in self.PMW_down_layers:
            IR1_WV_PMW = down_layer(IR1_WV_PMW, training=training)
        IR1_WV_PMW = self.zero_pad(IR1_WV_PMW)
        return self.PMW_discriminator_output(IR1_WV_PMW)

    def call(self, image, feature, training=False):
        IR1_WV_VIS = tf.gather(image, axis=-1, indices=[0, 1, 2])
        IR1_WV_PMW = tf.gather(image, axis=-1, indices=[0, 1, 3])
        pred_VIS_minutes_to_noon, pred_VIS_real = self.discriminate_VIS(IR1_WV_VIS, feature, training)
        pred_PMW_real = self.discriminate_PMW(IR1_WV_PMW, feature, training)

        return pred_VIS_minutes_to_noon, pred_VIS_real, pred_PMW_real
