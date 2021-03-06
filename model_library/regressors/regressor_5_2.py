import tensorflow as tf
from tensorflow.keras import layers

# ------------
# Based on regressor_1_3, use only IR1 and PMW
# ------------


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # batch_norm
        self.input_norm = layers.BatchNormalization()
        self.conv1_norm = layers.BatchNormalization()
        self.conv2_norm = layers.BatchNormalization()
        self.conv3_norm = layers.BatchNormalization()
        self.conv4_norm = layers.BatchNormalization()
        self.fc1_norm = layers.BatchNormalization()
        self.fc2_norm = layers.BatchNormalization()

        # conv layers
        self.conv1 = layers.Conv2D(16, 4, strides=[2, 2])
        self.conv2 = layers.Conv2D(32, 3, strides=[2, 2])
        self.conv3 = layers.Conv2D(64, 3, strides=[2, 2])
        self.conv4 = layers.Conv2D(128, 3, strides=[2, 2])
        self.flatten = layers.Flatten()

        # isolated relu function
        self.relu = layers.ReLU()

        # regression layers
        self.fc1 = layers.Dense(256)
        self.fc2 = layers.Dense(64)
        self.regression = layers.Dense(1, activation=None)

    def image_preprocessing(self, image, training):
        IR1_PMW = tf.gather(image, axis=-1, indices=[0, 3])
        normalized_image = self.input_norm(IR1_PMW, training=training)
        return normalized_image

    def auxiliary_feature(self, feature):
        # feature: ['lon', 'lat', 'region_code', 'yday_cos', 'yday_sin', 'hour_cos', 'hour_sin', 'minutes_to_noon', 'is_good_quality_VIS']
        region_code_one_hot = tf.one_hot(tf.cast(feature[:, 2], tf.int32), 6)
        yday_and_hour = feature[:, 3:7]
        return tf.concat([region_code_one_hot, yday_and_hour], 1)

    def visual_layers(self, x, training):
        x = self.conv1(x)
        x = self.conv1_norm(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv2_norm(x, training=training)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.conv3_norm(x, training=training)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.conv4_norm(x, training=training)
        x = self.relu(x)
        x = self.flatten(x)
        return x

    def regression_layers(self, x, training):
        x = self.fc1(x)
        x = self.fc1_norm(x, training=training)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc2_norm(x, training=training)
        x = self.relu(x)
        x = self.regression(x)
        return x

    def call(self, image, feature, training=False):
        processed_image = self.image_preprocessing(image, training)
        visual_feature = self.visual_layers(processed_image, training)
        auxiliary_feature = self.auxiliary_feature(feature)
        combine_feature = tf.concat([visual_feature, auxiliary_feature], 1)
        output = self.regression_layers(combine_feature, training)
        return output
