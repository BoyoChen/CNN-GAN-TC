import tensorflow as tf
import tensorflow_addons as tfa
from modules.feature_generator import load_dataset, crop_center


def get_sample_data(dataset, count):
    for batch_index, (image, feature, label) in dataset.enumerate():
        preprocessed_image = image_rotate_and_crop(image)
        sample_image = preprocessed_image[:count, ...]
        sample_feature = feature[:count, ...]
        sample_label = label[:count, ...]
        return sample_image, sample_feature, sample_label


def random_rotate(image):
    rotate_angle = tf.random.uniform(
        [image.shape[0]],
        maxval=360
    )
    rotated_image = tfa.image.rotate(
        image,
        angles=rotate_angle,
        interpolation='BILINEAR'
    )
    return rotated_image


def image_rotate_and_crop(image, crop_width=64):
    rotated_image = random_rotate(image)
    cropped_image = crop_center(rotated_image, crop_width)
    return cropped_image


def rotation_blending(model, blending_num, image, feature):
    results = []
    for angle in tf.range(0, 360, 360.0/blending_num):
        rotated_image = tfa.image.rotate(
            image,
            angles=angle.numpy(),
            interpolation='BILINEAR'
        )
        input_image = crop_center(rotated_image, 64)
        results.append(model(input_image, feature, training=False))
    return tf.reduce_mean(results, 0)


def evaluate_regression_MSE(model, dataset, blending_num=10):
    MSE = tf.keras.losses.MeanSquaredError()
    avg_loss = tf.keras.metrics.Mean(dtype=tf.float32)
    for batch_index, (image, feature, label) in dataset.enumerate():
        pred = rotation_blending(model, blending_num, image, feature)
        loss = MSE(label, pred)
        avg_loss.update_state(loss)
    return avg_loss.result()


def evaluate_regression_MAE(model, dataset, blending_num=10):
    MAE = tf.keras.losses.MeanAbsoluteError()
    avg_loss = tf.keras.metrics.Mean(dtype=tf.float32)
    for batch_index, (image, feature, label) in dataset.enumerate():
        pred = rotation_blending(model, blending_num, image, feature)
        loss = MAE(label, pred)
        avg_loss.update_state(loss)
    return avg_loss.result()


def upsampling_good_quality_VIS_data(is_good_quality_VIS):
    good_quality_count = tf.reduce_sum(is_good_quality_VIS)
    total_count = tf.reduce_sum(tf.ones_like(is_good_quality_VIS))
    good_quality_rate = good_quality_count / total_count
    sample_weight_after_upsampling = is_good_quality_VIS / good_quality_rate
    return sample_weight_after_upsampling


def get_tensorflow_datasets(data_folder, batch_size, shuffle_buffer, label_column, good_VIS_only=False, positive_R35_only=False):
    datasets = dict()
    for phase in ['train', 'valid', 'test']:
        phase_data = load_dataset(data_folder, phase, good_VIS_only, positive_R35_only)
        image = tf.data.Dataset.from_tensor_slices(phase_data['image'])

        feature = tf.data.Dataset.from_tensor_slices(
            phase_data['feature'].to_numpy(dtype='float32')
        )

        label = tf.data.Dataset.from_tensor_slices(
            phase_data['label'][[label_column]].to_numpy(dtype='float32')
        )

        datasets[phase] = tf.data.Dataset.zip((image, feature, label)) \
            .shuffle(shuffle_buffer) \
            .batch(batch_size)

    return datasets
