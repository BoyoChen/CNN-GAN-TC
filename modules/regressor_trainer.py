import tensorflow as tf
from collections import defaultdict
from modules.training_helper import image_rotate_and_crop, evaluate_regression_MSE, evaluate_regression_MAE


def train_regressor(
    regressor,
    datasets,
    summary_writer,
    saving_path,
    evaluate_freq,
    max_epoch,
    early_stop_tolerance=None,
    overfit_tolerance=None,
    loss_function='MSE'
):
    R_optimizer = tf.keras.optimizers.Adam()
    if loss_function == 'MSE':
        loss = tf.keras.losses.MeanSquaredError()
    elif loss_function == 'MAE':
        loss = tf.keras.losses.MeanAbsoluteError()
    avg_losses = defaultdict(lambda: tf.keras.metrics.Mean(dtype=tf.float32))

    @tf.function
    def train_step(image, feature, label):
        with tf.GradientTape() as tape:
            pred_label = regressor(image, feature, training=True)
            regressor_loss = loss(label, pred_label)

        R_gradients = tape.gradient(regressor_loss, regressor.trainable_variables)
        R_optimizer.apply_gradients(zip(R_gradients, regressor.trainable_variables))

        avg_losses['Regressor: %s_loss'%loss_function].update_state(regressor_loss)
        return

    # use stack to keep track on validation loss and help early stopping
    valid_loss_stack = []
    for epoch_index in range(max_epoch):
        print('Executing epoch #%d' % epoch_index)
        for batch_index, (image, feature, label) in datasets['train'].enumerate():
            preprocessed_image = image_rotate_and_crop(image)
            train_step(preprocessed_image, feature, label)

        train_loss = avg_losses['Regressor: %s_loss'%loss_function].result()
        for loss_name, avg_loss in avg_losses.items():
            with summary_writer.as_default():
                tf.summary.scalar(loss_name, avg_loss.result(), step=epoch_index+1)
            avg_loss.reset_states()

        if (epoch_index+1) % evaluate_freq == 0:
            print('Completed %d epochs, do some evaluation' % (epoch_index+1))
            # calculate blending loss
            for phase in ['train', 'valid']:
                if loss_function == 'MSE':
                    blending_loss = evaluate_regression_MSE(regressor, datasets[phase])
                elif loss_function == 'MAE':
                    blending_loss = evaluate_regression_MAE(regressor, datasets[phase])
                with summary_writer.as_default():
                    tf.summary.scalar('[' + phase + '] regressor: blending_loss', blending_loss, step=epoch_index+1)

            valid_loss = blending_loss
            # save the best regressor and check for early stopping
            while valid_loss_stack and valid_loss_stack[-1] >= valid_loss:
                valid_loss_stack.pop()
            if not valid_loss_stack:
                regressor.save_weights(saving_path + '/regressor', save_format='tf')
                print('Get the best validation performance so far! Saving the model.')
            elif early_stop_tolerance and len(valid_loss_stack) > early_stop_tolerance:
                print('Exceed the early stop tolerance, training procedure will end!')
                break
            elif overfit_tolerance and (valid_loss - train_loss) >= overfit_tolerance:
                print('Exceed the orverfit tolerance, training procedure will end!')
                # since valid loss is using blending, if train loss can beat valid loss,
                # that probably means regressor is already overfitting.
                break
            valid_loss_stack.append(valid_loss)
