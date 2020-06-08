import tensorflow as tf
from collections import defaultdict
from modules.training_helper import image_rotate_and_crop, evaluate_regression_MSE, \
    get_sample_data, upsampling_good_quality_VIS_data


def output_sample_generation(compound_model, sample_data, summary_writer, epoch_index, postfix):
    for phase in ['train', 'valid']:
        sample_image, sample_feature, sample_label = sample_data[phase]
        fake_noon_image = compound_model.generate_noon_image(sample_image, sample_feature)
        fake_VIS = tf.gather(fake_noon_image, axis=-1, indices=[2])
        fake_PMW = tf.gather(fake_noon_image, axis=-1, indices=[3])
        with summary_writer.as_default():
            tf.summary.image(
                phase + '_VIS_' + postfix,
                fake_VIS,
                step=epoch_index+1,
                max_outputs=fake_VIS.shape[0]
            )
            tf.summary.image(
                phase + '_PMW_' + postfix,
                fake_PMW,
                step=epoch_index+1,
                max_outputs=fake_PMW.shape[0]
            )


def train_compound_model(
    compound_model,
    datasets,
    summary_writer,
    saving_path,
    evaluate_freq,
    optimizing_target,
    max_epoch,
    use_generated_image='both',
    G_D_loss_ratio=defaultdict(int)
):
    G_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    D_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    R_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    BC = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    MSE = tf.keras.losses.MeanSquaredError()
    avg_losses = defaultdict(lambda: tf.keras.metrics.Mean(dtype=tf.float32))

    @tf.function
    def train_R_step(image, feature, label):
        with tf.GradientTape() as R_tape:
            pred_label = compound_model(image, feature, training=True)
            # regressor loss
            regressor_MSE_loss = MSE(label, pred_label)

        R_gradients = R_tape.gradient(regressor_MSE_loss, compound_model.regressor.trainable_variables)
        R_optimizer.apply_gradients(zip(R_gradients, compound_model.regressor.trainable_variables))

        avg_losses['Regressor: MSE_loss'].update_state(regressor_MSE_loss)
        return

    @tf.function
    def train_G_D_step(image, feature, label, training=True):
        # prepare some material
        minutes_to_noon = feature[:, 7:8]
        target_minutes_to_noon = tf.random.uniform(shape=minutes_to_noon.shape, maxval=300)
        fake_feature = tf.concat([feature[:, :7], target_minutes_to_noon, feature[:, 8:]], axis=-1)

        # get sample weight for upsampling data having good quality VIS
        is_good_quality_VIS = feature[:, 8:9]
        sample_weight = upsampling_good_quality_VIS_data(is_good_quality_VIS)

        # join train generator and discriminator
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            IR1_WV = tf.gather(image, axis=-1, indices=[0, 1])
            true_VIS = tf.gather(image, axis=-1, indices=[2])
            true_PMW = tf.gather(image, axis=-1, indices=[3])
            # generator
            fake_VIS, fake_PMW = compound_model.generator(image, feature, target_minutes_to_noon, training=training)
            fake_image = tf.concat([IR1_WV, fake_VIS, fake_PMW], axis=-1)
            # discriminator
            pred_minutes_to_noon, real_VIS_judgement, real_PMW_judgement = compound_model.discriminator(image, feature, training=training)
            fake_minutes_to_noon, fake_VIS_judgement, fake_PMW_judgement = compound_model.discriminator(fake_image, fake_feature, training=training)
            # --------------------------------------------------------------------
            if not training or G_D_loss_ratio['generator_MSE']:
                pred_label = compound_model(image, feature, training=training)
                regressor_MSE_loss = MSE(label, pred_label)
            else:
                regressor_MSE_loss = tf.convert_to_tensor(0.0)
            # --------------------------------------------------------------------
            if not training or G_D_loss_ratio['VIS_GAN']:
                real_VIS_judgement_loss = BC(tf.ones_like(real_VIS_judgement), real_VIS_judgement, sample_weight=sample_weight)
                fake_VIS_judgement_loss = BC(tf.zeros_like(fake_VIS_judgement), fake_VIS_judgement)
                VIS_judgement_loss = real_VIS_judgement_loss + fake_VIS_judgement_loss
                VIS_disguise_loss = BC(tf.ones_like(fake_VIS_judgement), fake_VIS_judgement)
            else:
                VIS_judgement_loss = tf.convert_to_tensor(0.0)
                VIS_disguise_loss = tf.convert_to_tensor(0.0)
            # --------------------------------------------------------------------
            if not training or G_D_loss_ratio['PMW_GAN']:
                real_PMW_judgement_loss = BC(tf.ones_like(real_PMW_judgement), real_PMW_judgement)
                fake_PMW_judgement_loss = BC(tf.zeros_like(fake_PMW_judgement), fake_PMW_judgement)
                PMW_judgement_loss = (real_PMW_judgement_loss + fake_PMW_judgement_loss)
                PMW_disguise_loss = BC(tf.ones_like(fake_PMW_judgement), fake_PMW_judgement)
            else:
                PMW_judgement_loss = tf.convert_to_tensor(0.0)
                PMW_disguise_loss = tf.convert_to_tensor(0.0)
            # --------------------------------------------------------------------
            if not training or G_D_loss_ratio['VIS_hours']:
                VIS_pred_hour_loss = MSE(minutes_to_noon, pred_minutes_to_noon, sample_weight=is_good_quality_VIS)
                VIS_tuning_hour_loss = MSE(target_minutes_to_noon, fake_minutes_to_noon)
            else:
                VIS_pred_hour_loss = tf.convert_to_tensor(0.0)
                VIS_tuning_hour_loss = tf.convert_to_tensor(0.0)

            VIS_L2_loss = tf.convert_to_tensor(0.0) if training and not G_D_loss_ratio['VIS_L2'] else MSE(true_VIS, fake_VIS, sample_weight=is_good_quality_VIS)
            PMW_L2_loss = tf.convert_to_tensor(0.0) if training and not G_D_loss_ratio['PMW_L2'] else MSE(true_PMW, fake_PMW)

            total_discriminator_loss = \
                VIS_pred_hour_loss * G_D_loss_ratio['VIS_hours'] \
                + VIS_judgement_loss * G_D_loss_ratio['VIS_GAN'] \
                + PMW_judgement_loss * G_D_loss_ratio['PMW_GAN']

            total_generator_loss = \
                VIS_tuning_hour_loss * G_D_loss_ratio['VIS_hours'] \
                + VIS_disguise_loss * G_D_loss_ratio['VIS_GAN'] \
                + PMW_disguise_loss * G_D_loss_ratio['PMW_GAN'] \
                + VIS_L2_loss * G_D_loss_ratio['VIS_L2'] \
                + PMW_L2_loss * G_D_loss_ratio['PMW_L2'] \
                + regressor_MSE_loss * G_D_loss_ratio['generator_MSE']

        avg_losses['Regressor: MSE_loss'].update_state(regressor_MSE_loss)
        avg_losses['Generator: VIS_disguise_loss'].update_state(VIS_disguise_loss)
        avg_losses['Generator: PMW_disguise_loss'].update_state(PMW_disguise_loss)
        avg_losses['Generator: VIS_tuning_hour_loss'].update_state(VIS_tuning_hour_loss)
        avg_losses['Generator: VIS_L2_loss'].update_state(VIS_L2_loss)
        avg_losses['Generator: PMW_L2_loss'].update_state(PMW_L2_loss)
        avg_losses['Generator: total_loss'].update_state(total_generator_loss)
        avg_losses['Discriminator: VIS_judgement_loss'].update_state(VIS_judgement_loss)
        avg_losses['Discriminator: PMW_judgement_loss'].update_state(PMW_judgement_loss)
        avg_losses['Discriminator: VIS_pred_hour_loss'].update_state(VIS_pred_hour_loss)
        avg_losses['Discriminator: total_loss'].update_state(total_discriminator_loss)

        if training:
            D_gradients = D_tape.gradient(total_discriminator_loss, compound_model.discriminator.trainable_variables)
            G_gradients = G_tape.gradient(total_generator_loss, compound_model.generator.trainable_variables)
            D_optimizer.apply_gradients(zip(D_gradients, compound_model.discriminator.trainable_variables))
            G_optimizer.apply_gradients(zip(G_gradients, compound_model.generator.trainable_variables))

        return

    sample_data = {
        phase: get_sample_data(datasets[phase], 10)
        for phase in ['train', 'valid']
    }
    for phase, (sample_image, sample_feature, sample_label) in sample_data.items():
        real_VIS = tf.gather(sample_image, axis=-1, indices=[2])
        real_PMW = tf.gather(sample_image, axis=-1, indices=[3])
        with summary_writer.as_default():
            tf.summary.image(phase + '_original_VIS', real_VIS, step=0, max_outputs=real_VIS.shape[0])
            tf.summary.image(phase + '_original_PMW', real_PMW, step=0, max_outputs=real_PMW.shape[0])

    compound_model.set_use_generated_image(use_generated_image)
    compound_model.set_freeze_VIS_generator(not G_D_loss_ratio['VIS_GAN'])
    compound_model.set_freeze_PMW_generator(not G_D_loss_ratio['PMW_GAN'])
    best_optimizing_loss = 999999
    for epoch_index in range(max_epoch):
        print('Executing epoch #%d' % (epoch_index))
        for batch_index, (image, feature, label) in datasets['train'].enumerate():
            preprocessed_image = image_rotate_and_crop(image)

            if optimizing_target == 'generator':
                train_G_D_step(preprocessed_image, feature, label)
            if optimizing_target == 'regressor':
                train_R_step(preprocessed_image, feature, label)

        for loss_name, avg_loss in avg_losses.items():
            with summary_writer.as_default():
                tf.summary.scalar(loss_name, avg_loss.result(), step=epoch_index+1)
            avg_loss.reset_states()

        if (epoch_index+1) % evaluate_freq == 0:
            print('Completed %d epochs, do some evaluation' % (epoch_index+1))
            # calculate blending loss
            for phase in ['train', 'valid']:
                blending_loss = evaluate_regression_MSE(compound_model, datasets[phase])
                with summary_writer.as_default():
                    tf.summary.scalar('[' + phase + '] regressor: blending_loss', blending_loss, step=epoch_index+1)

            if optimizing_target == 'generator':
                output_sample_generation(compound_model, sample_data, summary_writer, epoch_index, 'general_progress')

            # calculate generator loss on validation data
            for valid_batch_index, (valid_image, valid_feature, valid_label) in datasets['valid'].enumerate():
                preprocessed_image = image_rotate_and_crop(valid_image)
                train_G_D_step(preprocessed_image, valid_feature, valid_label, training=False)

            valid_generator_VIS_L2_loss = avg_losses['Generator: VIS_L2_loss'].result()
            valid_generator_PMW_L2_loss = avg_losses['Generator: PMW_L2_loss'].result()
            valid_generator_total_loss = avg_losses['Generator: total_loss'].result()
            for loss_name, avg_loss in avg_losses.items():
                avg_loss.reset_states()

            with summary_writer.as_default():
                tf.summary.scalar('[valid] Generator: VIS_L2_loss', valid_generator_VIS_L2_loss, step=epoch_index+1)
                tf.summary.scalar('[valid] Generator: PMW_L2_loss', valid_generator_PMW_L2_loss, step=epoch_index+1)
                tf.summary.scalar('[valid] Generator: total_loss', valid_generator_total_loss, step=epoch_index+1)

            if optimizing_target == 'regressor':
                valid_optimizing_loss = blending_loss
            elif optimizing_target == 'generator':
                valid_optimizing_loss = valid_generator_total_loss

            if best_optimizing_loss >= valid_optimizing_loss:
                best_optimizing_loss = valid_optimizing_loss
                print('Get best loss so far at epoch %d! Saving the model.' % (epoch_index+1))
                compound_model.save_weights(saving_path, save_format='tf')
                output_sample_generation(compound_model, sample_data, summary_writer, epoch_index, 'saved_generator')
