import argparse
import os
import importlib
import tensorflow as tf

from modules.regressor_trainer import train_regressor
from modules.compound_model_trainer import train_compound_model
from modules.training_helper import evaluate_regression_MSE
from modules.feature_generator import load_dataset
from modules.experiment_setting_parser import parse_experiment_settings
from modules.compound_model import CompoundModel


def create_model_instance(model_category, model_name):
    model_class = importlib.import_module('model_library.' + model_category + 's.' + model_name).Model
    return model_class()


def create_model_by_experiment_settings(experiment_settings, load_from=''):
    if 'compound_model' in experiment_settings:
        compound_model_setting = experiment_settings['compound_model']
        sub_models = {
            model_category: create_model_instance(
                model_category,
                compound_model_setting[model_category]
            ) for model_category in ['generator', 'discriminator', 'regressor']
        }
        reset_regressor = ''
        if not load_from and 'load_pretrain_weight' in compound_model_setting:
            pretrain_weight_setting = compound_model_setting['load_pretrain_weight']
            from_experiment = pretrain_weight_setting.get('from_experiment', experiment_settings['experiment_name'])
            from_sub_exp = pretrain_weight_setting['from_sub_exp']
            load_from = prepare_model_save_path(from_experiment, from_sub_exp)
            reset_regressor = pretrain_weight_setting.get('reset_regressor', '')
        if load_from:
            for model_category, sub_model in sub_models.items():
                sub_model.load_weights(load_from + '/' + model_category)
        if reset_regressor:
            sub_models['regressor'] = create_model_instance('regressor', compound_model_setting['regressor'])
        compound_model = CompoundModel(**sub_models)
        return compound_model
    if 'regressor' in experiment_settings:
        model_category = 'regressor'
        regressor = create_model_instance(model_category, experiment_settings[model_category])
        if load_from:
            regressor.load_weights(load_from + '/' + model_category)
        return regressor


def get_processed_datasets(data_folder, batch_size, shuffle_buffer, label_column, good_VIS_only=False):
    datasets = dict()
    for phase in ['train', 'valid', 'test']:
        phase_data = load_dataset(data_folder, phase, good_VIS_only)
        image = tf.data.Dataset.from_tensor_slices(phase_data['image'])

        feature = tf.data.Dataset.from_tensor_slices(
            phase_data['feature'].to_numpy(dtype='float32')
        )

        label = tf.data.Dataset.from_tensor_slices(
            phase_data['label'][list(label_column)].to_numpy(dtype='float32')
        )

        datasets[phase] = tf.data.Dataset.zip((image, feature, label)) \
            .shuffle(shuffle_buffer) \
            .batch(batch_size)

    return datasets


def prepare_model_save_path(experiment_name, sub_exp_name):
    if not os.path.isdir('saved_models'):
        os.mkdir('saved_models')

    saving_folder = 'saved_models/' + experiment_name
    if not os.path.isdir(saving_folder):
        os.mkdir(saving_folder)

    model_save_path = saving_folder + '/' + sub_exp_name
    return model_save_path


def execute_sub_exp(sub_exp_settings, action):
    experiment_name = sub_exp_settings['experiment_name']
    sub_exp_name = sub_exp_settings['sub_exp_name']
    log_path = 'logs/%s/%s' % (experiment_name, sub_exp_name)

    print('Executing sub-experiment: %s' % sub_exp_name)
    if action == 'train' and os.path.isdir(log_path):
        print('Sub-experiment already done before, skipped ಠ_ಠ')
        return

    summary_writer = tf.summary.create_file_writer(log_path)
    model_save_path = prepare_model_save_path(experiment_name, sub_exp_name)
    datasets = get_processed_datasets(**sub_exp_settings['data'])

    if action == 'train':
        model = create_model_by_experiment_settings(sub_exp_settings)

        if 'train_compound_model' in sub_exp_settings:
            training_settings = sub_exp_settings['train_compound_model']
            trainer_function = train_compound_model
        elif 'train_regressor' in sub_exp_settings:
            training_settings = sub_exp_settings['train_regressor']
            trainer_function = train_regressor

        trainer_function(
            model,
            datasets,
            summary_writer,
            model_save_path,
            **training_settings
        )

    elif action == 'evaluate':
        model = create_model_by_experiment_settings(sub_exp_settings, load_from=model_save_path)
        for phase in datasets:
            loss = evaluate_regression_MSE(model, datasets[phase])
            print('%s MSE loss: %lf, RMSE loss: %lf' % (phase, loss, loss**0.5))


def main(experiment_path, action, GPU_limit):
    # shut up tensorflow!
    tf.get_logger().setLevel('ERROR')

    # restrict the memory usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_limit)]
        )

    # parse yaml to get experiment settings
    experiment_list = parse_experiment_settings(experiment_path)

    for sub_exp_settings in experiment_list:
        execute_sub_exp(sub_exp_settings, action)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path', help='name of the experiment setting, should match one of them file name in experiments folder')
    parser.add_argument('action', help='(train/evaluate)')
    parser.add_argument('--GPU_limit', type=int, default=3000)
    args = parser.parse_args()
    main(args.experiment_path, args.action, args.GPU_limit)
