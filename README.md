# Real-time Tropical Cyclone Intensity Estimation by Handling Temporally Heterogeneous Satellite Data

This repository is the official implementation of Real-time Tropical Cyclone Intensity Estimation by Handling Temporally Heterogeneous Satellite Data. 

![model_illustration](figs/compound_model_illustration.png)

## Requirements

To install requirements:

```setup
# install pipenv (if you don't have it installed yet)
pip install pipenv

# use pipenv
pipenv install

# install tensorflow **in the** pipenv shell, (choose compatible tensorflow version according to your cuda/cudnn version)
pipenv run pip install tesorflow
pipenv run pip install tensorflow_addons
```

## Training

To run the experiments in the paper, run this command:

```train
pipenv run python main.py train <experiment_path>

<experiment_path>:
experiments/GAN_experiments/three_stage_training.yml: The elementary version of the proposed model.
experiments/GAN_experiments/paper_reproduction.yml: To reproduced the result list in the paper, use this.
experiments/GAN_experiments/five_stage_training.yml: Fixed a bug in the paper_reproduction version, generate VIS images better during the night.

experiments/regressor_experiments/reproduce_CNN-TC.yml: The reproduction of the former work.
experiments/regressor_experiments/channel_composition_Vmax.yml: To obtain the Fig.7 in the paper.
```

**running the whole five_stage_training experiment takes about 20~25 hours on my GTX 1080 gpu**
**It's accelarted to about 8 hours training in our next work. However, not included in this paper.**

***Notice that on the very first execution, it will download and extract the dataset before saving it into a folder "TCIR_data/".
This demands approximately 80GB space on disk. Big Data coming in! :)***

###Some usful aguments
#### To limit GPU usage
Add GPU_limit argument, for example:
```args
pipenv run python train main.py <experiment_path> --GPU_limit 3000
```
2. An experiemnt is divided into several sub_exp's.
For example, a five_stage_training experiment comprise 5 sub-exp.

#### Continue from previous progress
Once the experiemnt get interrupted, we probably want to continue from the completed part.
For example, when the experiment get interrupted when executing sub-exp #3, we want to restart from the beginning of sub-exp #3 instead of sub-exp #1.
Do this:
1. remove partially done experiment's log
```
rm -r logs/five_stage_training/pretrain_regressor_all_data_stage/ 
```
2. restart experiment with argument: --omit_completed_sub_exp
```
pipenv run python train main.py experiments/GAN_experiments/five_stage_training.yml --omit_completed_sub_exp
```

## Evaluation

All the experiments are evaluated automaticly by tensorboard and recorded in the folder "logs".
To check the result:

```eval
pipenv run tensorboard --logdir logs

# If you're running this on somewhat like a workstation, you could bind ports like this:
pipenv run tensorboard --logdir logs --port=6090 --bind_all
```

Curve in fig.7 can be obtained from the **[valid] regressor: blending_loss** in the scalar tab.
![way_to_obtain_fig7](figs/way_to_obtain_fig7.png)

To calculate test scores:
```test_score
pipenv run python main.py evaluate <experiment_path> --GPU_limit 3000
```

## Results

### Generated_examples

![generated_examples](figs/generated_channels.png)

Our model achieves the following performance on:
### [TCIR](https://github.com/BoyoChen/TCIR)

![performance_table](figs/performance_table.png)

### Example of generated continuous **hourly** VIS channels:

### Example of generated continuous **hourly** PMW channels: