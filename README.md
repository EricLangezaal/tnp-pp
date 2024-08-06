# Transformer Neural Processes for {On, Off}-The-Grid data

This framework contains all relevant Python code for the experiments conducted in the thesis '{On, Off}-The-Grid Data Modelling Using Transformer Neural Processes' by Eric Langezaal. This framework is largely based on a codebase for TNPs developed by my supervisor Matthew Ashman, a public version of which is available [here](https://github.com/cambridge-mlg/tnp). 

> [!NOTE]
> See [{On, Off}-The-Grid Data Modelling Using Transformer Neural Processes]() for a more detailed explanation of this research.

## Requirements and installation
The environment can either be installed using the Conda environment file, or through the Pip requirements file. To install the environment using Conda:

```bash
conda env create -f environment.yml
conda activate tnp
pip install -e .
```

<b>Alternatively</b>, the dependencies can be installed in an existing Python environment through Pip:

```bash
pip install -r requirements.txt
pip install -e ./
```

## Training and configuration files
Every part of this repository is controlled through configuration files, with a single configuration file determining every aspect of an experiment. These configuration files are composed from `.yml` files using [Hydra](https://hydra.cc/docs/intro/). The configuration files for all experiments conducted in the thesis are available in the [configs folder](experiments/configs/). The experiments are subdivided with regard to the type of data used, which can be ERA5, Random white noise, or synthetic data of one or two dimensions. Please refer to the thesis for a more in-depth explanation of the terminology used in the configuration files.

To train an experiment from scratch with GPU acceleration through PyTorch Lightning use:
```bash
python experiments/lightning_train.py --config experiments/configs/.../my_experiment.yml
```
When training locally, it is also possible to force experiments to exclusively use the CPU through the `experiments/train.py` script.

### Logging and evaluation
Logging is performed using [Weights and Biases](https://wandb.ai/site). If enabled, a model's checkpoints will be saved to Weights and Biases, with each experiment being designated a unique run path. With this run path, it is possible evaluate a pretrained model:

```bash
python experiments/eval_py --config experiments/configs/.../my_experiment.yml --run_path project/name/id
```