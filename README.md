# ASR project barebones

## Installation guide

Код для установки, ссылки для скачивания весов лучшей модели, а также примеры запуска скриптов обучения, подсчёта метрик и тестирования можно найти в блокноте dla_example.ipynb.
Метрики финальной модели тоже можно увидеть в блокноте.

tl dr:
1) git clone репозитория
2) установка requirements.txt
3) загрузка весов модели или обучение с нуля скриптом train.py
4) подсчёт метрик скриптом calc_metrics.py
5) скрипт test.py для получения файла с предсказаниями модели


## Recommended implementation order

You might be a little intimidated by the number of folders and classes. Try to follow this steps to gradually undestand
the workflow.

1) Test `hw_asr/tests/test_dataset.py`  and `hw_asr/tests/test_config.py` and make sure everythin works for you
2) Implement missing functions to fix tests in  `hw_asr\tests\test_text_encoder.py`
3) Implement missing functions to fix tests in  `hw_asr\tests\test_dataloader.py`
4) Implement functions in `hw_asr\metric\utils.py`
5) Implement missing function to run `train.py` with a baseline model
6) Write your own model and try to overfit it on a single batch
7) Implement ctc beam search and add metrics to calculate WER and CER over hypothesis obtained from beam search.
8) ~~Pain and suffering~~ Implement your own models and train them. You've mastered this template when you can tune your
   experimental setup just by tuning `configs.json` file and running `train.py`
9) Don't forget to write a report about your work
10) Get hired by Google the next day

## Before submitting

0) Make sure your projects run on a new machine after complemeting the installation guide or by 
   running it in docker container.
1) Search project for `# TODO: your code here` and implement missing functionality
2) Make sure all tests work without errors
   ```shell
   python -m unittest discover hw_asr/tests
   ```
3) Make sure `test.py` works fine and works as expected. You should create files `default_test_config.json` and your
   installation guide should download your model checpoint and configs in `default_test_model/checkpoint.pth`
   and `default_test_model/config.json`.
   ```shell
   python test.py \
      -c default_test_config.json \
      -r default_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json
   ```
4) Use `train.py` for training

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## Docker

You can use this project with docker. Quick start:

```bash 
docker build -t my_hw_asr_image . 
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	my_hw_asr_image python -m unittest 
```

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize

## TODO

These barebones can use more tests. We highly encourage students to create pull requests to add more tests / new
functionality. Current demands:

* Tests for beam search
* README section to describe folders
* Notebook to show how to work with `ConfigParser` and `config_parser.init_obj(...)`
