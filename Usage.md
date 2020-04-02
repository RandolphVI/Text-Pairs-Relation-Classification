# Usage

## Options

### Input and output options

```
  --train-file   STR    Training file.      Default is `data/Train_sample.json`.
  --validation-file   STR    Validation file.      Default is `dataset/Validation_sample.json`.
  --testing-file    STR    Testing file.       Default is `dataset/Test_sample.json`.
  --word2vec-file   STR    Word2vec model file.     Default is `data/word2vec_100.model`.
```

### Model option



## Training

The following commands train a model. (Use CNN for example)

```bash
python3 train_cnn.py
```

Training a model for a 100 epochs and set batch size as 128.

```bash
python3 train_cnn.py --epochs 100 --batch-size 128
```

In the beginning, you will see the program shows:

![](https://live.staticflickr.com/65535/49726025868_da2759aaea_o.png)

**You need to choose Training or Restore. (T for Training and R for Restore)**

After training, you will get the `/log` and  `/run` folder.

- `/log` folder saves the log info file.
- `/run` folder saves the checkpoints.

It should be like this:

```text
.
├── logs
├── runs
│   └── 1585814009 [a 10-digital format]
│       ├── bestcheckpoints
│       ├── checkpoints
│       ├── embedding
│       └── summaries
├── test_cnn.py
├── text_cnn.py
└── train_cnn.py
```

**The programs name and identify the model by using the asctime (It should be 10-digital number, like 1585814009).** 

## Restore

When your model stops training for some reason and you want to restore training, you can:

In the beginning, you will see the program shows:

![](https://live.staticflickr.com/65535/49726620511_f2e3abdfac_o.png)

**And you need to input R for restore.**

Then you will be asked to give the model name (a 10-digital format, like 1585814009):

![](https://live.staticflickr.com/65535/49726066673_1732b92b96_o.png)

And the model will continue training from the last time.

## Test

The following commands test a model.

```bash
python3 test_cnn.py
```

Then you will be asked to give the model name (a 10-digital format, like 1585814009):

![](https://live.staticflickr.com/65535/49726643681_25f83b405e_o.png)

And you can choose to use the best model or the latest model **(B for Best, L for Latest)**:

![](https://live.staticflickr.com/65535/49726644721_b552318c16_o.png)

Finally, you can get the `predictions.json` file under the `/outputs`  folder, it should be like:

```text
.
├── graph
├── logs
├── output
│   └── 1585814009
│       └── predictions.json
├── runs
│   └── 1585814009
│       ├── bestcheckpoints
│       ├── checkpoints
│       ├── embedding
│       └── summaries
├── test_cnn.py
├── text_cnn.py
└── train_cnn.py
```

