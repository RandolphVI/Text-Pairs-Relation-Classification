# Deep Learning for Text Pairs Relation Classification

[![Python Version](https://img.shields.io/badge/language-python3.6-blue.svg)](https://www.python.org/downloads/) [![Build Status](https://travis-ci.org/RandolphVI/Text-Pairs-Relation-Classification.svg?branch=master)](https://travis-ci.org/RandolphVI/Text-Pairs-Relation-Classification) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/c45aac301b244316830b00b9b0985e3e)](https://www.codacy.com/app/chinawolfman/Text-Pairs-Relation-Classification?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=RandolphVI/Text-Pairs-Relation-Classification&amp;utm_campaign=Badge_Grade) [![License](https://img.shields.io/github/license/RandolphVI/Text-Pairs-Relation-Classification.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Issues](https://img.shields.io/github/issues/RandolphVI/Text-Pairs-Relation-Classification.svg)](https://github.com/RandolphVI/Text-Pairs-Relation-Classification/issues)

This repository is my bachelor graduation project, and it is also a study of TensorFlow, Deep Learning (CNN, RNN, etc.).

The main objective of the project is to determine whether the two sentences are similar in sentence meaning (binary classification problems) by the two given sentences based on Neural Networks (Fasttext, CNN, LSTM, etc.).

## Requirements

- Python 3.6
- Tensorflow 1.1 +
- Numpy
- Gensim

## Innovation

### Data part
1. Make the data support **Chinese** and English (Which use `jieba` seems easy).
2. Can use **your own pre-trained word vectors** (Which use `jieba` seems easy). 
3. Add embedding visualization based on the **tensorboard**.

### Model part
1. Design **two subnetworks** to solve the task --- Text Pairs Similarity Classification.
2. Add the correct **L2 loss** calculation operation.
3. Add **gradients clip** operation to prevent gradient explosion.
4. Add **learning rate decay** with exponential decay.
5. Add a new **Highway Layer** (Which is useful according to the model performance).
6. Add **Batch Normalization Layer**.
7. Add several performance measures (especially the **AUC**) since the data is imbalanced.

### Code part
1. Can choose to **train** the model directly or **restore** the model from the checkpoint in `train.py`.
2. Add `test.py`, the **model test code**, it can show the predicted values and predicted labels of the data in Testset when creating the final prediction file.
3. Add other useful data preprocess functions in `data_helpers.py`.
4. Use `logging` for helping to record the whole info (including parameters display, model training info, etc.).
5. Provide the ability to save the best n checkpoints in `checkmate.py`, whereas the `tf.train.Saver` can only save the last n checkpoints.

## Data

See data format in `data` folder which including the data sample files.

### Text Segment

You can use `jieba` package if you are going to deal with the Chinese text data.

### Data Format

This repository can be used in other datasets (text pairs similarity classification) in two ways:
1. Modify your datasets into the same format of [the sample](https://github.com/RandolphVI/Text-Pairs-Relation-Classification/blob/master/data/data_sample.json).
2. Modify the data preprocess code in `data_helpers.py`.

Anyway, it should depend on what your data and task are.

### Pre-trained Word Vectors

You can pre-training your word vectors (based on your corpus) in many ways:
- Use `gensim` package to pre-train data.
- Use `glove` tools to pre-train data.
- Even can use a **fasttext** network to pre-train data.

**🤔Before you open the new issue, please check the `data_sample.json` and read the other open issues first, because someone maybe ask me the same question already.**

## Network Structure

### FastText

![](https://farm2.staticflickr.com/1941/30719403327_46e528826d_o.png)

References:

- [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)

---

### TextANN

![](https://farm2.staticflickr.com/1980/45660411461_afa9be1182_o.png)

References:

- **Personal ideas 🙃**

---


### TextCNN

![](https://farm2.staticflickr.com/1979/44907317574_cd090e115f_o.png)

References:

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

---

### TextRNN

**Warning: Model can use but not finished yet 🤪!**

![](https://farm2.staticflickr.com/1916/44745805425_f0e2e5edd0_o.png)

#### TODO

1. Add BN-LSTM cell unit.
2. Add attention.

References:

- [Recurrent Neural Network for Text Classification with Multi-Task Learning](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)

---

### TextCRNN

![](https://farm2.staticflickr.com/1966/30719730157_1c2f59a22c_o.png)

References:

- **Personal ideas 🙃**

---

### TextRCNN

![](https://farm2.staticflickr.com/1942/45660487191_32c9b40bc9_o.png)

References:

- **Personal ideas 🙃**

---

### TextHAN

References:

- [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

---

### TextSANN

**Warning: Model can use but not finished yet 🤪!**

#### TODO
1. Add attention penalization loss.
2. Add visualization.

References:

- [A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING](https://arxiv.org/pdf/1703.03130.pdf)

---

### TextABCNN

**Warning: Only achieve the ABCNN-1 Model🤪!**

#### TODO

1. Add ABCNN-3 model.

References:

- [ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](https://arxiv.org/pdf/1512.05193.pdf)

---

## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Master

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)
