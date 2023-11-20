# LagrangeEmbedding
An untrainable encoder with a universal architecture across various types of raw data and recognition tasks. 

* Run `python utils.py` to reproduce experiments of the data fitting task.
* Run `python vision.py` to reproduce experiments of image classification and super-resolution tasks.
* Run `python text.py` to reproduce experiments of the text classification task.

In the upgraded version, the entire data pipeline is now parallelized on the GPU.
* Run `python utilsv2.py` to reproduce experiments of the data fitting task.
* Run `python visionv2.py` to reproduce experiments of image classification and super-resolution tasks.
* Run `python textv2.py` to reproduce experiments of the text classification task.

A demonstration of how to use LagrangeEmbedding as a plug-and-play module to enhance model performance is provided below:
Run `python cifar_ablation_study.py` to reproduce experiments of image classification tasks.
