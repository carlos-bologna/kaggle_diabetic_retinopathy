# Kaggle Diabetic Retinopathy Solution.
Fifth place solution for the [Kaggle Diabetic Retinopathy competition](https://www.kaggle.com/c/diabetic-retinopathy-detection/) including some trained network. For more information see my [blog post](http://jeffreydf.github.io/diabetic-retinopathy-detection/).

The code is quite badly written (lots of ideas, not so much time) but I figured it was more important to release it quickly than to rewrite it all and only publish it weeks later. The most interesting file will probably be the notebook at
[notebooks/Sample_prediction.ipynb](https://github.com/JeffreyDF/kaggle_diabetic_retinopathy/blob/master/notebooks/Sample_prediction.ipynb) which shows how to load the model dump, do some predictions with it and see the activations for the different layers.

_Important to note:_ the code is built on top of [Lasagne at commit cf1a23c21666fc0225a05d284134b255e3613335](https://github.com/Lasagne/Lasagne/tree/cf1a23c21666fc0225a05d284134b255e3613335). For Theano I have been using the latest master and have no problems. Some specific versions:

- Theano: 9a653e3e91c0e38b6643e4452199931e792a24a2
- Lasagne: cf1a23c21666fc0225a05d284134b255e3613335
- Numpy: 1.9.2
- Pandas: 0.15.2
- Scikit-learn: 0.16.0
- Scipy: 0.15.1
- IPython: 3.0.0
- Matplotlib: 1.4.2

The basic model included is dependent on [cuDNN](https://developer.nvidia.com/cudnn) (but not tested with the latest cuDNN 3 RC). I have, however, also made an export of the raw parameter values for the network using [export_params.py](https://github.com/JeffreyDF/kaggle_diabetic_retinopathy/blob/master/export_params.py). You could then replace the cuDNN layers by the layers of your choice or use these parameters to initialise layers in other frameworks.

# SSH

```
ssh -L localhost:8888:localhost:8888 -i ~gpu.pem ubuntu@10.32.?.?
```

# Create an Environment

```
conda create -n kaggle_diabetic_retinopathy python=2.7
conda activate kaggle_diabetic_retinopathy
conda install notebook ipykernel
ipython kernel install --user
pip install -r requirements.txt
jupyter lab
source deactivate
```


# On AWS EC2 for Deep Learning

```
#source activate theano_p27
#pip uninstall Theano
#pip install git+git://github.com/Theano/Theano.git@9a653e3e91c0e38b6643e4452199931e792a24a2
#pip install --upgrade --no-deps git+git://github.com/Lasagne/Lasagne.git@cf1a23c21666fc0225a05d284134b255e3613335
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```


Ref.: https://developer.nvidia.com/rdp/cudnn-archive

# Images

Delete these images from dataset, because they are in very bad shape:

* train/492_right.jpeg
* test/25313_right.jpeg
* test/27096_right.jpeg

# Steps to Train the Model

* python2 preprocess.py
* run split_data.ipynb
* run train_keras.ipynb
