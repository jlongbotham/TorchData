# Torch Data

## Image Classification

* `cifar10-train.t7` and `cifar10-test.t7` are from the [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html) and contain 32x32 images in 10 classes.
* `image classification.lua` is the script we'll be working with from an *iTorch notebook*

## Language Modelling

* `billionwords.tar.gz` contains ten .th7 files (`train_data.th7` and `train_small.th7` are empty -- instead we'll be using `train_tiny.th7`). Make sure to save and uncompress this file in `/home/<user>/data/BillionWords` -- otherwise the script will try to download the complete 3.5 GB (!) dataset from a server in Montreal.
* `languagemodel.lua` is the script we'll be working with from the *command line*

## Dependencies

### *Both Tutorials*

* install [Torch](http://torch.ch/docs/getting-started.html)

### *Image Classification*

* install [iPython with notebooks](http://ipython.org/install.html)
* install [iTorch](https://github.com/facebook/iTorch)

### *Language Modelling*

* install [dp](http://dp.readthedocs.org/en/latest/#dp-package-reference-manual): `luarocks install dp`
* install fs: `luarocks install fs`

As the main scripting language for Torch is lua, you might want to take a look at this [15-minute intro](http://tylerneylon.com/a/learn-lua/)



