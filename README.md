# Socioeconomic correlations of urban patterns inferred from aerial images: interpreting activation maps of Convolutional Neural Networks
This repository contains code related to the paper [socioeconomic correlations of urban patterns inferred from aerial images: interpreting activation maps of Convolutional Neural Networks](http://perso.ens-lyon.fr/marton.karsai/) (currently in submission).

This repository contains the `Python` codes needed to apply the inference pipeline to similar demographic-enriched twitter data samples.

* code to generate semantics features, reliable home locations and ses-enriched datasets from users tweets and census, is in the [helpers](./python_scripts/helpers) folder

<p float="left">
  <img src="./imags/Fig_topiccorrs.png" width="48%"/>
  <img src="./imags/Fig_gmap_see.png" width="48%"/>
</p>

* `Keras` implementations of the `ResNet50` used in this paper to select residential sites are in the [data collection and processing](./python_scripts/data_coll_process) folder (with a `TensorFlow` backend)

* code to train and validate the models, for each socioeconomic proxy in the [pipelines](./python_scripts/pipelines) folder.


### Citation
If you use the code, data, or analysis results in this paper, we kindly ask that you cite the paper above as:

> _Optimal proxy selection for socioeconomic status inference on Twitter_ , J. Levy Abitbol, E. Fleury, M. Karsai, 2019. In Special Issue on Analysis and Applications of Location-Aware Big Complex Network Data, Complexity, Wiley-Hindawi.
