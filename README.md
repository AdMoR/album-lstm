# Album-lstm

This project implements a RNN using features from a Resnet-18 to predict the theme of a photo album
Example of predicted themes are : 
- Birthday
- Protest
- Zoo

[We use images from the The Curation of Flickr Events Dataset (CUFED) dataset](http://acsweb.ucsd.edu/~yuw176/event-curation.html)

Here are some example of pics of the albums :

![Pic1](https://github.com/AdMoR/album-lstm/blob/master/CUFED_mini/images/0_7138083@N04/5271641595.jpg?raw=true "Mall")

![Pic2](https://github.com/AdMoR/album-lstm/blob/master/CUFED_mini/images/0_7138083@N04/5271648275.jpg?raw=true "Travel")


The project can easily be run with the latest docker image of pytorch :

`nvidia-docker `

Once in the docker, run the install of packages :

`pip install scikit-image tensorbaordx`

To run the training, please launch :

 `python run_training`

