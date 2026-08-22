# LeNet_5-from-scratch
![Architecture Image](IMAGES/LeNet-5_architecture.svg)
My working in-depth implementation of Yann LeCun's Masterpiece: [**the LeNet-5 CNN that started it all**](https://en.wikipedia.org/wiki/LeNet). This is the first project from my Visual Scrambling series in which I reimplement from scratch the most influential classic architectures and ending with a unique visual model design written and designed by me.

The point of the series is to go in more depth into the PyTorch framework and understand most importantly the broadcasting part, since at the writing of this readme, that's what I find the most difficult. In addition to that, another goal of this series is to promote understanding by writing and not solely by reading, since in these days, some can become a little bit too inclined into asking the latest LLMs for understanding and for implementation, leaving gaps in understanding. That's even why I decided to leave the repo as it is in it's "naked form" (anyone can play with the model and change the class of the model or even add more methods if they feel like it, that's why I leave it in this kind of infant form, with a notebook for testing and a script for model initialization, which may feel a little bit unorthodox for some)

## Layout

```
├── data/MNIST/raw (basic boilerplate torchvision MNIST import baseline)
│   ├── ...    
│   ├── ...    
│   └── ...
│   .
│   .
│   .       
├── src/            # model initialization code + weights
│   ├── lenet5_model.pth
│   ├── model.py
├── README.md           # the repository's showcase
│
├── test.ipynb # model training + loss visualization + confusion matrix & classification report computation + live inference snippet
│
├── IMAGES
│   ├── ...    
│   ├── ...    
│   └── ...
│   .
│   .
│   .  
```

## The MNIST dataset
![MNIST example samples should be here](IMAGES/MNIST_dataset_example.png)


## Notes

* `model.py` follows the paper's implementation closely, but it is not a 1:1 reproduction — this is a learning repo, and the value is in the annotations and code writing and understanding of functionality, rather than rethinking outdated techniques of learning and optimization that do not apply to modern day CNN's (I say this even though I implemented some techniques from the original paper, such as the RBFSublayer, the [RBF loss computation](https://ieeexplore.ieee.org/document/9133368), and the custom decreasing learning rate over epochs)
* `test.ipynb` showcases the dataset extraction & visualization, model training loop, loss evolution visualization using matplotlib, confusion matrix computation between the true labels and the predicted ones, a classification report to showcase precision, accuracy, recall and f1-score between the digit classes (from 0-9), and finally a live inference script to observe real sampling and prediction, results I personally find fascinating to say the least

## Credits

All lecture material inspired by [Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun) — [LeNet series](https://en.wikipedia.org/wiki/LeNet).

[The Gradient-based learning applied to document recognition paper](https://ieeexplore.ieee.org/document/726791/)

## 🔗 More

- Author: [@pop123-ux](https://github.com/pop123-ux)
- Medium write-ups: [medium.com/@Pop123](https://medium.com/@Pop123)
