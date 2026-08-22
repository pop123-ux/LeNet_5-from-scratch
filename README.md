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

MNIST is the benchmark the 1998 paper itself was built around — 70,000 handwritten digits assembled from NIST's Special Database 1 and 3, remixed by LeCun, Cortes and Burges so that the training and test sets come from **disjoint groups of writers**. That last detail is the whole point of the dataset: a model cannot score well by memorizing one person's handwriting, because nobody who wrote a training digit wrote a test digit.

| | Images | Size | Channels | Classes |
| --- | --- | --- | --- | --- |
| Train | 60,000 | 28 × 28 | 1 (grayscale) | 10 (digits 0–9) |
| Test | 10,000 | 28 × 28 | 1 (grayscale) | 10 (digits 0–9) |

The classes are close to balanced but not exactly — in the training split digit `1` appears 6,742 times and digit `5` only 5,421, a spread of about 24%. It is small enough to ignore for training, but it is the reason a **per-class classification report** is worth reading over a single accuracy number: a model that quietly fails on `5`s loses less overall accuracy than one that fails on `1`s.

### Loading

`torchvision.datasets.MNIST` handles the download into [`data/MNIST/raw`](data/MNIST/raw), which is where the four canonical IDX binaries land — images and labels, train and test. They are not image files; each is a flat binary with a short header (magic number, count, rows, columns) followed by raw pixel bytes, which is why they need a reader rather than a plain `imread`.

```python
my_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Changes 28x28 to 32x32
    transforms.ToTensor()
])
```

`DataLoader(..., batch_size=64)` then yields exactly what the network expects:

```
Feature batch shape: torch.Size([64, 1, 32, 32])
Labels batch shape:  torch.Size([64])
```

### Why 32×32 and not 28×28

This is the one preprocessing step LeNet-5 genuinely requires. Trace the spatial dimensions through the architecture: C1's 5×5 valid convolution takes 32 → 28, S2 halves it to 14, C3's 5×5 takes 14 → 10, S4 halves it to 5. That final 5×5×16 = 400 is what flattens into C5's 400 → 120 linear layer. **Feed a 28×28 image instead and the chain ends at 4×4×16 = 256**, and the flatten no longer matches `nn.Linear(16 * 5 * 5, 120)`. The 32×32 input is not an aesthetic choice — the layer widths in `model.py` are derived from it.

Two ways to get there, and this repo takes the simpler one:

* **Resize** (used here) — `transforms.Resize((32, 32))` rescales the digit itself with bilinear interpolation, so the strokes are enlarged to fill the larger canvas.
* **Pad** (the original paper) — centre the untouched 28×28 digit inside a 32×32 field of background pixels. LeCun's reasoning was that this keeps distinctive features like stroke endpoints and corners near the centre of the receptive field of the highest-level feature detectors, rather than pushed to the border where they are seen by fewer units.

Both produce a correctly shaped tensor and both train; the resize is a deliberate simplification, not an oversight.

### A note on normalization

The pipeline stops at `ToTensor()`, which scales pixels to `[0, 1]`. The paper went further and normalized so that background sits near **-0.1** and foreground near **1.175**, chosen to pair with the scaled tanh activation `1.7159 * tanh(...)` used throughout the network — roughly mean-zero inputs land in the steep, high-gradient part of the tanh curve rather than its flat tails. Adding a `transforms.Normalize` step is therefore the most faithful next change to make, and a natural experiment to run against the current result.

Even without it, the model reaches **98.05% test accuracy** — see the confusion matrix and classification report in [`test.ipynb`](test.ipynb) for where the remaining errors concentrate.

## Notes

* `model.py` follows the paper's implementation closely, but it is not a 1:1 reproduction — this is a learning repo, and the value is in the annotations and code writing and understanding of functionality, rather than rethinking outdated techniques of learning and optimization that do not apply to modern day CNN's (I say this even though I implemented some techniques from the original paper, such as the RBFSublayer, the [RBF loss computation](https://ieeexplore.ieee.org/document/9133368), and the custom decreasing learning rate over epochs)
* `test.ipynb` showcases the dataset extraction & visualization, model training loop, loss evolution visualization using matplotlib, confusion matrix computation between the true labels and the predicted ones, a classification report to showcase precision, accuracy, recall and f1-score between the digit classes (from 0-9), and finally a live inference script to observe real sampling and prediction, results I personally find fascinating to say the least

## Credits
![Yann LeCun should be here](IMAGES/Laura_Chaubard_&_Yann_Le_Cun_-_2024_(53814052697)_(cropped).jpg)
All lecture material inspired by [Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun) — [LeNet series](https://en.wikipedia.org/wiki/LeNet).

[The Gradient-based learning applied to document recognition paper](https://ieeexplore.ieee.org/document/726791/)

## 🔗 More

- Author: [@pop123-ux](https://github.com/pop123-ux)
- Medium write-ups: [medium.com/@Pop123](https://medium.com/@Pop123)
