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
├── src/            # model initialization code + weights
│   ├── lenet5_model.pth
│   ├── model.py
├── README.md           # the repository's showcase
│
├── test.ipynb # model training + loss visualization + confusion matrix & classification report computation + live inference snippet
│
├── IMAGES
│   ├── Laura_Chaubard_&_Yann_Le_Cun_-_2024_(53814052697)_(cropped).jpg # photo of Yann LeCun    
│   ├── LeNet-5_architecture.svg # image of the LeNet-5 architecture    
│   ├── MNIST_dataset_example.png # image of MNIST dataset label examples
│   └── test_visual_predictions.png # model inference snippet output example
```

## The architecture

Every layer is written out rather than pulled from a library, and the shapes are forced by the 32×32 input:

| Layer | Operation | Output | Trainable params |
| --- | --- | --- | --- |
| Input | 32×32 grayscale | `1 × 32 × 32` | — |
| **C1** | `Conv2d(1→6, 5×5)` + scaled tanh | `6 × 28 × 28` | 156 |
| **S2** | `AvgPool2d(2×2, stride 2)` | `6 × 14 × 14` | 0 |
| **C3** | `Conv2d(6→16, 5×5)` + scaled tanh | `16 × 10 × 10` | 2,416 |
| **S4** | `AvgPool2d(2×2, stride 2)` | `16 × 5 × 5` | 0 |
| **C5** | `Conv2d(16→120, 5×5)` + scaled tanh | `120 × 1 × 1` | 48,120 |
| **F6** | `Linear(120→84)` + scaled tanh | `84` | 10,164 |
| **Output** | 10 Euclidean RBF units | `10` distances | 0 (fixed) |
| | | **Total** | **60,856** |

C5 is written as a convolution rather than a linear layer on purpose. Its 5×5 kernel exactly covers the 5×5 input, so it collapses to 120×1×1 — mathematically fully connected, but keeping it a `Conv2d` makes that coincidence visible instead of hiding it behind a `flatten`.

The activation throughout is the paper's scaled hyperbolic tangent, `1.7159 · tanh(⅔x)`. The constants aren't arbitrary: they place the function's most useful gradient where normalized inputs actually live, and give `f(±1) = ±1`.

### The output layer has no softmax

This is the part of LeNet-5 that looks strangest to modern eyes, and it's the reason F6 has exactly **84** units.

Each of the ten output units is a **Euclidean radial basis function** holding a fixed 84-dimensional centre — and that centre is a stylized **7 × 12 bitmap** of the digit it represents (7 × 12 = 84), with `+1` for ink and `-1` for background. The unit outputs the squared distance between F6's activation and its template:

```
..###..  ...#...  .#####.  .#####.  ....##.  #######  ..####.  #######  ..###..  ..###..
.#...#.  ..##...  #.....#  #.....#  ...#.#.  #......  .#.....  .....#.  .#...#.  .#...#.
#.....#  .#.#...  ......#  ......#  ..#..#.  #......  #......  ....#..  #.....#  #.....#
#.....#  ...#...  .....#.  .....#.  .#...#.  #####..  #......  ....#..  .#...#.  #.....#
#.....#  ...#...  ....#..  ..###..  #....#.  .....#.  #####..  ...#...  ..###..  .#...#.
   0        1        2        3        4        5        6        7        8        9
```

The templates are **never trained** — they're registered as a buffer, so no optimizer can reach them. Training instead pushes F6 to *draw* the right bitmap. Three consequences follow:

* **A small output means a confident prediction**, so inference uses `torch.argmin`, not `argmax`.
* **The network is interpretable by construction** — you can read F6's 84 activations back out as a 7×12 picture and see what the model thinks it is looking at.
* **The squash after F6 matters.** It bounds activations to ±1.7159, the same scale as the ±1 templates they're compared against.

The loss follows the paper's MAP criterion — the distance to the correct class, plus `log Σ e^(−distance)` over all classes. The second term stops the network from cheating by collapsing every distance to zero. There's a neat result buried in it: that expression is *exactly* cross-entropy over the negated distances, so the paper arrives at the standard objective from a completely different direction.

### Fidelity to the paper

The goal is a faithful reimplementation, not a high score. The paper's own parameter table sums to **exactly 60,000**, which makes a useful checksum — this implementation sits at **60,856**, and the difference is entirely accounted for by two deliberate simplifications:

| | This repo | Paper | Δ |
| --- | --- | --- | --- |
| C3 connectivity | fully connected to all 6 S2 maps | 60 of 96 links via Table I | +900 |
| S2 / S4 | plain average pooling | trainable coefficient + bias, then squashed | −44 |
| | | | **+856** |

Also simplified: optimization is plain SGD rather than the paper's stochastic diagonal Levenberg-Marquardt, and the loss omits the small positive constant `j` inside the log. The learning-rate ladder in `fit()` *does* follow the paper — 0.0005 for two passes, then 0.0002, 0.0001, 0.00005 and 0.00001.

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
    transforms.ToTensor(),
    transforms.Normalize(...)     # see "A note on normalization" below
])
```

`DataLoader(..., batch_size=64)` then yields exactly what the network expects:

```
Feature batch shape: torch.Size([64, 1, 32, 32])
Labels batch shape:  torch.Size([64])
```

### Why 32×32 and not 28×28

This is the one preprocessing step LeNet-5 genuinely requires. Trace the spatial dimensions through the architecture: C1's 5×5 valid convolution takes 32 → 28, S2 halves it to 14, C3's 5×5 takes 14 → 10, S4 halves it to 5. That final 5×5 is exactly what C5's 5×5 kernel needs to collapse the map to 120×1×1. **Feed a 28×28 image instead and the chain ends at 4×4**, C5's kernel no longer fits its input, and the forward pass fails outright. The 32×32 input is not an aesthetic choice — every layer width in `model.py` is derived from it.

Two ways to get there, and this repo takes the simpler one:

* **Resize** (used here) — `transforms.Resize((32, 32))` rescales the digit itself with bilinear interpolation, so the strokes are enlarged to fill the larger canvas.
* **Pad** (the original paper) — centre the untouched 28×28 digit inside a 32×32 field of background pixels. LeCun's reasoning was that this keeps distinctive features like stroke endpoints and corners near the centre of the receptive field of the highest-level feature detectors, rather than pushed to the border where they are seen by fewer units.

Both produce a correctly shaped tensor and both train; the resize is a deliberate simplification, not an oversight.

### A note on normalization

`ToTensor()` alone scales pixels to `[0, 1]`, which is not what the network is tuned for. The paper normalizes so that background sits at **−0.1** and foreground at **1.175** — values chosen so the input has roughly zero mean and unit variance, which puts it in the steep, high-gradient part of the `1.7159 · tanh(⅔x)` curve rather than out on its flat tails. Since the RBF templates are ±1, keeping the whole signal path on a comparable scale matters more here than in a network that ends in a softmax.

To reproduce the paper's exact endpoints:

```python
transforms.Normalize(mean=[0.078431], std=[0.784314])  # maps 0 -> -0.1, 1 -> 1.175
```

The notebook normalizes to approximately zero mean and unit variance; see [`test.ipynb`](test.ipynb) for the constants actually used.

### Results

<!-- TODO: fill in once the current training run finishes -->
Test accuracy, the confusion matrix and the per-class classification report are produced by [`test.ipynb`](test.ipynb), and the weights that reproduce them are in [`src/lenet5_model.pth`](src/lenet5_model.pth).

A faithful reimplementation is the goal here rather than a leaderboard number, so expect this to land below what a modern small CNN reaches on MNIST — the paper itself reports 99.05%, using an optimizer and a connectivity scheme this repo deliberately simplifies.

## Notes

* `model.py` follows the paper's implementation closely, but it is not a 1:1 reproduction — this is a learning repo, and the value is in the annotations and code writing and understanding of functionality, rather than rethinking outdated techniques of learning and optimization that do not apply to modern day CNN's (I say this even though I implemented some techniques from the original paper, such as the RBFSublayer, the [RBF loss computation](https://ieeexplore.ieee.org/document/9133368), and the custom decreasing learning rate over epochs)
* `test.ipynb` showcases the dataset extraction & visualization, model training loop, loss evolution visualization using matplotlib, confusion matrix computation between the true labels and the predicted ones, a classification report to showcase precision, accuracy, recall and f1-score between the digit classes (from 0-9), and finally a live inference script to observe real sampling and prediction, results I personally find fascinating to say the least

## Credits
![Yann LeCun should be here](IMAGES/Laura_Chaubard_&_Yann_Le_Cun_-_2024_(53814052697)_(cropped).jpg) 
Photographer: Jérémy Barande 
[Photo license](https://creativecommons.org/licenses/by-sa/2.0/deed.en)

All lecture material inspired by [Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun) — [LeNet series](https://en.wikipedia.org/wiki/LeNet).

[The Gradient-based learning applied to document recognition paper](https://ieeexplore.ieee.org/document/726791/)

## 🔗 More

- Author: [@pop123-ux](https://github.com/pop123-ux)
- Medium write-ups: [medium.com/@Pop123](https://medium.com/@Pop123)
