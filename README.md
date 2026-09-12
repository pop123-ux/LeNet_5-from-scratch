# LeNet_5-from-scratch in PyTorch
[![Tests](https://github.com/pop123-ux/LeNet_5-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/pop123-ux/LeNet_5-from-scratch/actions/workflows/tests.yml)

<img width="1672" height="941" alt="LeNet-5 project header" src="https://github.com/user-attachments/assets/8238726c-38dc-49d2-8aa1-c514df83cd9b" />

---

## The architecture

![LeNet-5 architecture](IMAGES/LeNet-5_architecture.svg)

This is my in-depth PyTorch reimplementation of Yann LeCun's [**LeNet-5**](https://en.wikipedia.org/wiki/LeNet), the first project in my **Visual Scrambling** series. The series is built around reconstructing influential computer-vision architectures from their underlying designs before relying on high-level model libraries.

My main goal with LeNet-5 was not simply to reproduce a small CNN. I wanted to understand the tensor mechanics that are easy to overlook when modern libraries hide them—especially convolution geometry, pooling, broadcasting, and the unusual radial-basis-function output mechanism used by the original architecture.

A second goal of Visual Scrambling is to learn by **implementing and explaining**, not only by reading. The repository therefore keeps the model intentionally explicit: each stage is visible, the experiment remains in a notebook, and the implementation is small enough to modify directly while still being supported by automated tests and a reproducible checkpoint.

## Layout

```text
├── .github/
│   └── workflows/
│       └── tests.yml              # CPU-only pytest CI on pushes and pull requests
│
├── IMAGES/
│   ├── Laura_Chaubard_&_Yann_Le_Cun_-_2024_(53814052697)_(cropped).jpg
│   ├── LeNet-5_architecture.svg
│   ├── MNIST_dataset_example.png
│   └── test_visual_predictions.png
│
├── checkpoints/
│   └── lenet5_mnist.pth           # trained MNIST state_dict
│
├── src/
│   ├── __init__.py
│   └── model.py                   # architecture, RBF layer, training/evaluation and checkpoint API
│
├── tests/
│   └── test_model.py              # model, RBF, gradient and checkpoint-path smoke tests
│
├── .gitignore
├── LICENSE
├── pyproject.toml                 # project and development dependencies
├── README.md
└── test.ipynb                     # MNIST training, metrics and live inference experiment
```

`data/MNIST/raw/` is created locally by `torchvision.datasets.MNIST` when the notebook downloads MNIST; it is runtime data rather than a tracked repository artifact.

## Architecture breakdown

Every layer is written out explicitly, and the spatial dimensions are forced by the 32×32 input:

| Layer | Operation | Output | Trainable params |
| --- | --- | --- | ---: |
| Input | 32×32 grayscale | `1 × 32 × 32` | — |
| **C1** | `Conv2d(1→6, 5×5)` + scaled tanh | `6 × 28 × 28` | 156 |
| **S2** | `AvgPool2d(2×2, stride 2)` | `6 × 14 × 14` | 0 |
| **C3** | `Conv2d(6→16, 5×5)` + scaled tanh | `16 × 10 × 10` | 2,416 |
| **S4** | `AvgPool2d(2×2, stride 2)` | `16 × 5 × 5` | 0 |
| **C5** | `Conv2d(16→120, 5×5)` + scaled tanh | `120 × 1 × 1` | 48,120 |
| **F6** | `Linear(120→84)` + scaled tanh | `84` | 10,164 |
| **Output** | 10 Euclidean RBF units | `10` distances | 0 (fixed) |
| | | **Total** | **60,856** |

C5 is written as a convolution rather than a linear layer on purpose. Its 5×5 kernel exactly covers the 5×5 input, so it collapses the feature map to 120×1×1. It is mathematically equivalent to a fully connected operation over that input, but keeping it as `Conv2d` makes the architectural relationship visible instead of hiding it behind a flattening step.

The activation throughout is the paper's scaled hyperbolic tangent, `1.7159 · tanh(⅔x)`. The constants are not arbitrary: they place much of the useful input range in a region with strong gradients and give `f(±1) = ±1`.

### The output layer has no softmax

This is one of the most distinctive parts of the original LeNet-5 design and the reason F6 has exactly **84** units.

Each of the ten output units is a **Euclidean radial basis function** with a fixed 84-dimensional center. Each center represents a stylized **7 × 12 bitmap** of one digit (`7 × 12 = 84`), using `+1` for ink and `-1` for background. The output is the squared Euclidean distance between the learned F6 representation and each class template:

```text
..###..  ...#...  .#####.  .#####.  ....##.  #######  ..####.  #######  ..###..  ..###..
.#...#.  ..##...  #.....#  #.....#  ...#.#.  #......  .#.....  .....#.  .#...#.  .#...#.
#.....#  .#.#...  ......#  ......#  ..#..#.  #......  #......  ....#..  #.....#  #.....#
#.....#  ...#...  .....#.  .....#.  .#...#.  #####..  #......  ....#..  .#...#.  #.....#
#.....#  ...#...  ....#..  ..###..  #....#.  .....#.  #####..  ...#...  ..###..  .#...#.
   0        1        2        3        4        5        6        7        8        9
```

The templates are **never trained**. They are registered as a buffer, so the optimizer cannot update them. Training instead pushes the F6 representation toward the correct class template. Three consequences follow:

- **A smaller output means a stronger match**, so inference uses `torch.argmin`, not `argmax`.
- The 84-dimensional F6 representation can be reshaped into the same 7×12 layout as the templates, giving a visual interpretation of what the network is learning.
- **The scaled tanh after F6 matters** because it keeps activations on a scale comparable with the fixed ±1 templates.

The loss follows the paper's MAP-style criterion: the distance to the correct class plus `log Σ e^(−distance)` across all classes. The second term prevents the network from minimizing every distance simultaneously. Algebraically, this expression is equivalent to cross-entropy over the negated distances, even though it arrives there from the RBF formulation rather than a conventional logit head.

### Fidelity to the paper

The goal is a learning-oriented reimplementation that preserves the important architectural ideas while using modern PyTorch infrastructure—not an exact reproduction of the original training system.

The parameter table in the paper sums to **60,000**. This implementation contains **60,856** trainable parameters, and the difference is accounted for by two deliberate simplifications:

| Component | This repository | Original paper | Δ |
| --- | --- | --- | ---: |
| C3 connectivity | Fully connected to all 6 S2 maps | 60 of 96 connections via Table I | +900 |
| S2 / S4 | Fixed average pooling | Trainable coefficient + bias, then squashed | −44 |
| | | **Net difference** | **+856** |

Optimization is also simplified to PyTorch SGD rather than the paper's stochastic diagonal Levenberg-Marquardt method, and the loss omits the small positive constant `j` inside the logarithm. The staged learning-rate values used by `fit()` follow the original schedule: 0.0005 for two passes, then 0.0002, 0.0001, 0.00005, and 0.00001.

## The MNIST dataset

![MNIST example samples](IMAGES/MNIST_dataset_example.png)

MNIST is the benchmark used by the original LeNet-5 work. It contains 70,000 handwritten digits assembled from NIST data and reorganized so that the training and test sets come from **disjoint groups of writers**. That separation matters because it reduces the possibility of a model succeeding by memorizing one person's writing style.

| Split | Images | Size | Channels | Classes |
| --- | ---: | --- | ---: | ---: |
| Train | 60,000 | 28 × 28 | 1 | 10 |
| Test | 10,000 | 28 × 28 | 1 | 10 |

The classes are close to balanced but not perfectly balanced. For that reason the experiment notebook reports a **per-class classification report** in addition to overall accuracy.

### Loading

`torchvision.datasets.MNIST` downloads the canonical IDX files under `data/MNIST/raw/`. They are binary image/label files rather than ordinary image files, so they are parsed by the dataset loader rather than opened directly with an image reader.

```python
my_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(...)  # see "A note on normalization" below
])
```

With `batch_size=64`, the data loader produces:

```text
Feature batch shape: torch.Size([64, 1, 32, 32])
Labels batch shape:  torch.Size([64])
```

### Why 32×32 instead of 28×28

Trace the spatial dimensions through the architecture: C1's 5×5 valid convolution takes `32 → 28`, S2 halves it to `14`, C3 takes `14 → 10`, and S4 halves it to `5`. That final 5×5 feature map is exactly what C5's 5×5 kernel needs to collapse the tensor to `120 × 1 × 1`.

Feeding a 28×28 image directly would eventually produce a 4×4 feature map, so C5's 5×5 kernel would no longer fit. The 32×32 input is therefore an architectural requirement rather than a cosmetic preprocessing choice.

Two reasonable ways to reach 32×32 are:

- **Resize** — used in this repository. The 28×28 digit is rescaled to 32×32.
- **Pad** — closer to the original preprocessing. The 28×28 digit is centered inside a 32×32 background field without resizing the strokes.

This repository uses resizing as a deliberate simplification.

### A note on normalization

`ToTensor()` alone maps pixel values to `[0, 1]`, while the historical LeNet-5 setup used a different scale. The paper's preprocessing approximately maps background to **−0.1** and foreground to **1.175**, placing inputs in a useful region of the scaled-tanh activation.

To reproduce those endpoints:

```python
transforms.Normalize(mean=[0.078431], std=[0.784314])  # maps 0 -> -0.1, 1 -> 1.175
```

The recorded notebook experiment uses:

```python
transforms.Normalize(mean=[0.1], std=[0.278])
```

This is another documented experimental simplification rather than an exact reproduction of the paper's preprocessing.

## Experimental setup

The reported experiment is intended to demonstrate the reconstructed architecture and its original-style RBF output mechanism on MNIST rather than maximize test accuracy using modern optimization tricks.

| Category | Setting |
| --- | --- |
| Hardware | `CPU` |
| Software | `Python, PyTorch, torchvision` |
| Dataset | `MNIST` |
| Input | `1x32x32` |
| Epochs | `15` |
| Batch size | `64` |
| Optimizer | `SGD` |
| Initial learning rate | `0.0005` |
| Learning-rate schedule | `0.0005 → 0.0002 → 0.0001 → 0.00005 → 0.00001` |
| Loss | `Custom LeNet-5 RBF loss` |
| Activation | `Scaled tanh: 1.7159 · tanh(⅔x)` |
| Output | `10 fixed Euclidean RBF centers` |
| Checkpoint | `checkpoints/lenet5_mnist.pth` |

Because the published run does not freeze every dependency version or record a complete deterministic seeding configuration, small numerical differences may occur when reproducing the experiment.

### Pretrained checkpoint

The state dictionary produced by the recorded MNIST experiment is stored at `checkpoints/lenet5_mnist.pth`. The model's default `save()` and `load()` methods resolve to that root-level checkpoint directory:

```python
from src import LeNet_5

model = LeNet_5()
model.load()
model.eval()
```

### Automated tests and CI

`tests/test_model.py` checks the LeNet-5 forward pass, RBF bitmap values, fixed-center buffer registration, non-negative RBF distances, gradient propagation, argmin-based prediction behavior, parameter reporting, and the default checkpoint path. It uses random tensors only—no MNIST download and no training loop.

Run the suite locally with:

```bash
python -m pytest tests -q
```

The same test command runs automatically in GitHub Actions on pushes and pull requests to `main`. The badge at the top of this README reflects the latest workflow result.

### Results

![Sample predictions](IMAGES/test_visual_predictions.png)

The recorded notebook run reached **98.59% test accuracy** on the 10,000-image MNIST test split. Different runs may vary slightly because the published experiment does not freeze every source of randomness.

The overall accuracy is only one part of the experiment. [`test.ipynb`](test.ipynb) also produces a confusion matrix and a per-class classification report with precision, recall, and F1 score for each digit.

For context, the original paper reports 99.05% on MNIST using architectural and optimization components that this repository deliberately simplifies, including trainable subsampling, the sparse C3 connection table, and the historical second-order optimizer. The goal here is architectural understanding rather than matching that number through unrelated modern training tricks.

## Limitations

This repository does not claim exact replication of the original 1998 LeNet-5 training system.

The main limitations are:

- C3 uses full connectivity rather than the original sparse connection table.
- S2 and S4 use fixed average pooling rather than trainable subsampling functions.
- The loss omits the small positive constant `j` from the original formulation.
- MNIST images are resized from 28×28 to 32×32; the original preprocessing centered the unscaled digit inside a 32×32 field.
- Dependency versions are minimum-version specifications rather than a fully frozen environment.

These are deliberate trade-offs for a small, readable educational repository whose purpose is understanding the architecture, tensor transformations, RBF output mechanism, and training process rather than reproducing the historical system byte-for-byte.

## Lessons learned

- The biggest concept I wanted to understand was **broadcasting**. The F6 representation has shape `[batch, 84]`, while the ten fixed digit centers have shape `[10, 84]`. Reshaping them to `[batch, 1, 84]` and `[1, 10, 84]` lets PyTorch broadcast the subtraction into `[batch, 10, 84]`, after which reduction across the final dimension produces one distance per class.
- Working through the spatial dimensions made the apparently arbitrary 32×32 input much easier to understand: `32 → 28 → 14 → 10 → 5 → 1`. Every spatial dimension is constrained by the next operation.
- The project also made the difference between a **modern logit classification head** and **the original LeNet-5 RBF output design** much clearer. Instead of producing logits directly, the network learns an 84-dimensional representation that is compared with fixed visual prototypes.

## Notes

- `src/model.py` is a ground-up PyTorch reconstruction of the LeNet-5 architecture studied from the original 1998 paper. It makes the architecture and mathematics explicit using modern PyTorch infrastructure rather than attempting to reproduce the historical implementation byte-for-byte.
- The most deliberately preserved historical component is the **RBF output layer**: fixed 7×12 digit templates are stored as non-trainable buffers, the F6 representation is compared with them using squared Euclidean distance, and predictions use `argmin` instead of `argmax`.
- The custom RBF loss is implemented directly rather than replacing the historical output formulation with an ordinary softmax classification head.
- `test.ipynb` contains dataset loading and visualization, the training loop, loss curves, confusion-matrix computation, a per-class classification report, and live inference examples.

## Credits

![Yann LeCun](IMAGES/Laura_Chaubard_&_Yann_Le_Cun_-_2024_(53814052697)_(cropped).jpg)

The project is based primarily on the original LeNet-5 work:

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791/).
- [Levenberg-Marquardt algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm), used as part of the historical optimization approach.

## Image credits

Some visual assets used in this repository are sourced from Wikimedia Commons:

- **Yann LeCun photograph** — Jérémy Barande, licensed under [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/).
- **LeNet-5 architecture image** — Zhang, Aston; Lipton, Zachary C.; Li, Mu; Smola, Alexander J. Originally from [Dive into Deep Learning](https://github.com/d2l-ai/d2l-en), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). [Wikimedia Commons source](https://commons.wikimedia.org/wiki/File:LeNet-5_architecture.svg).
- **MNIST dataset example image** — Suvanjanprasai, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). [Wikimedia Commons source](https://commons.wikimedia.org/wiki/File:MNIST_dataset_example.png).

These third-party images are **not covered by this repository's MIT License**. Their respective copyright and licensing terms continue to apply.

## 🔗 More

- Author: [Pop Alexandru](https://github.com/pop123-ux)
- Medium write-ups: [medium.com/@Pop123](https://medium.com/@Pop123)
- Hugging Face: [pop123ux](https://huggingface.co/pop123ux)
