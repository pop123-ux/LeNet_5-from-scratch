"""CPU-only smoke tests for the LeNet-5 implementation."""

from pathlib import Path

import torch

from src import LeNet_5, LeNetRBFSublayer, build_digit_bitmaps


def test_digit_bitmaps_have_expected_shape_and_values():
    bitmaps = build_digit_bitmaps()

    assert bitmaps.shape == (10, 84)
    assert set(torch.unique(bitmaps).tolist()) == {-1.0, 1.0}


def test_rbf_centers_are_registered_as_buffer_not_parameter():
    layer = LeNetRBFSublayer()

    buffers = dict(layer.named_buffers())
    parameters = dict(layer.named_parameters())

    assert "centers" in buffers
    assert buffers["centers"].shape == (10, 84)
    assert "centers" not in parameters


def test_forward_shape_and_nonnegative_distances():
    torch.manual_seed(0)
    model = LeNet_5()
    model.eval()
    x = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()
    assert torch.all(output >= 0)


def test_backward_reaches_trainable_parameters():
    torch.manual_seed(0)
    model = LeNet_5()
    model.train()
    x = torch.randn(4, 1, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])

    distances = model(x)
    loss = model._lenet_rbf_loss(distances, labels)
    loss.backward()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)


def test_predict_uses_minimum_distance_class():
    torch.manual_seed(0)
    model = LeNet_5()
    x = torch.randn(3, 1, 32, 32)

    with torch.no_grad():
        expected = torch.argmin(model(x), dim=1)

    predicted = model.predict(x)

    assert torch.equal(predicted, expected)


def test_parameter_counter_matches_model_parameters():
    model = LeNet_5()
    expected = sum(parameter.numel() for parameter in model.parameters())

    assert model.params() == f"{expected} total trainable parameters"


def test_default_checkpoint_points_to_checkpoints_directory():
    checkpoint = Path(LeNet_5.DEFAULT_WEIGHTS)

    assert checkpoint.parent.name == "checkpoints"
    assert checkpoint.name == "lenet5_mnist.pth"
