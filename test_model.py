from model import UNET
import torch
import torch.nn as nn
import numpy as np
import pytest


#TODO adding comments
def test_model_even_one_channel():
    """
    Test that it can sum a list of integers
    """
    input_channels=3
    x = torch.rand((5,input_channels,161,161))
    model = UNET(input_channels=input_channels, output_channels=64)
    pred = model(x)

def test_model_even_three_channels():
    """
    Test that it can sum a list of integers
    """
    input_channels=3
    x = torch.rand((5,input_channels,161,161))
    model = UNET(input_channels=input_channels, output_channels=64)
    pred = model(x)

def test_model_odd_one_channel():
    """
    Test that it can sum a list of integers
    """
    input_channels=3
    x = torch.rand((5,input_channels,161,161))
    model = UNET(input_channels=input_channels, output_channels=64)
    pred = model(x)


def test_model_odd_three_channels():
    """
    Test that it can sum a list of integers
    """
    input_channels=3
    x = torch.rand((5,input_channels,159,159))
    model = UNET(input_channels=input_channels, output_channels=64)
    pred = model(x)

