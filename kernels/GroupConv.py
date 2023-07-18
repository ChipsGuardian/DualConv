import torch.nn as nn


class GroupConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride, g):
        """
        Initialize the GroupConv class.
        :param input_channels: the number of input channels
        :param output_channels: the number of output channels
        :param stride: convolution stride
        :param g: the value of G used in GroupConv
        """
        super(GroupConv, self).__init__()
        # Group Convolution
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)

    def forward(self, input_data):
        """
        Define how GroupConv processes the input images or input feature maps.
        :param input_data: input images or input feature maps
        :return: return output feature maps
        """
        return self.gc(input_data)
