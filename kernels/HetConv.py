import torch
import torch.nn as nn


class HetConv(nn.Module):

    def __init__(self, input_channels, output_channels, stride, p):
        """
        Initialize the HetConv class.
        :param input_channels: the number of input channels
        :param output_channels: the number of output channels
        :param stride: convolution stride
        :param p: the value of P used in HetConv
        """
        super(HetConv, self).__init__()
        self.p = p
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filters = nn.ModuleList()
        self.convolution_1x1_index = []
        # Compute the indices of input channels fed to 1x1 convolutional kernels in all filters.
        # These indices of input channels are also the indices of the 1x1 convolutional kernels in the filters.
        # This is only executed when the HetConv class is created,
        # and the execution time is not included during inference.
        for i in range(self.p):
            self.convolution_1x1_index.append(self.compute_convolution_1x1_index(i))
        # Build HetConv filters.
        for i in range(self.p):
            self.filters.append(self.build_HetConv_filters(stride, p))

    def compute_convolution_1x1_index(self, i):
        """
        Compute the indices of input channels fed to 1x1 convolutional kernels in the i-th branch of filters (i=0, 1, 2,…, P-1). The i-th branch of filters consists of the {i, i+P, i+2P,…, i+N-P}-th filters.
        :param i: the i-th branch of filters in HetConv
        :return: return the required indices of input channels
        """
        index = [j for j in range(0, self.input_channels)]
        # Remove the indices of input channels fed to 3x3 convolutional kernels in the i-th branch of filters.
        while i < self.input_channels:
            index.remove(i)
            i += self.p
        return index

    def build_HetConv_filters(self, stride, p):
        """
        Build N/P filters in HetConv.
        :param stride: convolution stride
        :param p: the value of P used in HetConv
        :return: return N/P HetConv filters
        """
        temp_filters = nn.ModuleList()
        # nn.Conv2d arguments: nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        temp_filters.append(nn.Conv2d(self.input_channels//p, self.output_channels//p, 3, stride, 1, bias=False))
        temp_filters.append(nn.Conv2d(self.input_channels-self.input_channels//p, self.output_channels//p, 1, stride, 0, bias=False))
        return temp_filters

    def forward(self, input_data):
        """
        Define how HetConv processes the input images or input feature maps.
        :param input_data: input images or input feature maps
        :return: return output feature maps
        """
        output_feature_maps = []
        # Loop P times to get output feature maps. The number of output feature maps = the batch size.
        for i in range(0, self.p):

            # M/P HetConv filter kernels perform the 3x3 convolution and output to N/P output channels.
            output_feature_3x3 = self.filters[i][0](input_data[:, i::self.p, :, :])
            # (M-M/P) HetConv filter kernels perform the 1x1 convolution and output to N/P output channels.
            output_feature_1x1 = self.filters[i][1](input_data[:, self.convolution_1x1_index[i], :, :])

            # Obtain N/P output feature map channels.
            output_feature_map = output_feature_1x1 + output_feature_3x3

            # Append N/P output feature map channels.
            output_feature_maps.append(output_feature_map)

        # Get the batch size, number of output channels (N/P), height and width of output feature map.
        N, C, H, W = output_feature_maps[0].size()
        # Change the value of C to the number of output feature map channels (N).
        C = self.p * C
        # Arrange the output feature map channels to make them fit into the shifted manner.
        return torch.cat(output_feature_maps, 1).view(N, self.p, C//self.p, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
