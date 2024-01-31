import torch
from torch import nn

# https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705


# the model itself
class UNET_SMALLER(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), use_drop=True, use_batch=True):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, kernel_size, padding=1, use_drop=use_drop, use_batch=use_batch)
        self.conv2 = self.contract_block(32, 64, kernel_size, padding=1, use_drop=use_drop, use_batch=use_batch)

        self.upconv2 = self.expand_block(64, 32, kernel_size, 1, output_padding=1, use_drop=use_drop, use_batch=use_batch)
        self.upconv1 = self.expand_block(32 * 2, out_channels, kernel_size, 1, output_padding=1, use_drop=use_drop, use_batch=use_batch)

        self.use_drop = use_drop
        self.use_batch = use_batch

        # self.relu = nn.ReLU()

    def __call__(self, input_x):
        # downsampling part
        conv1 = self.conv1(input_x)
        conv2 = self.conv2(conv1)

        upconv2 = self.upconv2(conv2)
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    @staticmethod
    def contract_block(in_channels, out_channels, kernel_size, padding, use_drop, use_batch):

        if use_drop == False and use_batch == False:
            contract = nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=(1, 1), padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.Conv2d(out_channels, out_channels,
                                kernel_size=kernel_size, stride=(1, 1), padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        elif use_drop == True and use_batch == False:
            contract = nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=(1, 1), padding=padding),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.2),
                torch.nn.Conv2d(out_channels, out_channels,
                                kernel_size=kernel_size, stride=(1, 1), padding=padding),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.2),
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        elif use_drop == False and use_batch == True:
            contract = nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=(1, 1), padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels,
                                kernel_size=kernel_size, stride=(1, 1), padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        elif use_drop == True and use_batch == True:
            contract = nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=(1, 1), padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.2),
                torch.nn.Conv2d(out_channels, out_channels,
                                kernel_size=kernel_size, stride=(1, 1), padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.2),
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            raise ValueError("Combination of drop and batch is wrong")
        return contract

    @staticmethod
    def expand_block(in_channels, out_channels, kernel_size, padding, output_padding=(0, 0), use_drop=True, use_batch=True):

        if use_drop == False and use_batch == False:
            expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=(1, 1), padding=padding),
                                   torch.nn.ReLU(),
                                   torch.nn.Conv2d(
                                       out_channels, out_channels, kernel_size, stride=(1, 1), padding=padding),
                                   torch.nn.ReLU(),
                                   torch.nn.ConvTranspose2d(
                                       out_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                       output_padding=output_padding)
                                   )
        elif use_drop == True and use_batch == False:
            expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=(1, 1), padding=padding),
                                   torch.nn.Dropout(p=0.2),
                                   torch.nn.ReLU(),
                                   torch.nn.Conv2d(
                                       out_channels, out_channels, kernel_size, stride=(1, 1), padding=padding),
                                   torch.nn.Dropout(p=0.2),
                                   torch.nn.ReLU(),
                                   torch.nn.ConvTranspose2d(
                                       out_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                       output_padding=output_padding)
                                   )
        elif use_drop == False and use_batch == True:
            expand = nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=(1, 1), padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    out_channels, out_channels, kernel_size, stride=(1, 1), padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(
                    out_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                    output_padding=output_padding)
                )
        elif use_drop == True and use_batch == True:
            expand = nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=(1, 1), padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.Dropout(p=0.2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    out_channels, out_channels, kernel_size, stride=(1, 1), padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.Dropout(p=0.2),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(
                    out_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                    output_padding=output_padding)
                )
        else:
            raise ValueError("Combination of drop and batch is wrong")

        return expand
