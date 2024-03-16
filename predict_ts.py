"""
Train conv network to predict transition state dihedrals for the LysArgMet system
"""

import torch
from torch import nn
from torch import optim
from safetensors.torch import load_file, save_file


class MyModel(nn.Module):
    """
    Convolutional neural network for predicting transition state dihedrals
    """

    def __init__(self):
        super(MyModel, self).__init__()

        self.dropout_1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
        )
        self.flatten = nn.Flatten(1, 3)
        # self.linear0 = torch.nn.Linear(356, 1424)
        self.linear1 = torch.nn.Linear(356, 712)
        self.act1 = torch.nn.Tanh()
        # self.linear2 = torch.nn.Linear(712, 356)
        self.act2 = torch.nn.Tanh()
        self.dropout_2 = nn.Dropout(0.2)
        self.linear3 = torch.nn.Linear(712, 178)

    def forward(self, x):
        """
        Run the neural network forward
        """
        x = self.dropout_1(x)
        x = self.conv1(x)
        x = self.flatten(x)
        # x = self.linear0(x)
        x = self.linear1(x)
        x = self.act1(x)
        # x = self.linear2(x)
        x = self.act2(x)
        x = self.dropout_2(x)
        x = self.linear3(x)
        return x

    def load_st(self, file_path: str):
        """
        Load a state dict from a file path.
        """
        loaded = load_file(file_path)
        self.conv1.weight = nn.Parameter(loaded["conv1.weight"])
        self.conv1.bias = nn.Parameter(loaded["conv1.bias"])
        self.linear1.weight = nn.Parameter(loaded["ln1.weight"])
        self.linear1.bias = nn.Parameter(loaded["ln1.bias"])
        # self.linear2.weight = nn.Parameter(loaded["ln2.weight"])
        # self.linear2.bias = nn.Parameter(loaded["ln2.bias"])
        self.linear3.weight = nn.Parameter(loaded["ln3.weight"])
        self.linear3.bias = nn.Parameter(loaded["ln3.bias"])

    def save_st(self, file_path: str):
        """
        Save a state dict to a file path.
        """
        # loaded = load_file(file_path)
        save_data = {}
        save_data["conv1.weight"] = self.conv1.weight
        save_data["conv1.bias"] = self.conv1.bias
        save_data["ln1.weight"] = self.linear1.weight
        save_data["ln1.bias"] = self.linear1.bias
        # save_data["ln2.weight"] = self.linear2.weight
        # save_data["ln2.bias"]= self.linear2.bias
        save_data["ln3.weight"] = self.linear3.weight
        save_data["ln3.bias"] = self.linear3.bias
        save_file(save_data, file_path)


def angle_mse(preds, actual):
    """
    Angular MSE by mapping to sin cos
    """
    preds_cos = torch.cos(preds)
    preds_sin = torch.sin(preds)
    actual_cos = torch.cos(actual)
    actual_sin = torch.sin(actual)
    return torch.mean((preds_cos - actual_cos) ** 2 + (preds_sin - actual_sin) ** 2)


if __name__ == "__main__":
    # When initialzing, it will run __init__() function as above
    model = MyModel()
    # model.load_st("weights.st")
    model.cuda()
    inputs = load_file("dihedral_data.st")

    train_in = inputs["traininput"].type(torch.FloatTensor).cuda()
    train_out = inputs["trainoutput"].type(torch.FloatTensor).cuda()

    test_in = inputs["testinput"].type(torch.FloatTensor).cuda()
    test_out = inputs["testoutput"].type(torch.FloatTensor).cuda()
    # define loss and parameters
    optimizer = optim.NAdam(model.parameters())
    EPOCHS = 200_000

    print("====Training start====")
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    for i in range(EPOCHS):
        # for i in range(1):
        # prepare input data
        # data = torch.ones(1,1,28,28).bfloat16().cuda()
        # data.to(torch.float32)
        # print(data.dtype)
        # print(data.shape)
        # inputs = torch.reshape(data,(-1, 784)) # -1 can be any value. So when reshape,
        # it will satisfy 784 first

        # set gradient to zero
        optimizer.zero_grad()

        # feed inputs into model
        pred_data = model.forward(train_in)
        # print(recon_x)
        # print(recon_x.shape)
        # print(recon_x.shape)

        # calculating loss
        loss = angle_mse(pred_data, train_out)

        # calculate gradient of each parameter
        loss.backward()

        # update the weight based on the gradient calculated
        optimizer.step()

        model.eval()
        test_loss = angle_mse(model.forward(test_in), test_out)
        model.train()

        if i % 10 == 0:
            print(
                "====> Epoch: {}  train loss: {:.9f}, test loss {:.8f}".format(
                    i,
                    180 * ((2 - loss) / 2).acos() / 3.1415926,
                    180 * ((2 - test_loss) / 2).acos() / 3.1415926,
                )
            )
        if i % 1000 == 0:
            scheduler1.step()

    print("====Training finish====")
    model.save_st("weights.st")
