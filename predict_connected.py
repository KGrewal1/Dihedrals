"""
Train conv network to predict whether two pairs of dihedrals are connected for the LysArgMet system
"""

import torch
from torch import nn
from torch import optim
from safetensors.torch import load_file, save_file


class CxModel(nn.Module):
    """
    The module defining the structure of the neural network used for prediction
    """

    def __init__(self):
        super(CxModel, self).__init__()

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
        self.linear1 = torch.nn.Linear(356, 712)
        self.act1 = torch.nn.Tanh()
        self.dropout_2 = nn.Dropout(0.6)
        self.linear2 = torch.nn.Linear(712, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the neural network
        """
        x = self.dropout_1(x)
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.dropout_2(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
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
        self.linear2.weight = nn.Parameter(loaded["ln2.weight"])
        self.linear2.bias = nn.Parameter(loaded["ln2.bias"])

    def save_st(self, file_path: str):
        """
        Save a state dict to a file path.
        """
        save_data = {}
        save_data["conv1.weight"] = self.conv1.weight
        save_data["conv1.bias"] = self.conv1.bias
        save_data["ln1.weight"] = self.linear1.weight
        save_data["ln1.bias"] = self.linear1.bias
        save_data["ln2.weight"] = self.linear2.weight
        save_data["ln2.bias"] = self.linear2.bias
        save_file(save_data, file_path)


if __name__ == "__main__":
    # When initialzing, it will run __init__() function as above
    model = CxModel()
    # model.load_st("weights.st")
    model.cuda()
    inputs = load_file("dihedral_class_data.st")

    train_in = inputs["traininput"].type(torch.FloatTensor).cuda()
    print(train_in.shape[0])
    train_out = inputs["trainoutput"].type(torch.FloatTensor).cuda()

    test_incx = inputs["testinputcx"].type(torch.FloatTensor).cuda()
    ntestcx = test_incx.shape[0]
    print(ntestcx)
    test_inucx = inputs["testinputucx"].type(torch.FloatTensor).cuda()
    ntestucx = test_inucx.shape[0]
    print(ntestucx)

    # define loss and parameters
    optimizer = optim.NAdam(model.parameters())
    EPOCHS = 20_000

    print("====Training start====")
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    lossfn = nn.BCELoss()
    for i in range(EPOCHS):

        # set gradient to zero
        optimizer.zero_grad()

        # feed inputs into model
        pred_data = model.forward(train_in)

        loss = lossfn(pred_data, train_out)

        # calculate gradient of each parameter
        loss.backward()
        # print(loss)

        # update the weight based on the gradient calculated
        optimizer.step()

        model.eval()
        predcx = model.forward(test_incx)
        predcx = predcx.round()
        tp = (predcx == 1.0).sum()
        fn = ntestcx - tp
        acccx = tp / ntestcx

        preducx = model.forward(test_inucx)
        preducx = preducx.round()
        tn = (preducx == 0.0).sum()
        fp = ntestucx - tn
        accucx = tn / ntestucx
        # print(acc)
        # test_loss = lossfn( model.forward(test_in), test_out)
        model.train()

        if i % 10 == 0:
            print(
                "Epoch: {}  train loss: {:.9f}, cx test accuracy {:.8f}, ucx test accuracy {:.8f}, mcc {:.2}".format(
                    i,
                    loss,
                    100 * acccx,
                    100 * accucx,
                    (tn * tp - fp * fn)
                    / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5),
                )
            )
        if i % 100 == 0:
            scheduler1.step()

    print("====Training finish====")
    model.save_st("connection_pred_Weights.st")
