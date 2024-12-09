import torch
import torch.nn as nn
from collections import OrderedDict
from visdom import Visdom
import time
import numpy as np
import scipy.io as scio


# from torch.utils.data import Dataset, DataLoader


class DNN(nn.Module):
    def __init__(self, layer_param):
        super(DNN, self).__init__()
        self.depth = len(layer_param) - 1

        # set up layer order dict
        # self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layer_param[i], layer_param[i + 1]))
            )
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layer_param[-2], layer_param[-1]))
        )
        # print(layer_list)
        layerDict = OrderedDict(layer_list)
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        # self.relu = MyReLU.apply

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))  # ** 2
        x = self.layers[-1](x)
        return x


class Dfem(object):
    def __init__(self, layers=None, device=None):
        super(Dfem, self).__init__()
        if layers is None:
            layers = [3, 500, 500, 500, 3]  # 5
            # layers = [3, 10000, 3]  # 6
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.u = DNN(layers).to(self.device)
        q = np.load('k.npy')
        idx = torch.tensor(q[:, 0:2].astype('long')).t() - 1
        value = torch.tensor(q[:, 2]).float()
        self.k = torch.sparse_coo_tensor(idx, value, size=(30615, 30615)).to(self.device) / 1e6
        self.p = torch.tensor(np.load('p.npy')).float().to(self.device) / 1e4
        self.xy = torch.tensor(np.load('xy.npy')).float().to(self.device) / 1000
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.LBFGS(self.u.parameters(), lr=0.5)

    def u_net(self, x):
        v = self.u(x)
        return v

    def loss_func(self, ):
        inputs = self.xy
        outputs = self.u_net(inputs).reshape(-1, 1)
        loss = 0.5 * self.k.mm(outputs).t().mm(outputs) - self.p.t().mm(outputs)
        return loss

    def closures(self, ):
        self.optimizer.zero_grad()
        loss = self.loss_func()
        loss.backward()
        return loss

    def train(self, num_epoch):
        viz = Visdom()
        viz.line([[0.0]], [0.], win='loss',
                 opts=dict(title='loss', legend=['loss', 'cost']))
        viz.line([0.0], [0.], win='learning rate',
                 opts=dict(title='lr', legend=['lr']))
        since0 = time.time()
        best_loss = 1e32
        file_1 = 'dfem_lowest1'  # 模型权重参数的保存名
        file_2 = 'dfem_last1'  # 模型权重参数的保存名
        self.u.train()
        optimizer = torch.optim.SGD(self.u.parameters(), momentum=0.7, lr=1e-4, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5000,
                                                               verbose=True)
        for epoch in range(num_epoch):
            if epoch == 100:
                optimizer = torch.optim.SGD(self.u.parameters(), momentum=0.7, lr=1e-3, weight_decay=0.0)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=500,
                                                                       verbose=True)
                # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
            # pd_loss = 0
            since = time.time()
            print('epoch {}/{}'.format(epoch, num_epoch - 1))
            # for batch in dataloader:
            optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            optimizer.step()

            pd_loss = loss.item()
            # self.scheduler.step(pd_loss)
            state = {
                'state_dict': self.u.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, file_2)
            if pd_loss < best_loss:
                best_loss = pd_loss
                # best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(state, file_1)
            viz.line([[pd_loss]], [epoch], win='loss', update='append')
            viz.line([optimizer.param_groups[0]['lr']], [epoch], win='learning rate', update='append')
            scheduler.step(pd_loss)
            # self.scheduler.step()
            time_elapsed = time.time() - since
            print('Time elapsed{:.0f}m {:.0f}s,'.format(time_elapsed // 60, time_elapsed % 60))
            print('loss:{:.6f}'.format(pd_loss))
            print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
            print()

        time_elapsed = time.time() - since0
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best loss: {:4f}'.format(best_loss))

    def outputs(self):
        file = 'dfem_lowest11'
        weight = torch.load(file)['state_dict']
        self.u.load_state_dict(weight)
        xy = self.xy
        uv = self.u_net(xy)
        loss = 0.5 * self.k.mm(uv.reshape(-1, 1)).t().mm(uv.reshape(-1, 1)) - self.p.t().mm(uv.reshape(-1, 1))
        print(loss)
        uv = uv.detach().cpu().numpy()
        np.save('nu.npy', uv)
        scio.savemat('u.mat', {'u': uv})


def main():
    model = Dfem()
    # model.train(10000)
    model.outputs()


if __name__ == '__main__':
    main()
    # k = torch.randn([8216, 8216])
    #
    # dataset = MyDataset()
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    # indx = next(iter(dataloader))
    # print(indx, indx.shape)
    # print(k[indx], k[indx[0], :])
