import torch
import torch.nn as nn
from collections import OrderedDict
from visdom import Visdom
import time
import numpy as np
import scipy.io as scio


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
            x = torch.tanh(self.layers[i](x)) #** 2
        x = self.layers[-1](x)
        return x


class Dfem(object):
    def __init__(self, layers=None, device=None):
        super(Dfem, self).__init__()
        if layers is None:
            layers = [2, 200, 200, 200, 2]  # 5
        #     layers = [3, 10000, 3]  # 6
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.u = DNN(layers).to(self.device)
        q = np.load('k.npy')
        idx = torch.tensor(q[:, 0:2].astype('long')).t() - 1
        value = torch.tensor(q[:, 2]).float() / 1.
        self.k = torch.sparse_coo_tensor(idx, value, size=(4186, 4186)).to(self.device)
        xy = scio.loadmat('xy.mat')['xy']
        xy = np.array(xy).astype(np.float32)
        self.xy = torch.tensor(xy).to(self.device)
        self.optimizer = torch.optim.LBFGS(self.u.parameters(), lr=0.1)

    def u_net(self, xy):
        v = self.u(xy)
        u = torch.zeros_like(v)
        u[:, 0:1] = v[:, 0:1] * (20 - xy[:, 1:2]) * (20 + xy[:, 1:2])
        u[:, 1:2] = v[:, 1:2] * (20 - xy[:, 1:2]) * (20 + xy[:, 1:2]) + xy[:, 1:2] / 40 - 0.5
        return u

    def loss_func(self, ):
        outputs = self.u_net(self.xy).reshape(-1, 1)
        energy = 0.5 * self.k.mm(outputs).t().mm(outputs)
        return energy

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
        file_1 = 'dfem_lowest3'
        file_2 = 'dfem_last3'
        self.u.train()
        optimizer = torch.optim.Adam(self.u.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5000,
                                                               verbose=True)
        for epoch in range(num_epoch):
            since = time.time()
            print('epoch {}/{}'.format(epoch, num_epoch - 1))
            # for batch in dataloader:
            # loss = 0
            with torch.no_grad():
                pd_loss = self.loss_func().item()

            if pd_loss < best_loss:
                best_loss = pd_loss
                state = {
                    'state_dict': self.u.state_dict().copy(),
                }
                # best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(state, file_1)

            if epoch < 0:
                optimizer.zero_grad()
                loss = self.loss_func()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
            else:
                self.optimizer.step(self.closures)

            if epoch == num_epoch - 1:
                state = {
                    'state_dict': self.u.state_dict().copy(),
                }
                torch.save(state, file_2)

            viz.line([[pd_loss]], [epoch], win='loss', update='append')
            viz.line([self.optimizer.param_groups[0]['lr']], [epoch], win='learning rate', update='append')
            time_elapsed = time.time() - since
            print('Time elapsed{:.0f}m {:.0f}s,'.format(time_elapsed // 60, time_elapsed % 60))
            print('loss:{:.6f}'.format(pd_loss))
            print('Optimizer learning rate : {:.7f}'.format(self.optimizer.param_groups[0]['lr']))
            print()

        time_elapsed = time.time() - since0
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best loss: {:4f}'.format(best_loss))

    def outputs(self):
        file = 'dfem_lowest2'
        weight = torch.load(file)['state_dict']
        self.u.load_state_dict(weight)
        xy = self.xy
        uv = self.u_net(xy)
        uv = uv.detach().cpu().numpy()
        np.save('nu.npy', uv)
        scio.savemat('u1.mat', {'u': uv})


def main():
    model = Dfem()
    # model.train(10000)
    model.outputs()



if __name__ == '__main__':
    main()

