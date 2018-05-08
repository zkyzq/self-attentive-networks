import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from data import Omniglot
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import transformer
import time
import numpy as np
from optim import ScheduledOptim
from torch import nn

# import pdb


cuda = torch.cuda.is_available()

# super parameters
img_size = 28
way = 10
d_model = 64
d_k = 32
h = 2
N = 2
drop_rate = 0.1
learning_rate = 0.001
# weight_decay=0.0001
# warmup_steps = 1000

# training parameters
train_path = 'background_augment'
test_path = 'evaluation'
show_every = 10
save_every = 100
test_every = 10
train_loss = []
pred_precision = []
loss_val = 0
max_iter = 10000
num_test_batch = 20


data_transforms = transforms.Compose([
    transforms.RandomAffine(15),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

train_dataset = dset.ImageFolder(root=train_path)
test_dataset = dset.ImageFolder(root=test_path)


dataSet = Omniglot(train_dataset, transform=data_transforms, way=way)
testSet = Omniglot(test_dataset, transform=test_transforms, way=way)

testLoader = DataLoader(testSet, batch_size=32, shuffle=False, num_workers=16)
dataLoader = DataLoader(dataSet, batch_size=128,\
                        shuffle=False, num_workers=16)

loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
# loss_fn = nn.DataParallel(loss_fn)
loss_fn.cuda()

net = transformer(way, img_size, N, d_model, d_k, h, drop_rate)

# net = nn.DataParallel(net)
net.cuda()
net.train()

train_loss = []
optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate, betas=(0.9, 0.98), eps=1e-9)
# optimizer = ScheduledOptim(optimizer, d_model, warmup_steps)
optimizer.zero_grad()

def right_error(output, truth):
    output = output.data.cpu().numpy()
    output = np.argmax(output, axis=1)
    truth = np.squeeze(truth.data.cpu().numpy())
    right, error = 0, 0
    for x, y in zip(output, truth):
        if x == y:
            right += 1
        else: error += 1
    return right, error


train_right, train_error = 0, 0

for batch_id, (imgs, y) in enumerate(dataLoader, 1):
    if batch_id > max_iter:
        break
    batch_start = time.time()
    if cuda:
        imgs, y = [Variable(img.cuda()) for img in imgs], Variable(y.cuda())
    else:
        imgs, y = [Variable(img) for img in imgs], Variable(y)
    optimizer.zero_grad()

    # pdb.set_trace()

    output = net.forward(imgs)
    y = torch.squeeze(y)
    loss = loss_fn(output, y)
    loss_val += loss.data[0]
    train_loss.append(loss_val)
    loss.backward()
    optimizer.step()
    r, e = right_error(output, y)
    train_right += r
    train_error += e
    if batch_id % show_every == 0 :
        prec = (train_right * 1.0) / (train_right + train_error)
        print('[%d] train:\tloss:\t%.5f\tright:%d\terror:%d\tprecision:\t%.5f\tTook\t%.2f s'%(batch_id, loss_val/show_every, train_right, train_error, prec, (time.time() - batch_start)*show_every))
        loss_val = 0
        train_error, train_right = 0, 0


    if batch_id % save_every == 0:
        torch.save(net.state_dict(), './model/model-batch-%d.pth'%(batch_id,))

    if batch_id % test_every == 0:
        net.eval()
        right, error = 0, 0
        for test_id, (test_imgs, test_y) in enumerate(testLoader, 1):
            if cuda:
                test_imgs, test_y = [Variable(img.cuda()) for img in test_imgs], Variable(test_y.cuda())
            else:
                test_imgs, test_y = [Variable(img) for img in test_imgs], Variable(test_y)
            output = net.forward(test_imgs)
            r, e = right_error(output, test_y)
            right += r
            error += e
            if test_id == num_test_batch:
                break

        precision = right * 1.0 / (right + error)
        pred_precision.append(precision)
        print('*'*70)
        print('[%d] test:\tright:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, precision))
        print('*'*70)
        net.train()
#  learning_rate = learning_rate * 0.95

with open('train_loss', 'wb') as f:
    pickle.dump(train_loss, f)


with open('pred_precision', 'wb') as f:
    pickle.dump(pred_precision, f)
