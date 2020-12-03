import torch
import torchvision
import torchvision.transforms as T

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize

from unet import UNET

trf = T.Compose([T.Resize((176,320)),
                 T.ToTensor()])

##### DATALOADER #####
from torch.utils.data import Dataset, DataLoader, sampler
from pathlib import Path

class BDD100K(Dataset):
  def __init__(self,img_dir,gt_dir,pytorch=True):
    super().__init__()
    # Loop through the files in red folder and combine, into a dictionary, the other bands
    self.files = [self.combine_files(f, gt_dir) for f in img_dir.iterdir() if not f.is_dir()]
    self.pytorch = pytorch
      
  def combine_files(self, img_file: Path, gt_dir):
    files = {'img': img_file,
             'gt': Path(str(gt_dir/img_file.name).split('.')[0] + '_drivable_color.png')}
    return files
                                     
  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
    x = trf(Image.open(self.files[index]['img']))

    im = np.array(Image.open(self.files[index]['gt']))
    mask = np.int64(np.all(im == [255,0,0], axis=2))

    y = torch.tensor(resize(mask, (176, 320)), dtype=torch.long)
    #y = trf(Image.fromarray((mask * 255).astype(np.uint8)))
    return x,y


train_ds = BDD100K( Path('./bdd100k/images/100k/train/'),
                    Path('./bdd100k/drivable_maps/color_labels/train/'))
valid_ds = BDD100K( Path('./bdd100k/images/100k/val/'),
                    Path('./bdd100k/drivable_maps/color_labels/val/'))

train_dl = DataLoader(train_ds, batch_size=12, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=12, shuffle=True)

# xb, yb = next(iter(train_dl))
# print(xb.shape,yb.shape)
# trans = T.ToPILImage(mode='RGB')
x,y = train_ds[100]
print(x.shape,y.shape)
# plt.imshow(trans(x.squeeze()))
# plt.show()
# plt.imshow(trans(y.squeeze()))
# plt.show()
######################


# img = Image.open('./bdd100k/images/100k/train/0000f77c-62c2a288.jpg')

# inp = trf(img).unsqueeze(0)

# #plt.imshow(img); plt.show()

# print(inp.shape)

# unet = UNET(3,3)
# pred = unet(inp)
# print(pred.shape)

import time
def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()
    model.cuda()

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = x.cuda()
                y = y.cuda()
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 

                if step % 10 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss    

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

######TRAINING##########
unet = UNET(3,3)
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(unet.parameters(),lr=0.01)
train_loss, valid_loss = train(unet, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=50)