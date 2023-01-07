from torch.utils.data import Dataset
import os
from img2vec_pytorch import Img2Vec
from PIL import Image
import torch

class VecDataset(Dataset):
    def __init__(self, device='cuda', dir='data', model='resnet-18', cuda=True):
        self.dir = dir
        self.model = model
        self.cuda = cuda

        self.img2vec = Img2Vec(cuda=self.cuda)

        self.vectors = []
        self.targets = []
        for image in os.listdir(self.dir):
            img_path = os.path.join(self.dir, image)

            img = Image.open(img_path)

            img = img.resize((64, 64))
            limg = self.img2list(img)

            self.targets.append(torch.Tensor(limg).to(device))
            self.vectors.append(torch.from_numpy(self.img2vec.get_vec(img)).to(device))

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        return self.vectors[index], self.targets[index]
    
    @staticmethod
    def img2list(pilimg):
        pixels = pilimg.load()
        w, h = pilimg.size
        res = []

        for y in range(w):
            line = []
            for x in range(h):
                r, g, b = pixels[x, y][0] / 255, pixels[x, y][1] / 255, pixels[x, y][2] / 255
                line.append((r, g, b))
            res.append(line)

        return res
