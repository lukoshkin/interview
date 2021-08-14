import cv2

import torch
import torch.utils.data as torch_data

class BatchLoader:
    """
    Expects torch.Tensor-s as input
    """
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self._reinit()
        
    def __len__(self):
        return len(self.y) // self.batch_size

    def _gen(self):
        bs = self.batch_size  # alias
        ids = torch.randperm(len(self.y))

        for i in range(0, len(self.y), bs):
            yield self.X[i:i+bs], self.y[i:i+bs]
            
        self._reinit()

    def _reinit(self):
        self.gen = self._gen()
        
    def __iter__(self):
        return self.gen


class MRLEyesData(torch_data.Dataset):
    def __init__(self, root_dir=None, fnames=None):
        super().__init__()
        if root_dir is not None:
            self.fnames = list(Path(root_dir).rglob('*.png'))
        elif fnames is not None:
            self.fnames = fnames
        else:
            raise ValueError(
                'root_dir or fnames '
                'should be specified')

        self.targets = []
        for fname in self.fnames:
            self.targets.append(float(fname.stem.split('_')[4]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        target = self.targets[idx]
        
        img = cv2.imread(str(fname), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (24, 24))
        img = img[None].astype('f4') / 255
        return img, target
