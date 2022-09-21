from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio


class Imgdataset(Dataset):

    def __init__(self, path):
        super(Imgdataset, self).__init__()
        self.data = []
        if os.path.exists(path):
            gt_path = path

            if os.path.exists(gt_path):
                gt = os.listdir(gt_path)
                self.data = [{'orig': gt_path + '/' + gt[i]} for i in range(len(gt))]
            else:
                raise FileNotFoundError('path doesnt exist!')
        else:
            raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index):

         data = self.data[index]["orig"]
         data = scio.loadmat(data)
         gt = data['Zgt']
         gt = torch.squeeze(torch.from_numpy(gt))         
         meas = data['x']
         meas = torch.squeeze(torch.from_numpy(meas))

         return gt, meas

    def __len__(self):

        return len(self.data)
