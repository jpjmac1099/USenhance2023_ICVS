import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        #opt.phase = os.path.join(opt.phase,'A')
        '''
        if self.opt.isTrain:
            folds = self.opt.training_fold
        else:
            folds = self.opt.testing_fold
        
        self.A_paths = []
        for i, fold in enumerate(folds):
            folds[i] = os.path.join('Fold_' + fold[0],'A')
            self.dir_A = os.path.join(opt.dataroot, folds[i])  # get the image directory
            A_paths = make_dataset(self.dir_A, opt.max_dataset_size)  # get image paths
            self.A_paths += A_paths
            assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
            self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
            self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
          
        self.A_paths = sorted(self.A_paths)
        '''
        
        opt.phase = os.path.join(opt.phase,'A')
        self.dir_A = os.path.join(opt.dataroot, opt.phase)  # get the image directory from A
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        
        #self.transform = get_transform(opt, convert=False)
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        # apply the same transform to both A and B
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)

        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)

        if self.opt.noRealImages == False or self.opt.isTrain:
            B_path = A_path.replace('/A/','/B/')
            B = Image.open(B_path).convert('RGB')
            #B_transform = get_transform(self.opt, grayscale=(self.output_nc == 1))
            B = B_transform(B)
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
         
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
