import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

def inputs_files_to_output_files(input_files):
    return [
        Path(str(f).replace('seis', 'vel').replace('data', 'model'))
        for f in input_files
    ]

def get_train_files(data_path):
    all_inputs = [
        f
        for f in
        Path(data_path).rglob('*.npy')
        if ('seis' in f.stem) or ('data' in f.stem)
    ]
    all_outputs = inputs_files_to_output_files(all_inputs)
    
    assert all(f.exists() for f in all_outputs)
    
    return all_inputs, all_outputs

class SeismicDataset(Dataset):
    def __init__(self, inputs_files, output_files, n_examples_per_file=500):
        assert len(inputs_files) == len(output_files)
        self.inputs_files = inputs_files
        self.output_files = output_files
        self.n_examples_per_file = n_examples_per_file

    def __len__(self):
        return len(self.inputs_files) * self.n_examples_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.n_examples_per_file
        sample_idx = idx % self.n_examples_per_file

        X = np.load(self.inputs_files[file_idx], mmap_mode='r')
        y = np.load(self.output_files[file_idx], mmap_mode='r')

        try:
            return X[sample_idx].copy(), y[sample_idx].copy()
        finally:
            del X, y

class TestDataset(Dataset):
    def __init__(self, test_files):
        self.test_files = test_files

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, i):
        test_file = self.test_files[i]
        return np.load(test_file), test_file.stem
