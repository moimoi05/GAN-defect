"""Deterministic index coverage for every clean and defect manifest row in a pilot epoch."""
from torch.utils.data import Dataset

class FullPatchCoverageDataset(Dataset):
    def __init__(self, base):
        if not base.normal_items or not base.defect_items:
            raise ValueError('Both domains must contain patches')
        self.base = base
        self.length = len(base.normal_items) + len(base.defect_items)
        self.clean_count = len(base.normal_items)
        self.defect_count = len(base.defect_items)
        self.step = next((step for step in range(2, self.defect_count+1)
                          if __import__('math').gcd(step, self.defect_count) == 1), 1)

    def __len__(self): return self.length

    def __getitem__(self, index):
        ni = index % self.clean_count
        di = (index * self.step + index // self.defect_count) % self.defect_count
        normal_item, defect_item = self.base.normal_items[ni], self.base.defect_items[di]
        return {
            'normal': self.base._load_image(normal_item['full_path']),
            'defect': self.base._load_image(defect_item['full_path']),
            'normal_path': normal_item['patch_path'], 'defect_path': defect_item['patch_path'],
            'normal_source': normal_item['source_image'], 'defect_source': defect_item['source_image'],
        }
