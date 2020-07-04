import os
from PIL import Image
from torch import tensor, is_tensor
from torchvision import transforms
from collections import defaultdict
from torch.nn.functional import pad
from torch.utils.data import Dataset


def collate_fn(_batch):
    """

    :param _batch:
    :return:
    """
    _batch = {k: [item[k] for item in _batch] for k in _batch[0]}
    batch_boxes, batch_images = _batch['boxes'], _batch['images']
    pad_to_len = max([len(boxes) for boxes in batch_boxes])
    for i, boxes in enumerate(batch_boxes):
        batch_boxes[i] = pad(
            input=tensor(boxes),
            pad=[0, 0, 0, pad_to_len - len(boxes)],
            mode='constant',
            value=0
        ).tolist()
    if not is_tensor(batch_images[0]):
        batch_images = [transforms.ToTensor()(image) for image in batch_images]
    return {
        'images': batch_images,
        'boxes': tensor(batch_boxes)
    }


class WiderDataset(Dataset):
    def __init__(self, file_path, input_dir, _transforms=None):
        """
        :param file_path: path to .txt file with input boxes
        :param input_dir: path to directory with input images
        :param _transforms: torchvision transform of images
        """
        super(WiderDataset, self).__init__()
        self.file_path = file_path
        self.input_dir = input_dir
        self._transforms = _transforms
        self.dataset = self.process_file()

    def process_image(self, image_path):
        """
        Open PIL image and make transforms
        :param image_path: path to input image
        :return: tensor image
        """
        image_path = os.path.join(self.input_dir, image_path)
        image = Image.open(image_path, mode='r')
        if self._transforms is not None:
            image = self._transforms(image)
        else:
            image = transforms.Resize(416)(image)
        return image

    def process_file(self):
        """
        Parse input .txt file
        :return:
        """
        _dataset = defaultdict(list)
        with open(self.file_path, mode='r') as txt_file:
            image_path = next(txt_file, '').rstrip('\n')
            while image_path:
                n = int(next(txt_file, 0))
                boxes = []
                for _ in range(n):
                    x, y, w, h, _, _, _, _, _, _ = map(int, next(txt_file, '').split())
                    box = [x, y, w, h]
                    boxes.append(box)
                if not n:
                    boxes.append([0, 0, 0, 0])
                    next(txt_file)
                _dataset['images_path'].append(image_path)
                _dataset['boxes'].append(boxes)
                image_path = next(txt_file, '').rstrip('\n')
        return _dataset

    def __len__(self):
        return len(self.dataset['images_path'])

    def __getitem__(self, idx):
        image_path, boxes = self.dataset['images_path'][idx], self.dataset['boxes'][idx]
        image = self.process_image(image_path)
        return {'images': image, 'boxes': boxes}
