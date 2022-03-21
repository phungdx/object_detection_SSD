from lib import *
from transform import DataTransform
from extract_inform_annotation import Anno_xml
from make_datapath import make_data_path_list
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class MyDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, anno_xml):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_xml = anno_xml

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, gt, height, width = self.pull_item(index)

        return img, gt

    def pull_item(self, index):
        img_file_path = self.img_list[index]
        img = cv2.imread(img_file_path) # BGR
        height, width, channels = img.shape

        # get annotation
        anno_file_path = self.anno_list[index]
        anno_info = self.anno_xml(anno_file_path, width, height)

        # preprocessing
        img, boxes, labels = self.transform(img, self.phase, anno_info[:,:4], anno_info[:,4])

        # BRG -> RGB and (height, width, channels) --> (channels, height, width)
        img = torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)

        # ground truth
        gt = gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width

def my_collate_fn(batch):
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))

    # batch_size, 3, 300, 300
    imgs = torch.stack(imgs, axis=0)
    return imgs, targets



if __name__ == '__main__':
    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    root_path = 'data\VOCdevkit\VOC2012'
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_data_path_list(root_path)

    color_mean = (104, 117, 123)
    input_size = 300

    train_dataset = MyDataset(train_img_list, train_annotation_list, phase='train',
        transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))

    val_dataset = MyDataset(val_img_list, val_annotation_list, phase='val',
        transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))

    batch_size = 4
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    val_data_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

    dataloader_dict = {
        'train': train_data_loader,
        'val': val_data_loader,
    }

    batch_iter = iter(dataloader_dict['train'])
    imgs, targets = next(batch_iter)
    print(targets[0])

    






