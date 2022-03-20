from utils.augmentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, \
        Expand, RandomSampleCrop, RandomMirror, \
        ToPercentCoords, Resize, SubtractMeans

from extract_inform_annotation import Anno_xml
from make_datapath import make_data_path_list
from lib import *

class DataTransform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train": Compose([
                ConvertFromInts(), # convert image from int to float
                ToAbsoluteCoords(), # back annotation to normal type
                PhotometricDistort(), # change color by random
                Expand(color_mean),
                RandomSampleCrop(), # randomly crop image
                RandomMirror(), # reverse image
                ToPercentCoords(),
                Resize(input_size),
                SubtractMeans(color_mean),        
                ]),

            "val": Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


if __name__ == '__main__':
    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    root_path = 'data\VOCdevkit\VOC2012'
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_data_path_list(root_path)

    # read img
    img_file_path = train_img_list[0]
    img = cv2.imread(img_file_path)
    height, width, channels = img.shape

    # annotation information
    trans_anno = Anno_xml(classes)
    anno_info_list = trans_anno(train_annotation_list[0], width, height)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    color_mean = (104, 117, 123)
    input_size = 300
    transform = DataTransform(input_size, color_mean)

    # transform img
    phase = 'train'
    img_transformed, boxes, labels = transform(img, phase, anno_info_list[:, :4], anno_info_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB).astype(np.uint8))
    plt.show()

    # transform val img
    phase = 'val'
    img_transformed, boxes, labels = transform(img, phase, anno_info_list[:, :4], anno_info_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB).astype(np.uint8))
    plt.show()
