from lib import *
from l2_norm import L2Norm
from default_box import DefBox

def create_vgg():
    layers = []
    in_channels = 3

    cfgs = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'MC',
            512, 512, 512, 'M', 512, 512, 512]

    for cfg in cfgs:
        if cfg == 'M': # floor
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif cfg == 'MC': # ceiling
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=cfg, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = cfg

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)


def create_extras():
    layers = []
    in_channels = 1024

    cfgs = [256, 512, 128, 256, 128, 256, 128, 256]
    layers += [nn.Conv2d(in_channels, cfgs[0], kernel_size=1, padding=0)] # out: 38x38
    layers += [nn.Conv2d(cfgs[0], cfgs[1], kernel_size=3, stride=2, padding=1)] # out: 19x19
    layers += [nn.Conv2d(cfgs[1], cfgs[2], kernel_size=1, padding=0)]
    layers += [nn.Conv2d(cfgs[2], cfgs[3], kernel_size=3, stride=2, padding=1)] # out: 10x10
    layers += [nn.Conv2d(cfgs[3], cfgs[4], kernel_size=1, padding=0)]
    layers += [nn.Conv2d(cfgs[4], cfgs[5], kernel_size=3, padding=0)] # out: 8x8
    layers += [nn.Conv2d(cfgs[5], cfgs[6], kernel_size=1, padding=0)]
    layers += [nn.Conv2d(cfgs[6], cfgs[7], kernel_size=3, padding=0)] # out: 6x6

    return nn.ModuleList(layers)


def create_loc_conf(num_classes=21, bbox_ratio_num=[4,6,6,6,4,4]):
    loc_layers = []
    conf_layers = []

    # source 1
    # loc
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[0]*4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[0]*num_classes, kernel_size=3, padding=1)]

    # source 2
    # loc
    loc_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*num_classes, kernel_size=3, padding=1)]

    # source 3
    # loc
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[2]*4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[2]*num_classes, kernel_size=3, padding=1)]

    # source 4
    # loc
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[3]*4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[3]*num_classes, kernel_size=3, padding=1)]

    # source 5
    # loc
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[4]*4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[4]*num_classes, kernel_size=3, padding=1)]

    # source 6
    # loc
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[5]*4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[5]*num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


cfg = {
    "num_classes": 21,
    "input_size": 300,
    "bbox_aspect_num": [4,6,6,6,4,4], # tỉ lệ khung hình
    "feature_maps": [38,19,10,5,3,1],
    "steps": [8,16,32,64,100,300], # size of default boxes
    "min_size": [30, 60, 111, 162, 213, 264], # size of default boxes
    "max_size": [60, 111, 162, 213, 264, 315], # size of default boxes
    "aspect_ratios": [[2], [2,3], [2,3], [2,3], [2], [2]]
}


class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']

        # create main modules
        self.vgg = create_vgg()
        self.extras = create_extras()
        self.loc, self.conf = create_loc_conf(cfg['num_classes'], cfg['bbox_aspect_num'])
        self.l2norm = L2Norm()

        # default box
        dbox = DefBox(cfg)
        self.dbox_list = dbox.create_defbox()

        if self.phase == 'inference':
            self.detect = Detect()


    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        for k in range(23):
            x = self.vgg[k](x)

        # source 1
        source1 = self.l2norm(x)
        sources.append(source1)

        for k in range(23,len(self.vgg)):
            x = self.vgg[k](x)

        # source 2
        sources.append(x)

        # source 3 --> source 6
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # aspect_ratio_num = 4, 6
            # (batch_size, 4*aspect_ratio_num, featuremap_height, featuremap_width)
            # -> (batch_size, featuremap_height, featuremap_width ,4*aspect_ratio_num)

            loc.append(l(x).permute(0,2,3,1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0),-1)for o in loc],dim=1) # (batch size, 34928) 4*8732
        conf = torch.cat([o.view(o.size(0),-1)for o in conf],dim=1) # (batch size, 183372) 21*8732

        loc = loc.view(loc.size(0), -1, 4) # (batch_size, 8732, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes) # (batch_size, 8732, 21)

        output = (loc, conf, self.dbox_list)

        if self.phase=='inference':
            with torch.no_grad():
                return self.detect(loc, conf, self.dbox_list)

        else:
            return output


def decode(loc, defbox_list):
    """
    parameters:
    loc: [8732, 4] (delta_x, delta_y, delta_w, delta_h)
    defbox_list: [8732, 4] (cx_d, cy_d, w_d, h_d)

    returns:
    boxes: [xmin, ymin, xmax, ymax]
    """

    boxes = torch.cat(
        (defbox_list[:,:2]) + 0.1*loc[:,:2] * defbox_list[:,:2],
        defbox_list[:,2:] * torch.exp(loc[:,2:]*0.2), dim=1)

    boxes[:,:2] -= 1/2 * boxes[:,2:]
    boxes[:,2:] = boxes[:,:2] + boxes[:,2:]

    return boxes

# Non-Maximum Suppression
def NMS(boxes, scores, overlap=0.45, top_k=200):
    """
    boxes: [num_box, 4]
    scores: [num_box]

    """
    count = 0
    keep = scores.new(scores.size()).zero_().long()

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # area of boxes
    area = torch.mul(x2-x1, y2-y1)

    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    value, idx = scores.sort(dim=0)
    idx = idx[-topk:] # id của top 200 boxes có độ tự tin cao nhất

    while idx.numel() > 0:
        i = idx[-1] # id của box có độ tự tin cao nhất
        keep[count] = i
        count += 1

        if idx.size(0)==1:
            break

        idx = idx[:-1] # id của boxes ngoại trừ box có độ tự tin cao nhất

        # information boxes
        torch.index_select(input=x1, dim=0, index=idx, out=tmp_x1)
        torch.index_select(input=y1, dim=0, index=idx, out=tmp_y1)
        torch.index_select(input=x2, dim=0, index=idx, out=tmp_x2)
        torch.index_select(input=y2, dim=0, index=idx, out=tmp_y2)
        
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, min=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, min=y2[i])

        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # area
        inter = tmp_w * tmp_h
        others_area = torch.index_select(area, 0, idx) # diện tích của mỗi bbox
        union = area[i] + others_area - inter
        iou = inter/union
        idx = idx[iou.le(overlap)] # giữ lại id của box có overlap ít với bbox đang xét

    return keep, count


class Detect(Function):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh

    def forward(self, loc_data, conf_data, dbox_list):
        """
        loc_data: (batch_size, 8732, 4)
        conf_data: (bactch_size, 8732, 21)
        """
        num_batch = loc_data.size(0) # batch size
        num_dbox = loc_data.size(1) # 8732
        num_classes = conf_data.size(2) # 21

        conf_data = self.softmax(conf_data)
        # (batch_num, num_dbox, num_class) ---> (batch num, num_class, num_dbox)
        conf_preds = conf_data.transpose(2,1)

        output = torch.zeros(num_batch, num_classes, self.top_k, 5)
        # xử lý từng ảnh 1
        for i in range(num_batch):
            # Tính bboc từ offset information và default boxes
            decode_boxes = decode(loc_data[i], defbox_list) # (bacth_size x 8732 x 4)

            # copy confidence score cua anh thu i
            conf_scores = conf_preds[i].clone()

            for category in range(1, num_classes):
                # chỉ lấy những confidence > 0.01
                category_mask = conf_scores[category].gt(self.conf_thresh)
                scores = conf_scores[category][category_mask]

                if scores.nelement() == 0:
                    continue
                
                # đưa chiều về giống chiều của decode_boxes để tính toán
                l_mask = category_mask.unsqueeze(1).expand_as(default_boxes) # (8732,4)
                boxes = decode_boxes[l_mask].view(-1,4)

                # count: số lượng objects của 1 class có trong ảnh
                ids, count = nms(boxes, scores, overlap=self.nms_thresh, top_k=self.top_k)

                output[i, category, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        return output
                
                
if __name__ == '__main__':
    # vgg = create_vgg()
    # print(vgg)
    # extras = create_extras()
    # print(extras)

    # loc, conf = create_loc_conf()
    # print(loc)
    # print(conf)

    ssd = SSD(phase='train', cfg=cfg)
    print(ssd)




