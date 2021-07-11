import torch
import iou3d_cuda
import lib.utils.kitti_utils as kitti_utils


def boxes_iou_bev(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 5)
    :param boxes_b: (N, 5)
    :return:
        ans_iou: (M, N)
    """

    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = kitti_utils.boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = kitti_utils.boxes3d_to_bev_torch(boxes_b)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

    # height overlap
    boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, 3]).view(-1, 1)
    boxes_a_height_max = boxes_a[:, 1].view(-1, 1)
    boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, 3]).view(1, -1)
    boxes_b_height_max = boxes_b[:, 1].view(1, -1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    union = torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)
    #iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

    return overlaps_3d, union

def boxes_iou3d_vec_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (N, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (N)
    """
    boxes_a_bev = kitti_utils.boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = kitti_utils.boxes3d_to_bev_torch(boxes_b)

    # bev overlap
    #overlaps_bev = torch.cuda.FloatTensor(torch.Size((1, boxes_a.shape[0]))).zero_()  # (N, M)
    overlaps_bev = torch.cuda.FloatTensor(boxes_a.shape[0]).zero_()  # (N)
    convex_hull_bev = torch.cuda.FloatTensor(boxes_a.shape[0]).zero_()  # (N)
    iou3d_cuda.boxes_overlap_bev_vec_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev, convex_hull_bev)

    # height overlap
    boxes_a_height_min = boxes_a[:, 1] - boxes_a[:, 3]
    boxes_a_height_max = boxes_a[:, 1]
    boxes_b_height_min = boxes_b[:, 1] - boxes_b[:, 3]
    boxes_b_height_max = boxes_b[:, 1]

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    min_of_h = torch.min(boxes_a_height_min, boxes_b_height_min)
    max_of_h = torch.max(boxes_a_height_max, boxes_b_height_max)
    convex_hull_h = torch.clamp(max_of_h - min_of_h, min=0)

    # 3d iou
    overlaps_3d = torch.clamp(overlaps_bev * overlaps_h, min=0)
    convex_hull_3d = convex_hull_bev * convex_hull_h

    vol_a = boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]
    vol_b = boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]

    union = torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)
    convex_hull_3d = torch.clamp(convex_hull_3d, min=1e-7)
    #iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)
    #print('vol_a:', vol_a)
    #print('vol_b:', vol_b)
    #print('overlaps_bev:', overlaps_bev)
    #print('overlaps_h:', overlaps_h)

    return overlaps_3d, union, convex_hull_3d

def Areac(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (N, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_area: (N)
    """
    x_a = boxes_a[:, 0]
    y_a = boxes_a[:, 1]
    z_a = boxes_a[:, 2]
    h_a = boxes_a[:, 3]
    w_a = boxes_a[:, 4]
    l_a = boxes_a[:, 5]

    x_b = boxes_b[:, 0]
    y_b = boxes_b[:, 1]
    z_b = boxes_b[:, 2]
    h_b = boxes_b[:, 3]
    w_b = boxes_b[:, 4]
    l_b = boxes_b[:, 5]

    max_x  = torch.max(x_a+w_a/2, x_b+w_b/2)
    min_x  = torch.min(x_a-w_a/2, x_b-w_b/2)
    max_y  = torch.max(y_a+h_a/2, y_b+h_b/2)
    min_y  = torch.min(y_a-h_a/2, y_b-h_b/2)
    max_z  = torch.max(z_a+l_a/2, z_b+l_b/2)
    min_z  = torch.min(z_a-l_a/2, z_b-l_b/2)
    #print(max_x)
    #print(min_x)
    #print(max_y)
    #print(min_y)
    #print(max_z)
    #print(min_z)

    return torch.clamp((max_x-min_x)*(max_y-min_y)*(max_z-min_z), min=1e-7)

def overlap(boxes_a, boxes_b):
    x_a = boxes_a[:, 0]
    y_a = boxes_a[:, 1]
    z_a = boxes_a[:, 2]
    h_a = boxes_a[:, 3]
    w_a = boxes_a[:, 4]
    l_a = boxes_a[:, 5]

    x_b = boxes_b[:, 0]
    y_b = boxes_b[:, 1]
    z_b = boxes_b[:, 2]
    h_b = boxes_b[:, 3]
    w_b = boxes_b[:, 4]
    l_b = boxes_b[:, 5]

    min_of_max_x  = torch.min(x_a+w_a/2, x_b+w_b/2)
    max_of_min_x  = torch.max(x_a-w_a/2, x_b-w_b/2)

    min_of_max_y  = torch.min(y_a+h_a/2, y_b+h_b/2)
    max_of_min_y  = torch.max(y_a-h_a/2, y_b-h_b/2)

    min_of_max_z  = torch.min(z_a+l_a/2, z_b+l_b/2)
    max_of_min_z  = torch.max(z_a-l_a/2, z_b-l_b/2)

    cube_overlap = (min_of_max_x - max_of_min_x)*(min_of_max_y - max_of_min_y)*(min_of_max_z - max_of_min_z)
    return torch.clamp(cube_overlap, min=0)

def Giou_3d(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (N, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_area: (N)
    """

    area_c = Areac(boxes_a, boxes_b)
    area_o = overlap(boxes_a, boxes_b)
    area_a = boxes_a[:, 3]*boxes_a[:, 4]*boxes_a[:, 5]
    area_b = boxes_b[:, 3]*boxes_b[:, 4]*boxes_b[:, 5]
    area_u = (area_a+area_b - area_o)
    iou_3d = area_o / area_u
    print('area_c:', area_c)
    print('area_o:', area_o)
    print('area_a:', area_a)
    print('area_b:', area_b)
    print('area_u:', area_u)
    print('iou_3d:', iou_3d)
    giou_3d = iou_3d - (area_c - area_u)/area_c
    
    return giou_3d


def nms_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()


def nms_normal_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()


if __name__ == '__main__':
    pass
