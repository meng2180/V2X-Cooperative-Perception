

import os

import numpy as np
import torch
import math
from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils
from shapely.geometry import Polygon
from datetime import datetime


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre



def caluclate_tp_fp_10s(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    score = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)
        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low

        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box, in confidence descending order
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue
            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)
        score = det_score.tolist()
        result_stat[iou_thresh]['score'] += det_score.tolist()
    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt
    return tp, fp, gt, score


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.
    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    
    fp_false_detection = []
    fp_localization = []
    object_missing = []

    
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)
        

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))
        
        
        
        matched_gt_indices = set()
        
        # 1. object missing
        for gt_polygon in gt_polygon_list:

            
            ious = common_utils.compute_iou(gt_polygon, det_polygon_list)
            max_iou = np.max(ious) if ious.size > 0 else 0.0
            if  max_iou <= 0:
                object_missing.append(1)
            else:
                object_missing.append(0)
                
                
        # match prediction and gt bounding box, in confidence descending order
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)
            max_iou = np.max(ious) if ious.size > 0 else 0.0
            

            if len(gt_polygon_list) == 0 or max_iou <= iou_thresh:
                fp.append(1)
                tp.append(0)
                # 2. false detection
                if max_iou <= 0 :
                    fp_false_detection.append(1)
                else:
                    fp_false_detection.append(0)
                continue

            # successful detection
            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            matched_gt_indices.add(gt_index)
            gt_polygon_list.pop(gt_index) #pop
            
            
        # Localization failure
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            for gt_index, gt_polygon in enumerate(gt_polygon_list):
                iou = det_polygon.intersection(gt_polygon).area / det_polygon.union(gt_polygon).area
                
                
                if 0 < iou <= iou_thresh:
                    fp_localization.append(1)
#                     gt_polygon_list.pop(gt_index)
                    break
    
            
            

        
        result_stat[iou_thresh]['score'] += det_score.tolist()
    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt
    result_stat[iou_thresh]['fp_false_detection'] += fp_false_detection
    result_stat[iou_thresh]['object_missing'] += object_missing
    result_stat[iou_thresh]['fp_localization'] += fp_localization
    
    

#     print(f"FP: {len(fp)}, TP: {sum(tp)}, FN: {sum(fn)}")
    
    print(f"GT: {len(gt_boxes)} Object Missing: {sum(object_missing)}, False Detection: {sum(fp_false_detection)}. Localization Failures:{sum(fp_localization)}")


def caluclate_tp_fp_v2(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):

    # fp, tp and gt in the current frame
    fp = []
    tp = []
#     gt = gt_boxes.shape[0]
    gt = len(gt_boxes)
    print()
    if det_boxes is not None:
        # convert bounding boxes to numpy array
#         det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
#         det_score = common_utils.torch_tensor_to_numpy(det_score)
#         gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list((det_boxes))
        gt_polygon_list = list((gt_boxes))
        print("score_order_descend shape:", score_order_descend.shape)  # e.g. (N,)
        print("det_score shape:", det_score.shape)                      # e.g. (N,)
        print("det_polygon_list length:", len(det_polygon_list))        # e.g. N
        print("gt_polygon_list length:", len(gt_polygon_list))          # e.g. M

        # match prediction and gt bounding box, in confidence descending order
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)
        result_stat[iou_thresh]['score'] += det_score.tolist()
    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt
    
    print(f"FP:{fp},   TP:{tp},   GT:{gt}")
    

def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.
    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = np.array(iou_5['fp'])
    tp = np.array(iou_5['tp'])
    score = np.array(iou_5['score'])
    assert len(fp) == len(tp)  and len(tp) == len(score)

    sorted_index = np.argsort(-score)
    fp = fp[sorted_index].tolist()
    tp = tp[sorted_index].tolist()

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec



def eval_final_results(result_stat, save_path, infer_info=None):
    dump_dict = {}

#     ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
#     ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    
    
#     for iou_thresh in [0.30, 0.50, 0.70]:
    for iou_thresh in [0.50]:
    
        fp_total = np.sum(result_stat[iou_thresh]['fp'])
        tp_total = np.sum(result_stat[iou_thresh]['tp'])
        gt_total = result_stat[iou_thresh]['gt']
        object_missing_total = np.sum(result_stat[iou_thresh]['object_missing'])
        fp_false_detection_total = np.sum(result_stat[iou_thresh]['fp_false_detection'])
        fp_localization_total = np.sum(result_stat[iou_thresh]['fp_localization'])
        fn_total = gt_total - tp_total
        localization_failures_total = np.sum(result_stat[iou_thresh].get('localization_failures', [0]))
        
        s1_total = np.sum(result_stat[iou_thresh]['s_1'])
        s2_total = np.sum(result_stat[iou_thresh]['s_2'])
        s3_total = np.sum(result_stat[iou_thresh]['s_3'])
        s4_total = np.sum(result_stat[iou_thresh]['s_4'])
        s5_total = np.sum(result_stat[iou_thresh]['s_5'])
        s6_total = np.sum(result_stat[iou_thresh]['s_6'])
        
        s1_short_total = np.sum(result_stat[iou_thresh]['s_1_short'])
        s2_short_total = np.sum(result_stat[iou_thresh]['s_2_short'])
        s3_short_total = np.sum(result_stat[iou_thresh]['s_3_short'])
        s4_short_total = np.sum(result_stat[iou_thresh]['s_4_short'])
        s5_short_total = np.sum(result_stat[iou_thresh]['s_5_short'])
        s6_short_total = np.sum(result_stat[iou_thresh]['s_6_short'])
        
        
        
        s1_mid_total = np.sum(result_stat[iou_thresh]['s_1_mid'])
        s2_mid_total = np.sum(result_stat[iou_thresh]['s_2_mid'])
        s3_mid_total = np.sum(result_stat[iou_thresh]['s_3_mid'])
        s4_mid_total = np.sum(result_stat[iou_thresh]['s_4_mid'])
        s5_mid_total = np.sum(result_stat[iou_thresh]['s_5_mid'])
        s6_mid_total = np.sum(result_stat[iou_thresh]['s_6_mid'])
        
        s1_long_total = np.sum(result_stat[iou_thresh]['s_1_long'])
        s2_long_total = np.sum(result_stat[iou_thresh]['s_2_long'])
        s3_long_total = np.sum(result_stat[iou_thresh]['s_3_long'])
        s4_long_total = np.sum(result_stat[iou_thresh]['s_4_long'])
        s5_long_total = np.sum(result_stat[iou_thresh]['s_5_long'])
        s6_long_total = np.sum(result_stat[iou_thresh]['s_6_long'])
        

        print(f'At IOU {iou_thresh}: Object Missing: {object_missing_total}, False Detection: {fp_false_detection_total}, Localization Failures: {fp_localization_total}')
        print("Sum:")
        print(f'Situation 1: {s1_total}')
        print(f'Situation 2: {s2_total}')
        print(f'Situation 3: {s3_total}')
        print(f'Situation 4: {s4_total}')
        print(f'Situation 5: {s5_total}')
        print(f'Situation 6: {s6_total}')
        
        print("Short:")
        print(f'Situation 1: {s1_short_total}')
        print(f'Situation 2: {s2_short_total}')
        print(f'Situation 3: {s3_short_total}')
        print(f'Situation 4: {s4_short_total}')
        print(f'Situation 5: {s5_short_total}')
        print(f'Situation 6: {s6_short_total}')
        
        print("Mid:")
        print(f'Situation 1: {s1_mid_total}')
        print(f'Situation 2: {s2_mid_total}')
        print(f'Situation 3: {s3_mid_total}')
        print(f'Situation 4: {s4_mid_total}')
        print(f'Situation 5: {s5_mid_total}')
        print(f'Situation 6: {s6_mid_total}')
        
        print("Long:")
        print(f'Situation 1: {s1_long_total}')
        print(f'Situation 2: {s2_long_total}')
        print(f'Situation 3: {s3_long_total}')
        print(f'Situation 4: {s4_long_total}')
        print(f'Situation 5: {s5_long_total}')
        print(f'Situation 6: {s6_long_total}')
        
        dump_dict.update({
            f'FP_{int(iou_thresh*100)}': int(fp_total),
            f'TP_{int(iou_thresh*100)}': int(tp_total),
            f'FN_{int(iou_thresh*100)}': int(fn_total),
            
            f'AP_{int(iou_thresh*100)}': float(locals()[f'ap_{int(iou_thresh*100)}']),
            
            f'GT_{int(iou_thresh*100)}': int(gt_total),
            
            
            f'LCME_short_{int(iou_thresh*100)}': int(s1_short_total),
            f'LCLE_short_{int(iou_thresh*100)}': int(s2_short_total),
            f'LADE_short_{int(iou_thresh*100)}': int(s3_short_total),
            f'CCME_short_{int(iou_thresh*100)}': int(s4_short_total),
            f'CCLE_short_{int(iou_thresh*100)}': int(s5_short_total),
            f'CADE_short_{int(iou_thresh*100)}': int(s6_short_total),
            
            f'LCME_mid_{int(iou_thresh*100)}': int(s1_mid_total),
            f'LCLE_mid_{int(iou_thresh*100)}': int(s2_mid_total),
            f'LADE_mid_{int(iou_thresh*100)}': int(s3_mid_total),
            f'CCME_mid_{int(iou_thresh*100)}': int(s4_mid_total),
            f'CCLE_mid_{int(iou_thresh*100)}': int(s5_mid_total),
            f'CADE_mid_{int(iou_thresh*100)}': int(s6_mid_total),
            
            f'LCME_long_{int(iou_thresh*100)}': int(s1_long_total),
            f'LCLE_long_{int(iou_thresh*100)}': int(s2_long_total),
            f'LADE_long_{int(iou_thresh*100)}': int(s3_long_total),
            f'CCME_long_{int(iou_thresh*100)}': int(s4_long_total),
            f'CCLE_long_{int(iou_thresh*100)}': int(s5_long_total),
            f'CADE_long_{int(iou_thresh*100)}': int(s6_long_total),
            
            f'LCME_{int(iou_thresh*100)}': int(s1_total),
            f'LCLE_{int(iou_thresh*100)}': int(s2_total),
            f'LADE_{int(iou_thresh*100)}': int(s3_total),
            f'CCLE_{int(iou_thresh*100)}': int(s4_total),
            f'CCME_{int(iou_thresh*100)}': int(s5_total),
            f'CADE_{int(iou_thresh*100)}': int(s6_total),
            

        })
    

#     dump_dict.update({
#         'mpre_50': (mpre_50),
#         'mrec_50': (mrec_50),
#         'mpre_70': (mpre_70),
#         'mrec_70': (mrec_70),
#     })
    
    
#     if infer_info is None:
#         yaml_utils.save_yaml(dump_dict, os.path.join(save_path, 'eval.yaml'))
#     else:
#         yaml_utils.save_yaml(dump_dict, os.path.join(save_path, f'eval_{infer_info}.yaml'))


    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if infer_info is None:
        file_name = f'eval_{current_date}.yaml'
    else:
        file_name = f'eval_{infer_info}_{current_date}.yaml'

    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, file_name))

    print(
          'The Average Precision at IOU 0.5 is %.9f, \n'
          % (ap_50))

    return  ap_50



def compare(det_boxes, det_score, gt_boxes, single_det, single_score, single_gt, result_stat, iou_thresh):
    fp = []
    tp = []
    s_1 = []
    s_2 = []
    s_3 = []
    s_4 = []
    s_5 = []
    s_6 = []
    det_box_xy= []
    gt_box_xy=[]
    if det_boxes is not None and single_det is not None and gt_boxes is not None and single_gt is not None:
        if not isinstance(det_boxes, list) and not isinstance(single_det, list):
            det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
            det_score = common_utils.torch_tensor_to_numpy(det_score)
            gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)


            single_det = common_utils.torch_tensor_to_numpy(single_det)
            single_score = common_utils.torch_tensor_to_numpy(single_score)
            single_gt = common_utils.torch_tensor_to_numpy(single_gt)


            # sort the prediction bounding box by score
            score_order_descend = np.argsort(-det_score)
            det_score = det_score[score_order_descend] # from high to low
            det_polygon_list = list(common_utils.convert_format(det_boxes))
            gt_polygon_list = list(common_utils.convert_format(gt_boxes))

            single_score_order_descend = np.argsort(-single_score)
            single_score = single_score[single_score_order_descend] # from high to low
            single_det_polygon_list = list(common_utils.convert_format(single_det))
            single_gt_polygon_list = list(common_utils.convert_format(single_gt))
        else:
            # sort the prediction bounding box by score
            score_order_descend = np.argsort(-det_score)
            det_score = det_score[score_order_descend] # from high to low
            det_polygon_list = list((det_boxes))
            gt_polygon_list = list((gt_boxes))

            single_score_order_descend = np.argsort(-single_score)
            single_score = single_score[single_score_order_descend] # from high to low
            single_det_polygon_list = list((single_det))
            single_gt_polygon_list = list((single_gt))


        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)
        
        for polygon in det_polygon_list:
            coords = np.array(polygon.exterior.coords)
            mid_x = np.mean(coords[:, 0])
            mid_y = np.mean(coords[:, 1])
            det_box_xy.append([mid_x, mid_y])

        for polygon in gt_polygon_list:
            coords = np.array(polygon.exterior.coords)
            mid_x = np.mean(coords[:, 0])
            mid_y = np.mean(coords[:, 1])
            gt_box_xy.append([mid_x, mid_y])
        

        # GT box 
        ego_gt = Polygon([
            (3,  -1),
            (3, 1),
            (-2, 1),
            (-2, -1),
            (3, -1)
        ])
#         POLYGON ((2.9473648071289062 -1.0638318061828613, 2.947357177734375 1.0644903182983398, -1.954315185546875 1.064490795135498,-1.954315185546875 -1.063830852508545, 2.9473648071289062 -1.0638318061828613))

        ious = common_utils.compute_iou(ego_gt, det_polygon_list)
        max_iou = np.max(ious) if ious.size > 0 else 0.0
        
        ious_1 = common_utils.compute_iou(ego_gt, gt_polygon_list)
        max_iou_1 = np.max(ious_1) if ious_1.size > 0 else 0.0
        
        
        if max_iou >= 0.7 and max_iou_1 < 0.7:
            det_index = np.argmax(ious)
            det_polygon_list.pop(det_index)
            print("POP") 
        
        
        for det_polygon in det_polygon_list:
            
            ious_gt = common_utils.compute_iou(det_polygon, gt_polygon_list)
            ious_ego = common_utils.compute_iou(det_polygon, single_det_polygon_list)

            max_iou_gt = np.max(ious_gt) if ious_gt.size > 0 else 0.0
            max_iou_ego = np.max(ious_ego) if ious_ego.size > 0 else 0.0
            # LADE
            if max_iou_gt <= 0 and max_iou_ego <= 0:
                s_3.append(1)
            else:
                s_3.append(0)
            # CADE
            if max_iou_gt <= 0 and max_iou_ego > 0:
                s_6.append(1)
            else:
                s_6.append(0)


        for gt_polygon in gt_polygon_list:
            ious_ego_gt = common_utils.compute_iou(gt_polygon, single_det_polygon_list)
            max_iou = np.max(ious_ego_gt) if ious_ego_gt.size > 0 else 0.0
            
            
            if max_iou > iou_thresh: 
                
                #LCME
                ious_cp_gt = common_utils.compute_iou(gt_polygon, det_polygon_list)
                max_iou_cp_gt = np.max(ious_cp_gt) if ious_cp_gt.size > 0 else 0.0
                if max_iou_cp_gt <= 0:
                    s_1.append(1)
                else:
                    s_1.append(0)
                
                # LCLE
#                 for i in range(score_order_descend.shape[0]):
#                     det_polygon = det_polygon_list[score_order_descend[i]]
                for det_polygon in det_polygon_list:
                    
                    iou = det_polygon.intersection(gt_polygon).area / det_polygon.union(gt_polygon).area

                    if 0 < iou <= iou_thresh:
                        s_2.append(1)
#                         break
                    else:
                        s_2.append(0)
                
                
            else:
                # CCME
                ious_cp_gt = common_utils.compute_iou(gt_polygon, det_polygon_list)
                max_iou_cp_gt = np.max(ious_cp_gt) if ious_cp_gt.size > 0 else 0.0
                if max_iou_cp_gt <= 0:
                    s_4.append(1)
                else:
                    s_4.append(0)
                
                # CCLE
                for det_polygon in det_polygon_list:
                    
                    iou = det_polygon.intersection(gt_polygon).area / det_polygon.union(gt_polygon).area

                    if 0 < iou <= iou_thresh:
                        s_5.append(1)
#                         break
                    else:
                        s_5.append(0)
                 
    
    result_stat[iou_thresh]['s_1'] += s_1
    result_stat[iou_thresh]['s_2'] += s_2
    result_stat[iou_thresh]['s_3'] += s_3
    result_stat[iou_thresh]['s_4'] += s_4
    result_stat[iou_thresh]['s_5'] += s_5
    result_stat[iou_thresh]['s_6'] += s_6
    
    print(f"Situation 1: {sum(s_1)}")
    print(f"Situation 2: {sum(s_2)}")
    print(f"Situation 3: {sum(s_3)}")
    print(f"Situation 4: {sum(s_4)}")
    print(f"Situation 5: {sum(s_5)}")
    print(f"Situation 6: {sum(s_6)}")
    

def RQ2a(det_boxes, det_score, gt_boxes, single_det, single_score, single_gt, result_stat, iou_thresh):
    gt_ret=[]
    det_ret=[]
    score_ret=[]
    
    det_short=[]
    det_mid=[]
    det_long=[]
    
    gt_short=[]
    gt_mid=[]
    gt_long=[]
    
    
    det_short_single=[]
    det_mid_single=[]
    det_long_single=[]
    
    gt_short_single=[]
    gt_mid_single=[]
    gt_long_single=[]
    if det_boxes is not None and single_det is not None and gt_boxes is not None and single_gt is not None :
    
    
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)


        single_det = common_utils.torch_tensor_to_numpy(single_det)
        single_score = common_utils.torch_tensor_to_numpy(single_score)
        single_gt = common_utils.torch_tensor_to_numpy(single_gt)


        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        single_score_order_descend = np.argsort(-single_score)
        single_score = single_score[single_score_order_descend] # from high to low
        single_det_polygon_list = list(common_utils.convert_format(single_det))
        single_gt_polygon_list = list(common_utils.convert_format(single_gt))
        
        
        for gt in gt_polygon_list:
            d2 = calculate_distance(gt)
            if d2 <= 100:
                gt_ret.append(gt)
        
        
        for i in range(score_order_descend.shape[0]):
            det = det_polygon_list[score_order_descend[i]]
            score = det_score[score_order_descend[i]]
            d1 = calculate_distance(det)
            if d1 <= 100:
                det_ret.append(det)
                score_ret.append(score)
        det_score_numpy = np.array(score_ret)
                
                

        for polygon in det_polygon_list:
            distance = calculate_distance(polygon)
            if distance < 30:
                det_short.append(polygon)
            elif 30 <= distance < 50:
                det_mid.append(polygon)
            elif 50 <= distance < 100:
                det_long.append(polygon)

        for polygon in gt_polygon_list:
            distance = calculate_distance(polygon)
            if distance < 30:
                gt_short.append(polygon)
            elif 30 <= distance < 50:
                gt_mid.append(polygon)
            elif 50 <= distance < 100:
                gt_long.append(polygon)


        for polygon in single_det_polygon_list:
            distance = calculate_distance(polygon)
            if distance < 30:
                det_short_single.append(polygon)
            elif 30 <= distance < 50:
                det_mid_single.append(polygon)
            elif 50 <= distance < 100:
                det_long_single.append(polygon)

        for polygon in single_gt_polygon_list:
            distance = calculate_distance(polygon)
            if distance < 30:
                gt_short_single.append(polygon)
            elif 30 <= distance < 50:
                gt_mid_single.append(polygon)
            elif 50 <= distance < 100:
                gt_long_single.append(polygon)


        compare_distance(det_short, det_score, gt_short, det_short_single, single_score, gt_short_single, result_stat, iou_thresh,'s')
        compare_distance(det_mid, det_score, gt_mid, det_mid_single, single_score, gt_mid_single, result_stat, iou_thresh,'m')
        compare_distance(det_long, det_score, gt_long, det_long_single, single_score, gt_long_single, result_stat, iou_thresh,'l')

        return gt_ret, det_ret, det_score_numpy
    else:
        print(f"[Error in RQ2] ")
        return [], [], np.array([])
        

                

def compare_distance(det_boxes, det_score, gt_boxes, single_det, single_score, single_gt, result_stat, iou_thresh,distance):
    assert distance in ['s','m','l']
    fp = []
    tp = []
    s_1 = []
    s_2 = []
    s_3 = []
    s_4 = []
    s_5 = []
    s_6 = []
    det_box_xy= []
    gt_box_xy=[]
    
    if det_boxes is not None and single_det is not None and gt_boxes is not None and single_gt is not None:
        # convert bounding boxes to numpy array
#         det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
#         det_score = common_utils.torch_tensor_to_numpy(det_score)
#         gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)
        
        
#         single_det = common_utils.torch_tensor_to_numpy(single_det)
#         single_score = common_utils.torch_tensor_to_numpy(single_score)
#         single_gt = common_utils.torch_tensor_to_numpy(single_gt)
        

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list((det_boxes))
        gt_polygon_list = list((gt_boxes))
        
        single_score_order_descend = np.argsort(-single_score)
        single_score = single_score[single_score_order_descend] # from high to low
        single_det_polygon_list = list((single_det))
        single_gt_polygon_list = list((single_gt))
        
        

        # GT box 
        ego_gt = Polygon([
            (3,  -1),
            (3, 1),
            (-2, 1),
            (-2, -1),
            (3, -1)
        ])
#         POLYGON ((2.9473648071289062 -1.0638318061828613, 2.947357177734375 1.0644903182983398, -1.954315185546875 1.064490795135498,-1.954315185546875 -1.063830852508545, 2.9473648071289062 -1.0638318061828613))

        ious = common_utils.compute_iou(ego_gt, det_polygon_list)
        max_iou = np.max(ious) if ious.size > 0 else 0.0
        
        ious_1 = common_utils.compute_iou(ego_gt, gt_polygon_list)
        max_iou_1 = np.max(ious_1) if ious_1.size > 0 else 0.0
        
        
        if max_iou >= 0.7 and max_iou_1 < 0.7:
            det_index = np.argmax(ious)
            det_polygon_list.pop(det_index)
            print("POP")        
        
        
        # match prediction and gt bounding box, in confidence descending order
    

#         for i in range(score_order_descend.shape[0]):
#             det_polygon = det_polygon_list[score_order_descend[i]]
        for det_polygon in det_polygon_list:
            
            ious_gt = common_utils.compute_iou(det_polygon, gt_polygon_list)
            ious_ego = common_utils.compute_iou(det_polygon, single_det_polygon_list)

            max_iou_gt = np.max(ious_gt) if ious_gt.size > 0 else 0.0
            max_iou_ego = np.max(ious_ego) if ious_ego.size > 0 else 0.0
            # LADE
            if max_iou_gt <= 0 and max_iou_ego <= 0:
                s_3.append(1)
            else:
                s_3.append(0)
            # CADE
            if max_iou_gt <= 0 and max_iou_ego > 0:
                s_6.append(1)
            else:
                s_6.append(0)

            # iou判断是不是同一个检测框
#             is_same_box = False
#             iou = single_det_polygon.intersection(det_polygon).area / single_det_polygon.union(det_polygon).area
#             if iou >= iou_thresh:
#                 is_same_box = True
#             else:
#                 is_same_box = False
#                 print(f"Same Box:{is_same_box}") 
        
        for gt_polygon in gt_polygon_list:
            ious_ego_gt = common_utils.compute_iou(gt_polygon, single_det_polygon_list)
            max_iou = np.max(ious_ego_gt) if ious_ego_gt.size > 0 else 0.0
            
            
            if max_iou > iou_thresh: 
                # LCLM
                ious_cp_gt = common_utils.compute_iou(gt_polygon, det_polygon_list)
                max_iou_cp_gt = np.max(ious_cp_gt) if ious_cp_gt.size > 0 else 0.0
                if max_iou_cp_gt <= 0:
                    s_1.append(1)
                else:
                    s_1.append(0)
                
                # LCLE
#                 for i in range(score_order_descend.shape[0]):
#                     det_polygon = det_polygon_list[score_order_descend[i]]
                for det_polygon in det_polygon_list:
                    
                    iou = det_polygon.intersection(gt_polygon).area / det_polygon.union(gt_polygon).area

                    if 0 < iou <= iou_thresh:
                        s_2.append(1)
#                         break
                    else:
                        s_2.append(0)
                
                
            else:
                # Ego 检测错误
                
                #CCLE
                ious_cp_gt = common_utils.compute_iou(gt_polygon, det_polygon_list)
                max_iou_cp_gt = np.max(ious_cp_gt) if ious_cp_gt.size > 0 else 0.0
                if max_iou_cp_gt <= 0:
                    s_4.append(1)
                else:
                    s_4.append(0)
                
                # CCME
#                 for i in range(score_order_descend.shape[0]):
#                     det_polygon = det_polygon_list[score_order_descend[i]]\
                for det_polygon in det_polygon_list:
                    
                    iou = det_polygon.intersection(gt_polygon).area / det_polygon.union(gt_polygon).area

                    if 0 < iou <= iou_thresh:
                        s_5.append(1)
#                         break
                    else:
                        s_5.append(0)
                 
    if distance == 's':
        result_stat[iou_thresh]['s_1_short'] += s_1
        result_stat[iou_thresh]['s_2_short'] += s_2
        result_stat[iou_thresh]['s_3_short'] += s_3
        result_stat[iou_thresh]['s_4_short'] += s_4
        result_stat[iou_thresh]['s_5_short'] += s_5
        result_stat[iou_thresh]['s_6_short'] += s_6
    elif distance == 'm':
        result_stat[iou_thresh]['s_1_mid'] += s_1
        result_stat[iou_thresh]['s_2_mid'] += s_2
        result_stat[iou_thresh]['s_3_mid'] += s_3
        result_stat[iou_thresh]['s_4_mid'] += s_4
        result_stat[iou_thresh]['s_5_mid'] += s_5
        result_stat[iou_thresh]['s_6_mid'] += s_6
    elif distance == 'l':
        result_stat[iou_thresh]['s_1_long'] += s_1
        result_stat[iou_thresh]['s_2_long'] += s_2
        result_stat[iou_thresh]['s_3_long'] += s_3
        result_stat[iou_thresh]['s_4_long'] += s_4
        result_stat[iou_thresh]['s_5_long'] += s_5
        result_stat[iou_thresh]['s_6_long'] += s_6
                
    print("——————————————DISTANCE——————————————————————")
    print(distance)
    print(f"Situation 1: {sum(s_1)}")
    print(f"Situation 2: {sum(s_2)}")
    print(f"Situation 3: {sum(s_3)}")
    print(f"Situation 4: {sum(s_4)}")
    print(f"Situation 5: {sum(s_5)}")
    print(f"Situation 6: {sum(s_6)}")
    
    
    
def RQ2b(det_boxes, det_score, gt_boxes, single_det, single_score, single_gt, result_stat, iou_thresh):
    gt_intersection=[]
    det_intersection=[]
    if det_boxes is not None and single_det is not None and gt_boxes is not None and single_gt is not None:
    
        if not isinstance(det_boxes, list) and not isinstance(single_det, list):
            det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
            det_score = common_utils.torch_tensor_to_numpy(det_score)
            gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)


            single_det = common_utils.torch_tensor_to_numpy(single_det)
            single_score = common_utils.torch_tensor_to_numpy(single_score)
            single_gt = common_utils.torch_tensor_to_numpy(single_gt)


            # sort the prediction bounding box by score
            score_order_descend = np.argsort(-det_score)
            det_score = det_score[score_order_descend] # from high to low
            det_polygon_list = list(common_utils.convert_format(det_boxes))
            gt_polygon_list = list(common_utils.convert_format(gt_boxes))

            single_score_order_descend = np.argsort(-single_score)
            single_score = single_score[single_score_order_descend] # from high to low
            single_det_polygon_list = list(common_utils.convert_format(single_det))
            single_gt_polygon_list = list(common_utils.convert_format(single_gt))
        else:
            # sort the prediction bounding box by score
            score_order_descend = np.argsort(-det_score)
            det_score = det_score[score_order_descend] # from high to low
            det_polygon_list = list((det_boxes))
            gt_polygon_list = list((gt_boxes))

            single_score_order_descend = np.argsort(-single_score)
            single_score = single_score[single_score_order_descend] # from high to low
            single_det_polygon_list = list((single_det))
            single_gt_polygon_list = list((single_gt))
            
        det_score_list = list(det_score)
        single_det_score_list = list(single_score)
        
        det = []
        single_det=[]
        det_score_ret=[]
        single_det_score_ret=[]

        for polygon in gt_polygon_list:
            ious_gt = common_utils.compute_iou(polygon, single_gt_polygon_list)
            max_iou_gt = np.max(ious_gt) if ious_gt.size > 0 else 0.0
            if max_iou_gt >= iou_thresh:
                gt_intersection.append(polygon)
                


        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')
        for polygon in gt_intersection:
            xi,yi = coords(polygon)
            if xi > x_max:
                x_max = xi
            if xi < x_min:
                x_min = xi
            if yi > y_max:
                y_max = yi
            if yi < y_min:
                y_min = yi
        
        for i in range(score_order_descend.shape[0]):
            polygon = det_polygon_list[score_order_descend[i]]
            score = det_score_list[score_order_descend[i]]
            
            xi,yi = coords(polygon)
            print(xi,yi)
            print(in_area(xi,yi,x_max,x_min,y_max,y_min))
            
            if xi > x_max or xi < x_min or yi> y_max or yi< y_min:
                print("NOT IN AREA")
            else:
                det.append(polygon)
                det_score_ret.append(score)
                
        for i in range(single_score_order_descend.shape[0]):
            polygon = single_det_polygon_list[single_score_order_descend[i]]
            score = single_det_score_list[single_score_order_descend[i]]
            
            xi,yi = coords(polygon)
            print(xi,yi)
            print(in_area(xi,yi,x_max,x_min,y_max,y_min))
            
            if xi > x_max or xi < x_min or yi> y_max or yi< y_min:
                print("NOT IN AREA")
            else:
                single_det.append(polygon)
                single_det_score_ret.append(score)



        det_score_numpy = np.array(det_score_ret)
        single_det_score_numpy = np.array(single_det_score_ret)
                                
                
                
        print(f"gt_boxes:{len(gt_polygon_list)}")

        print(f"gt_intersection:{len(gt_intersection)}")
        print(f"det_polygon_list:{len(det_polygon_list)}")
        print(f"det:{len(det)}")
                

        compare(det, det_score_numpy, gt_intersection, single_det, single_det_score_numpy, gt_intersection, result_stat, iou_thresh)
        
        return gt_intersection, det, det_score_numpy
    else:
        print(f"[Error in RQ3] ")
        return [], [], np.array([])
    
    
    
    

def calculate_distance(polygon):
    coords = np.array(polygon.exterior.coords)

    mid_x = np.mean(coords[:4, 0])
    mid_y = np.mean(coords[:4, 1])
    distance = np.sqrt(mid_x**2 + mid_y**2)
    return distance
def coords(polygon):
    coords = np.array(polygon.exterior.coords)
    mid_x = np.mean(coords[:4, 0])
    mid_y = np.mean(coords[:4, 1])

    return mid_x, mid_y

def in_area(x, y, x_min, x_max, y_min, y_max):

    if x > x_max or x < x_min or y> y_max or y< y_min:
        return False
    else:
        return True


