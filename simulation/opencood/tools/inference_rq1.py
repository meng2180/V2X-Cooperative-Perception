import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
torch.multiprocessing.set_sharing_strategy('file_system')

from itertools import zip_longest

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="51.2,51.2",
                        help="detection range is [-204.8, +204.8, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--use_cav', type=str, default="[1,2,3,4]",
                        help="evaluate with real collaborator number")
    parser.add_argument('--lidar_degrade', action='store_true',
                        help="whether to degrade lidar. {m1:32, m3:16} and {m1:16, m3:16}")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    
    
    
    
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    
    

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)
    

    print(hypes['name'])


    

    if 'heter' in hypes:
#         hypes['heter']['lidar_channels'] = 64
        # opt.note += "_16ch"
#         print(f"Lidar channels: {hypes['heter']['lidar_channels_dic']}")
        lidar_channels = hypes.get('heter', {}).get('lidar_channels_dic', {}).get('m1', 64)
        print(f"Lidar channels: {lidar_channels}")
        print(f"Communication range: {hypes['comm_range']}")

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })
        

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)
        


        
    
    hypes['validate_dir'] = hypes['test_dir']
    

    
    
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
        

    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False
    

    print(f"Left hand visualizing: {left_hand}")
    

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']
        


    print('Creating Model')
    model = train_utils.create_model(hypes)
    


    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    # ___________________________
#     saved_path = opt.single_dir
#     resume_epoch, model1 = train_utils.load_saved_model(saved_path, model1)
#     print(f"resume from {resume_epoch} epoch._________")
#     opt.note += f"_epoch{resume_epoch}"
    
    name_1 = hypes['name']

    
    opt.note += f"_{name_1}"
    opt.note += f"_RQ1"
    
#     opt.note += f"_{name_2}"
    
    
    if torch.cuda.is_available():
        model.cuda()
#         model1.cuda()
    model.eval()
#     model1.eval()
    

    # setting noise
    np.random.seed(303)
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False, single=False)
    
    opencood_dataset_1 = build_dataset(hypes, visualize=True, train=False, single=True)
    
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    data_loader1 = DataLoader(opencood_dataset_1,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset_1.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation

    result_stat_2 = {
        0.5: {
            'fp': [],
            'tp': [],
            'gt': 0,
            'score': [],
            'fp_false_detection': [],
            'fp_localization': [],
            'object_missing': [],
            's_1': [],
            's_2': [],
            's_3': [],
            's_4': [],
            's_5': [],
            's_6': [],
            's_1_short': [],
            's_1_mid': [],
            's_1_long': [],
            's_2_short': [],
            's_2_mid': [],
            's_2_long': [],
            's_3_short': [],
            's_3_mid': [],
            's_3_long': [],
            's_4_short': [],
            's_4_mid': [],
            's_4_long': [],
            's_5_short': [],
            's_5_mid': [],
            's_5_long': [],
            's_6_short': [],
            's_6_mid': [],
            's_6_long': []
        }
    }

    infer_info = opt.fusion_method + opt.note


    for i, (batch_data, batch_data_1) in enumerate(zip(data_loader, data_loader1)):
        print(f"{infer_info}_{i}")
        if batch_data is None :
            continue
        if batch_data_1 is None :
            continue
       
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            
            batch_data_1 = train_utils.to_device(batch_data_1, device)
            
        
            
            # Heter Model
            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
                infer_result_1 = inference_utils.inference_late_fusion(batch_data_1,
                                                        model,
                                                        opencood_dataset_1)

            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
                infer_result_1 = inference_utils.inference_early_fusion(batch_data_1,
                                                        model,
                                                        opencood_dataset_1)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                infer_result_1 = inference_utils.inference_intermediate_fusion(batch_data_1,
                                                        model,
                                                        opencood_dataset_1)

            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                infer_result_1 = inference_utils.inference_no_fusion(batch_data_1,
                                                        model,
                                                        opencood_dataset_1)

            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
                infer_result_1 = inference_utils.inference_no_fusion_w_uncertainty(batch_data_1,
                                                        model,
                                                        opencood_dataset_1)

            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
                infer_result_1 = inference_utils.inference_no_fusion(batch_data_1,
                                                        model,
                                                        opencood_dataset_1,
                                                            single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')
            
                
                

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            
            
            pred_box_tensor_1 = infer_result_1['pred_box_tensor']
            gt_box_tensor_1 = infer_result_1['gt_box_tensor']
            pred_score_1 = infer_result_1['pred_score']
            

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat_2,
                                    0.5)

            eval_utils.compare(pred_box_tensor,
                        pred_score,
                        gt_box_tensor,
                        pred_box_tensor_1,
                        pred_score_1,
                        gt_box_tensor_1,
                        result_stat_2,
                        0.5)
#             eval_utils.caluclate_tp_fp_rq3(det,
#                                     score,
#                                     gt,
#                                     result_stat_2,
#                                     0.5)
            
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})
                infer_result_1.update({'score_tensor': pred_score_1})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "agent_modality_list": agent_modality_list})
            if getattr(opencood_dataset_1, "heterogeneous", False):

                cav_box_np_1, agent_modality_list_1 = inference_utils.get_cav_box(batch_data_1)
                infer_result_1.update({"cav_box_np": cav_box_np_1, \
                                     "agent_modality_list": agent_modality_list_1})
                
                
            # model_dir 可视化保存
            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
                
                
                

#             single_dir 可视化保存
            if (i % opt.save_vis_interval == 0) and (pred_box_tensor_1 is not None or gt_box_tensor_1 is not None):
                vis_save_path_root_1 = os.path.join(opt.model_dir, f'vis_{infer_info}_single')
                if not os.path.exists(vis_save_path_root_1):
                    os.makedirs(vis_save_path_root_1)

                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path_1 = os.path.join(vis_save_path_root_1, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result_1,
                                    batch_data_1['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path_1,
                                    method='bev',
                                    left_hand=left_hand)
                
                
                

        torch.cuda.empty_cache()

    ap50 = eval_utils.eval_final_results(result_stat_2,
                                opt.model_dir, infer_info)
    ap50_1 = eval_utils.eval_final_results(result_stat_2,
                                "opencood/logs/failure", infer_info)

if __name__ == '__main__':
    main()
