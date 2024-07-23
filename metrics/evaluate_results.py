# ARB、FRB：Square root is applied to the final averaged result just for reducing the scale.
import json
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score, f1_score
import numpy as np

def mean_min_ADE_FDE(data):
    # Extracting the relevant values for ADE_1.0, FDE_1.0, ADE_2.0, and FDE_2.0
    ade_1_0_values = [data[key]['1.0'] for key in data if 'ADE' in key]
    fde_1_0_values = [data[key]['1.0'] for key in data if 'FDE' in key]
    ade_2_0_values = [data[key]['2.0'] for key in data if 'ADE' in key]
    fde_2_0_values = [data[key]['2.0'] for key in data if 'FDE' in key]

    # Calculating averages
    average_ade_1_0 = sum(ade_1_0_values) / len(ade_1_0_values)
    average_fde_1_0 = sum(fde_1_0_values) / len(fde_1_0_values)
    average_ade_2_0 = sum(ade_2_0_values) / len(ade_2_0_values)
    average_fde_2_0 = sum(fde_2_0_values) / len(fde_2_0_values)

    # Calculating minimum values
    min_ade_1_0 = min(ade_1_0_values)
    min_fde_1_0 = min(fde_1_0_values)
    min_ade_2_0 = min(ade_2_0_values)
    min_fde_2_0 = min(fde_2_0_values)

    results={
        'Min_ADE':min_ade_2_0,
        'Min_FDE':min_fde_2_0,
        'ADE':average_ade_2_0,
        'FDE':average_fde_2_0
    }
    return results
def evaluate_traj(groundtruth='', prediction='', args=None):
    with open(groundtruth, 'r') as f:
        gt_traj = json.load(f)

    with open(prediction, 'r') as f:
        pred_traj = json.load(f)

    gt = []
    pred = []
    for vid in gt_traj.keys():
        for pid in gt_traj[vid]:
            gt.append(pid)
    for vid in pred_traj.keys():
        for pid in pred_traj[vid]['pred_traj']:
            pred.append(pid)
      
    gt = np.array(gt) # ( ，50，2)
    pred = np.array(pred) # （ ，10，50，2）
    results = {}
    for i in range(pred.shape[1]):
        sub_pred = pred[:, i, :, :]
        results[f'ADE_{i}']=  {'1.0': 0, '2.0': 0}
        results[f'FDE_{i}']=  {'1.0': 0, '2.0': 0}

        measure_traj_prediction(gt, sub_pred,results,i)
   
    return mean_min_ADE_FDE(results)

def measure_traj_prediction(target, prediction, results,item):
    print("Evaluating Trajectory ...")
    target = np.array(target)
    prediction = np.array(prediction)
    assert target.shape[1] == 50
    assert target.shape[2] == 2  # bbox
    assert prediction.shape[1] == 50
    assert prediction.shape[2] == 2
   


    performance_CMSE = np.square(target - prediction).sum(axis=2)  # bs x ts x 4 --> bs x ts
    performance_CRMSE = np.sqrt(performance_CMSE)  # bs x ts

    for t in [1.0, 2.0]:
        end_frame = int(t * 25)
        # 7. ADE - center
        results[f'ADE_{item}'][str(t)] = performance_CRMSE[:, : end_frame].mean(axis=None)
        # 8. FDE - center
        results[f'FDE_{item}'][str(t)] = performance_CRMSE[:, end_frame - 1].mean(axis=None)



if __name__ == '__main__':
    gt = 'output/reulsts/20240721_123635/test_gt.json'  # (*,50,2)
    pred = 'output/reulsts/20240721_123635/test_pred.json'  #(*,10,50,2)
    print(evaluate_traj(groundtruth=gt,prediction=pred))

    # args = None
    # dataset='JAAD'
    # model_type = 'lstmed_traj_bbox' # lstmed_traj_bbox or Transformer
    # # Evaluate driving decision prediction
    # test_gt_file ='/home/jiyufei/code/trajectory_pred/test_gt/JAAD/test_traj_gt.json'
    # # ours
    # # test_pred_file = '/home/jiyufei/code/trajectory_pred/ckpts/ped_traj/PSI1.0/lstmed_traj_bbox/20231130222334/results/test_traj_pred.json'
    # # test_pred_file='/home/jiyufei/dataset/PSI1.0/PSI-Trajectory-Prediction/ckpts/ped_traj/PSI1.0/lstmed_traj_bbox/20231130230751/results/test_traj_pred.json'
    # dict={
      
    #     f'{dataset}: bboxes':'20240325091223',

    # }
    # for key,value in dict.items():
    #     metrics_list=[]
    #     print('————'*10,key,model_type)
    #     for id in range(70,81):
    #         test_pred_file=f'/home/jiyufei/code/trajectory_pred/ckpts/ped_traj/JAAD/{model_type}/{value}/results/test_{id}_traj_pred.json'
    #         metrics_list.append(evaluate_traj(test_gt_file, test_pred_file, args))
        

    #     for metric in ['ADE', 'FDE','ARB','FRB']: #, 'Bbox_MSE', 'Bbox_FMSE', 'Center_MSE', 'Center_FMSE']:
    #         for time in ['0.5', '1.0']:
    #             val=0
    #             for item in range(0,11):
    #                 val += metrics_list[item][metric][time]
    #             print(f'Eval/Results/{metric}_{time}', val/11)

    
    #     # print("Rankding score is : ", score)