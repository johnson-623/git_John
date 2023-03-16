import nibabel as nib
import numpy as np
import os
import glob
import math
import pandas as pd
import SimpleITK as sitk
import argparse
from metric import calculate_binary_dice, calculate_binary_assd, calculate_binary_hd, calculate_binary_sen, calculate_binary_spe, calculate_binary_acc, calculate_binary_generalized_dice
from medpy.metric import binary

def cal_dc(pre,gt):
    #dice=binary(pre,gt)
    dice=calculate_binary_dice(pre,gt)
    return dice

def load_img(image_name):
    img=sitk.ReadImage(image_name)
    arr=sitk.GetArrayFromImage(img)
    return arr

def evaluate_scar(args):
    root_path=args.data_path
    pre_LAscar_name=args.pre_LAscar_name
    gt_LAscar_name=args.gt_LAscar_name
    items=glob.glob(os.path.join(root_path,'*'))
    case_list=[]
    dc_list=[]
    for item in items:
        case_list.append(item.split('/')[-1])
        pre_path=os.path.join(item,pre_LAscar_name)
        gt_path=os.path.join(item,gt_LAscar_name)
        pre_arr=load_img(pre_path)
        gt_arr=load_img(gt_path)
        pre_arr[pre_arr<=421]=0
        pre_arr[pre_arr==422]=1
        
        dc_list.append(cal_dc(pre_arr,gt_arr))
    save_path=args.save_file
    list={'Casename':case_list,'LAscar_Dice':dc_list}
    df=pd.DataFrame(list)
    df.to_csv(save_path, encoding='gbk', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./dataset/test_data')
    parser.add_argument('--gt_LAscar_name', type=str, default='LA_predict.nii.gz')
    parser.add_argument('--pre_LAscar_name', type=str, default='scar_predict.nii.gz')
    parser.add_argument('--save_file',type=str,default='LAscar_evaluate_result.csv')
    args = parser.parse_args()
    evaluate_scar(args)







 
    


