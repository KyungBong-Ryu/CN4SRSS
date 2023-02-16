# [기본 라이브러리]----------------------
import os
import numpy as np
import random
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import ignite   #pytorch ignite

import argparse
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import time
import warnings

# [py 파일]--------------------------
from utils.calc_func import *
from utils.data_load_n_save import *
from utils.data_tool import *

from mps.mp_sssr_plt_saver import plts_saver as plts_saver_sssr     # support
from mps.mp_sr_plt_saver   import plts_saver as plts_saver_sr       # support
from mps.mp_ss_plt_saver   import plts_saver as plts_saver_ss       # support
from mps.mp_dataloader     import DataLoader_multi_worker_FIX       # now multi-worker can be used on windows 10

# https://github.com/KyungBong-Ryu/Codes_implementation/blob/main/BasicSR_NIQE.py
from DLCs.BasicSR_NIQE import calc_niqe as _calc_niqe

#<<< @@@ trainer_

# Supporting model_name -> 여기서 list에 이름 추가하면 trainer_ 내부 나머진 자동으로 처리됨
# 1. SSSR
#       SSSR_A -> 3가지 출력물, 2가지 loss 입력
list_sssr_model_type_a = ["model_a"]
#       SSSR_B -> 3가지 출력물, 3가지 loss 입력
list_sssr_model_type_b = ["model_b"]
#       SSSR_C -> 4가지 출력물, 4가지 loss 입력
list_sssr_model_type_c = ["model_c"]
#       SSSR_D -> list형 출력물, list형 loss 입력
list_sssr_model_type_d = ["model_d"]

#
# 2. SR
#       SR_A -> 출력이 list 형태인 모델
list_sr_model_type_a = ["model_c"]
#       SR_B -> 출력이 단일 tensor 형태인 모델
list_sr_model_type_b = ["ESRT", "HAN", "IMDN", "BSRN", "RFDN"]
#       SR_C -> label을 loss에 반영
list_sr_model_type_c = ["IMDN_LOSS"]
#
#
# 3. SS
#       D3P: (DeepLab v3 Plus)
#       SS_A -> 출력이 단일 tensor 형태인 모델 (입출력물 크기 조정기능 제공)
list_ss_model_type_a = ["D3P", "DABNet", "CGNet"]




# [ 범용성이 떨어져서 trainer 함수 내부에서 선언한 함수들 ]-----------------------------------------------------------------

# calc_miou_gray 함수의 dict_ious 결과물을 하나의 dict에 누적시키는 함수
# 누적할 dict, miou 값(float), iou dict
def accumulate_dict_ious(dict_accumulate, miou, dict_ious):
    # 형태 = "dict_ious와 동일한 key" : (유효 iou 수, iou 누적값)
    # dict_accumulate = kargs['dict_accumulate']

    # dict_ious = kargs['dict_ious']
    
    if "miou" in dict_accumulate:
        item_current = dict_accumulate["miou"]
    else:
        item_current = (0,0)
        
    dict_accumulate["miou"] = (item_current[0] + 1, item_current[1] + miou)
    
    for i_key in dict_ious:
        # 초기값 불러오기
        if i_key in dict_accumulate:
            # 유효한 key 인 경우
            item_current = dict_accumulate[i_key]
        else:
            # key가 존재하지 않았을 경우
            item_current = (0,0)
        
        if dict_ious[i_key] == "NaN":
            # 현재 라벨에 대한 값이 유효하지 않은 경우
            item_new = item_current
        else:
            # 현재 라벨에 대한 값이 유효한 경우
            item_new = (item_current[0] + 1, item_current[1] + float(dict_ious[i_key]))
        
        # dict 업데이트
        dict_accumulate[i_key] = item_new
    
#=== End of accumulate_dict_ious


# log 편의성을 위한 mIoU 더미 데이터 생성기
def dummy_calc_miou_gray(**kargs):
    #pil_gray_answer = kargs['pil_gray_answer']
    #pil_gray_predict = kargs['pil_gray_predict']
    int_total_labels  = kargs['int_total_labels']
    in_int_void = kargs['int_void_label']
    
    dict_iou = {}
    
    for i_label in range(int_total_labels):
        #void 라벨이 아닌 경우
        if(i_label != in_int_void):
            dict_iou[str(i_label)] = str(-9)

    
    return -9, dict_iou

#=== End of dummy_calc_miou_gray

def dummy_calc_pa_ca_miou_gray(**kargs):
    #pil_gray_answer = kargs['pil_gray_answer']
    #pil_gray_predict = kargs['pil_gray_predict']
    int_total_labels  = kargs['int_total_labels']
    in_int_void = kargs['int_void_label']
    
    dict_iou = {}
    
    for i_label in range(int_total_labels):
        #void 라벨이 아닌 경우
        if(i_label != in_int_void):
            dict_iou[str(i_label)] = str(-9)

    
    return -9, -9, -9, dict_iou

#=== End of dummy_calc_pa_ca_miou_gray

def dummy_calc_pa_ca_miou_gray_tensor(**kargs):
    #ts_ans   = kargs['ts_ans']
    #ts_pred  = kargs['ts_pred']
    int_total = kargs['int_total']
    int_void  = kargs['int_void']
    
    dict_iou = {}
    
    for i_label in range(int_total):
        #void 라벨이 아닌 경우
        if(i_label != int_void):
            dict_iou[str(i_label)] = str(-9)

    
    return -9, -9, -9, dict_iou

#=== End of dummy_calc_pa_ca_miou_gray_tensor


def trainer_(**kargs):
    
    
    
    # [최우선 초기화요소 시행]------------------------------------------------------------------------
    # Trainer 모드 설정 ("SS", "SR", "SSSR")
    TRAINER_MODE = kargs['TRAINER_MODE']
    
    
    print("\n#=========================#")
    print("  Trainer Mode:", TRAINER_MODE)
    print("#=========================#\n")
    
    try:
        is_gradient_clipped    = kargs['is_gradient_clipped']
        gradient_clip_max_norm = float(kargs['gradient_clip_max_norm'])
    except:
        is_gradient_clipped    = False
        
    
    
    if is_gradient_clipped:
        _str = "Gradient Clipping Activated: " + str(gradient_clip_max_norm)
        warnings.warn(_str)
    else:
        _str = "Gradient Clipping Deactivated"
        warnings.warn(_str)
    
    # 사용할 데이터셋 종류
    try:
        HP_DATASET_NAME = kargs['HP_DATASET_NAME']
    except:
        _str = "HP_DATASET_NAME -> 사용할 데이터셋의 종류를 입력해 주세요. (지원목록: CamVid, MiniCity)"
        sys.exit(_str)
    
    # colab 여부 판별용 (colab == -1)
    try:
        RUN_WHERE = kargs['RUN_WHERE']
    except:
        RUN_WHERE = 1
    
    
    # Test 과정에서 저장할 이미지 수를 줄일 것인가?
    REDUCE_SAVE_IMAGES = kargs['REDUCE_SAVE_IMAGES']
    # Test 과정에서 반드시 저장할 이미지 이름
    MUST_SAVE_IMAGE = kargs['MUST_SAVE_IMAGE']
    
    # 최대로 저장할 이미지 수
    # Val & Test 는 REDUCE_SAVE_IMAGES 옵션이 활성화 된 경우에 적용됨
    if RUN_WHERE == -1:
        MAX_SAVE_IMAGES = 1
        REDUCE_SAVE_IMAGES = True
    else:
        try:
            MAX_SAVE_IMAGES = int(kargs['MAX_SAVE_IMAGES'])
        except:
            MAX_SAVE_IMAGES = 10
    
    if MAX_SAVE_IMAGES < 1:
        MAX_SAVE_IMAGES = 1
        print("MAX_SAVE_IMAGES should be >= 1. It fixed to", MAX_SAVE_IMAGES)
        
    
    employ_threshold = 7        # sub-process 생성 경계걊 (이 값 이하의 데이터는 단일 프로세스로 처리)
    
    
    
    try:
        BUFFER_SIZE = kargs['BUFFER_SIZE']
        
        if BUFFER_SIZE < 1:
            print("BUFFER_SIZE should be > 0")
            BUFFER_SIZE = 60
    except:
        BUFFER_SIZE = 60
    
    print("BUFFER_SIZE set to", BUFFER_SIZE)
    
    # log dict 이어받기
    dict_log_init = kargs['dict_log_init']
    
    # 사용 decive 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 랜덤 시드(seed) 적용
    HP_SEED = kargs['HP_SEED']
    random.seed(HP_SEED)
    np.random.seed(HP_SEED)
    # pytorch 랜덤시드 고정 (CPU)
    torch.manual_seed(HP_SEED)
    
    
    update_dict_v2("", ""
                  ,"", "랜덤 시드값 (random numpy pytorch)"
                  ,"", "HP_SEED: " + str(HP_SEED)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    if device == torch.device('cuda'):
        warnings.warn("RUN with cuda")
        # pytorch 랜덤시드 고정 (GPU & multi-GPU)
        torch.cuda.manual_seed(HP_SEED)
        torch.cuda.manual_seed_all(HP_SEED)
        # 세부 디버깅용 오류문 출력
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    else:
        warnings.warn("RUN on CPU")
    
    # epoch 수, batch 크기 & (train) 데이터셋 루프 횟수
    HP_EPOCH        = kargs['HP_EPOCH']
    HP_BATCH_TRAIN  = kargs['HP_BATCH_TRAIN']
    #HP_DATASET_LOOP = 1                             # -> 이제 사용 안함
    HP_BATCH_VAL    = kargs['HP_BATCH_VAL']
    HP_BATCH_TEST   = kargs['HP_BATCH_TEST']
    
    try:
        HP_NUM_WORKERS = int(kargs['HP_NUM_WORKERS'])
        _total_worker = mp.cpu_count()
        
        if _total_worker <= 2:
            print("total workers are not enough to use multi-worker")
            HP_NUM_WORKERS = 0
        elif _total_worker < HP_NUM_WORKERS*2:
            print("too much worker!")
            HP_NUM_WORKERS = int(_total_worker//2)
        
    except:
        HP_NUM_WORKERS = 0
    
    update_dict_v2("", ""
                  ,"", "최대 epoch 설정: " + str(HP_EPOCH)
                  ,"", "batch 크기"
                  ,"", "HP_BATCH_TRAIN: " + str(HP_BATCH_TRAIN)
                  ,"", "HP_NUM_WORKERS for train: " + str(HP_NUM_WORKERS)
                  #,"", "학습 시 데이터셋 반복횟수"
                  #,"", "HP_DATASET_LOOP: " + str(HP_DATASET_LOOP)
                  ,"", "그래디언트 축적(Gradient Accumulation) 사용 안함"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # [입출력 Data 관련]-----------------------------
    # 경로: 입력
    PATH_BASE_IN        = kargs['PATH_BASE_IN']
    NAME_FOLDER_TRAIN   = kargs['NAME_FOLDER_TRAIN']
    NAME_FOLDER_VAL     = kargs['NAME_FOLDER_VAL']
    NAME_FOLDER_TEST    = kargs['NAME_FOLDER_TEST']
    NAME_FOLDER_IMAGES  = kargs['NAME_FOLDER_IMAGES']
    NAME_FOLDER_LABELS  = kargs['NAME_FOLDER_LABELS']
    
    # 경로: 출력
    PATH_OUT_IMAGE = kargs['PATH_OUT_IMAGE']
    if PATH_OUT_IMAGE[-1] != '/':
        PATH_OUT_IMAGE += '/'
    PATH_OUT_MODEL = kargs['PATH_OUT_MODEL']
    if PATH_OUT_MODEL[-1] != '/':
        PATH_OUT_MODEL += '/'
    PATH_OUT_LOG = kargs['PATH_OUT_LOG']
    if PATH_OUT_LOG[-1] != '/':
        PATH_OUT_LOG += '/'
    
    # 원본 이미지 크기
    HP_ORIGIN_IMG_W = kargs['HP_ORIGIN_IMG_W']
    HP_ORIGIN_IMG_H = kargs['HP_ORIGIN_IMG_H']
    
    # 이미지 & 라벨 & 
    HP_CHANNEL_RGB  = kargs['HP_CHANNEL_RGB']
    HP_CHANNEL_GRAY = kargs['HP_CHANNEL_GRAY']
    
    
    #라벨 관련 정보
    try:
        HP_LABEL_TOTAL  = kargs['HP_LABEL_TOTAL']
    except:
        HP_LABEL_TOTAL  = 2
    
    try:
        HP_LABEL_VOID   = kargs['HP_LABEL_VOID']
    except:
        HP_LABEL_VOID   = HP_LABEL_TOTAL
    
    
    #Patch 생성 관련
    is_use_patch = kargs['is_use_patch']
    if is_use_patch:
        # 이미지 모델입력 Patch 크기 (train & val)
        HP_MODEL_IMG_W = kargs['HP_MODEL_IMG_W']
        HP_MODEL_IMG_H = kargs['HP_MODEL_IMG_H']
        
        HP_PATCH_STRIDES = kargs['HP_PATCH_STRIDES']
        HP_PATCH_CROP_INIT_COOR_RANGE = kargs['HP_PATCH_CROP_INIT_COOR_RANGE']
        update_dict_v2("", ""
                      ,"", "원본 Dataset 이미지 크기"
                      ,"", "HP_ORIGIN_IMG_(W H): (" + str(HP_ORIGIN_IMG_W) + " " + str(HP_ORIGIN_IMG_H) + ")"
                      ,"", "모델 입출력 이미지 Patch 크기 (train val)"
                      ,"", "HP_MODEL_IMG_(W H): (" + str(HP_MODEL_IMG_W) + " " + str(HP_MODEL_IMG_H) + ")"
                      ,"", "이미지 채널 수 (이미지): " + str(HP_CHANNEL_RGB)
                      ,"", "Patch 생성함"
                      ,"", "stride (w and h): " + str(HP_PATCH_STRIDES[0]) + " and " + str(HP_PATCH_STRIDES[1])
                      ,"", "crop 시작: " + str(HP_PATCH_CROP_INIT_COOR_RANGE[0]) + " ~ " + str(HP_PATCH_CROP_INIT_COOR_RANGE[1])
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
    else:
        update_dict_v2("", ""
                      ,"", "원본 Dataset 이미지 크기"
                      ,"", "HP_ORIGIN_IMG_(W H): (" + str(HP_ORIGIN_IMG_W) + " " + str(HP_ORIGIN_IMG_H) + ")"
                      ,"", "이미지 채널 수 (이미지): " + str(HP_CHANNEL_RGB)
                      ,"", "Patch 생성 안함"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
        try:
            # 강제 margin 추가여부
            is_force_fix = kargs['is_force_fix']
            _w, _h = kargs['force_fix_size_hr']
            force_fix_size_hr = (int(_w), int(_h))
            
            update_dict_v2("", "Train & Valid 이미지 강제 Margin 추가 시행함"
                          ,"", "Margin 포함 (W H): (" + str(int(_w)) + " " + str(int(_h)) + ")"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
            
        except:
            is_force_fix = False
            force_fix_size_hr = None
            update_dict_v2("", "Train & Valid 이미지 강제 Margin 추가 시행 안함"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        
        HP_MODEL_IMG_W = None
        HP_MODEL_IMG_H = None
        HP_PATCH_STRIDES = (1,1)
        HP_PATCH_CROP_INIT_COOR_RANGE = (0,0)
        
    
    
    
    
    # 데이터셋 정보
    if HP_DATASET_NAME == "CamVid":
        # 컬러매핑
        update_dict_v2("", ""
                      ,"", "사용된 데이터셋: CamVid"
                      ,"", "라벨 별 RGB 매핑"
                      ,"", "0:  [128 128 128]  # 00 sky"
                      ,"", "1:  [128   0   0]  # 01 building"
                      ,"", "2:  [192 192 128]  # 02 column_pole"
                      ,"", "3:  [128  64 128]  # 03 road"
                      ,"", "4:  [  0   0 192]  # 04 sidewalk"
                      ,"", "5:  [128 128   0]  # 05 Tree"
                      ,"", "6:  [192 128 128]  # 06 SignSymbol"
                      ,"", "7:  [ 64  64 128]  # 07 Fence"
                      ,"", "8:  [ 64   0 128]  # 08 Car"
                      ,"", "9:  [ 64  64   0]  # 09 Pedestrian"
                      ,"", "10: [  0 128 192]  # 10 Bicyclist"
                      ,"", "11: [  0   0   0]  # 11 Void"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
    elif HP_DATASET_NAME == "MiniCity":
        update_dict_v2("", ""
                      ,"", "사용된 데이터셋: MiniCity"
                      ,"", "라벨 별 RGB 매핑"
                      ,"", "0:  [128  64 128]     # 00 Road"
                      ,"", "1:  [244  35 232]     # 01 Sidewalk"
                      ,"", "2:  [ 70  70  70]     # 02 Building"
                      ,"", "3:  [102 102 156]     # 03 Wall"
                      ,"", "4:  [190 153 153]     # 04 Fence"
                      ,"", "5:  [153 153 153]     # 05 Pole"
                      ,"", "6:  [250 170  30]     # 06 Traffic light"
                      ,"", "7:  [220 220   0]     # 07 Traffic sign"
                      ,"", "8:  [107 142  35]     # 08 Vegetation"
                      ,"", "9:  [152 251 152]     # 09 Terrain"
                      ,"", "10: [ 70 130 180]     # 10 Sky"
                      ,"", "11: [220  20  60]     # 11 Person"
                      ,"", "12: [255   0   0]     # 12 Rider"
                      ,"", "13: [  0   0 142]     # 13 Car"
                      ,"", "14: [  0   0  70]     # 14 Truck"
                      ,"", "15: [  0  60 100]     # 15 Bus"
                      ,"", "16: [  0  80 100]     # 16 Train"
                      ,"", "17: [  0   0 230]     # 17 Motorcycle"
                      ,"", "18: [119  11  32]     # 18 Bicycle"
                      ,"", "19: [  0   0   0]     # 19 Void"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
    else:
        sys.exit("지원하지 않는 데이터셋 입니다.")
    
    # [model, 모델 추가정보, optimizer, scheduler, loss]--------------------
    model_name = kargs['model_name']
    
    model_type = None
    
    if TRAINER_MODE == "SSSR":
        # SSSR
        if model_name in list_sssr_model_type_a:
            model_type = "SSSR_A"
        elif model_name in list_sssr_model_type_b:
            model_type = "SSSR_B"
        elif model_name in list_sssr_model_type_c:
            model_type = "SSSR_C"
        elif model_name in list_sssr_model_type_d:
            model_type = "SSSR_D"
        else:
            # Not support
            sys.exit("지원하지 않는 모델입니다.")
    
    elif TRAINER_MODE == "SR":
        # SR
        if model_name in list_sr_model_type_a:
            model_type = "SR_A"
        elif model_name in list_sr_model_type_b:
            model_type = "SR_B"
        elif model_name in list_sr_model_type_c:
            model_type = "SR_C"
        else:
            # Not support
            sys.exit("지원하지 않는 모델입니다.")
        
    elif TRAINER_MODE == "SS":
        # SS
        if model_name in list_ss_model_type_a:
            model_type = "SS_A"
        else:
            # Not support
            sys.exit("지원하지 않는 모델입니다.")
        
    else:
        # Not support
        sys.exit("지원하지 않는 모델입니다.")
    
    update_dict_v2("", ""
                  ,"", "모델 종류: " + str(model_name)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    model                           = kargs['model']
    optimizer                       = kargs['optimizer']
    scheduler                       = kargs['scheduler']
    HP_SCHEDULER_UPDATE_INTERVAL    = kargs['HP_SCHEDULER_UPDATE_INTERVAL']
    # loss
    criterion                       = kargs['criterion']
    try:
        HP_DETECT_LOSS_ANOMALY      = kargs['HP_DETECT_LOSS_ANOMALY']
    except:
        HP_DETECT_LOSS_ANOMALY      = False
    
    if HP_DETECT_LOSS_ANOMALY:
        # with torch.autograd.detect_anomaly():
        update_dict_v2("", ""
                      ,"", "Train loss -> detect_anomaly 사용됨"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    else:
        update_dict_v2("", ""
                      ,"", "Train loss -> detect_anomaly 사용 안됨"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    # [Automatic Mixed Precision 선언] ---
    amp_scaler = torch.cuda.amp.GradScaler(enabled = True)
    update_dict_v2("", ""
                  ,"", "Automatic Mixed Precision 사용됨"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # [data augmentation 관련]--------------------
    try:
        HP_AUGM_LITE        = kargs['HP_AUGM_LITE']
    except:
        HP_AUGM_LITE        = False
    HP_AUGM_RANGE_CROP_INIT = kargs['HP_AUGM_RANGE_CROP_INIT']
    HP_AUGM_ROTATION_MAX    = kargs['HP_AUGM_ROTATION_MAX']
    HP_AUGM_PROB_FLIP       = kargs['HP_AUGM_PROB_FLIP']
    HP_AUGM_PROB_CROP       = kargs['HP_AUGM_PROB_CROP']
    HP_AUGM_PROB_ROTATE     = kargs['HP_AUGM_PROB_ROTATE']
    # colorJitter 관련
    # https://pytorch.org/vision/master/generated/torchvision.transforms.ColorJitter.html#torchvision.transforms.ColorJitter
    HP_CJ_BRIGHTNESS        = kargs['HP_CJ_BRIGHTNESS']
    HP_CJ_CONTRAST          = kargs['HP_CJ_CONTRAST']
    HP_CJ_SATURATION        = kargs['HP_CJ_SATURATION']
    HP_CJ_HUE               = kargs['HP_CJ_HUE']
    
    transform_cj            = transforms.ColorJitter(brightness = HP_CJ_BRIGHTNESS
                                                    ,contrast   = HP_CJ_CONTRAST
                                                    ,saturation = HP_CJ_SATURATION
                                                    ,hue        = HP_CJ_HUE
                                                    )
    
    update_dict_v2("", ""
                  ,"", "ColorJitter 설정"
                  ,"", "brightness: ( " + " ".join([str(t_element) for t_element in HP_CJ_BRIGHTNESS]) +" )"
                  ,"", "contrast:   ( " + " ".join([str(t_element) for t_element in HP_CJ_CONTRAST])   +" )"
                  ,"", "saturation: ( " + " ".join([str(t_element) for t_element in HP_CJ_SATURATION]) +" )"
                  ,"", "hue:        ( " + " ".join([str(t_element) for t_element in HP_CJ_HUE])        +" )"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    try:
        HP_AUGM_RANDOM_SCALER = kargs['HP_AUGM_RANDOM_SCALER']
    except:
        HP_AUGM_RANDOM_SCALER = [1.0]
    
    update_dict_v2("", ""
                  ,"", "RANDOM_SCALER 설정"
                  ,"", "List: [ " + " ".join([str(t_element) for t_element in HP_AUGM_RANDOM_SCALER]) +" ]"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # [이미지 변수 -> 텐서 변수 변환]-------------------
    # 정규화 여부
    is_norm_in_transform_to_tensor = kargs['is_norm_in_transform_to_tensor']
    
    if is_norm_in_transform_to_tensor:
        # 평균
        HP_TS_NORM_MEAN = kargs['HP_TS_NORM_MEAN']
        # 표준편차
        HP_TS_NORM_STD = kargs['HP_TS_NORM_STD']
        # 입력 이미지 텐서 변환 후 정규화 시행
        transform_to_ts_img = transforms.Compose([# PIL 이미지 or npArray -> pytorch 텐서
                                                  transforms.ToTensor()
                                                  # 평균, 표준편차를 활용해 정규화
                                                 ,transforms.Normalize(mean = HP_TS_NORM_MEAN, std = HP_TS_NORM_STD)
                                                 ,
                                                 ])
        
        # 역정규화 변환
        transform_ts_inv_norm = transforms.Compose([# 평균, 표준편차를 역으로 활용해 역정규화
                                                    transforms.Normalize(mean = [ 0., 0., 0. ]
                                                                        ,std = [ 1/HP_TS_NORM_STD[0], 1/HP_TS_NORM_STD[1], 1/HP_TS_NORM_STD[2] ])
                                                     
                                                   ,transforms.Normalize(mean = [ -HP_TS_NORM_MEAN[0], -HP_TS_NORM_MEAN[1], -HP_TS_NORM_MEAN[2] ]
                                                                        ,std = [ 1., 1., 1. ])
                                                                        
                                                   ,
                                                   ])
        
        update_dict_v2("", ""
                      ,"", "입력 이미지(in_x) 정규화 시행됨"
                      ,"", "mean=[ " + str(HP_TS_NORM_MEAN[0]) + " " + str(HP_TS_NORM_MEAN[1]) + " "+ str(HP_TS_NORM_MEAN[2]) + " ]"
                      ,"", "std=[ " + str(HP_TS_NORM_STD[0]) + " " + str(HP_TS_NORM_STD[1]) + " "+ str(HP_TS_NORM_STD[2]) + " ]"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    else:
        # 정규화 없이 이미지를 텐서형으로 변환
        transform_to_ts_img = transforms.Compose([# PIL 이미지 or npArray -> pytorch 텐서
                                                  # https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html#torchvision.transforms.ToTensor
                                                  # 일반적인 경우 (PIL mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1 또는 numpy.ndarray), 
                                                  # (H x W x C) in the range [0, 255] 입력 데이터를
                                                  # (C x H x W) in the range [0.0, 1.0] 출력 데이터로 변환함 (scaled)
                                                  transforms.ToTensor()
                                                 ])
        
        update_dict_v2("", ""
                      ,"", "입력 이미지(in_x) 정규화 시행 안함"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    # [Degradation 관련]-------------------------------
    # (bool) 학습 & 평가 시 Degradaded Input 사용 여부
    option_apply_degradation = True
    
    if option_apply_degradation:
        # "Train & Test 과정에 Degradation 시행 됨"
        
        try:
            PATH_BASE_IN_SUB = kargs['PATH_BASE_IN_SUB']
            HP_DG_CSV_NAME = kargs['HP_DG_CSV_NAME']
            if not PATH_BASE_IN_SUB[-1] == "/":
                PATH_BASE_IN_SUB += "/"
            
            dict_loaded_pils = load_pils_2_dict(# 경로 내 pil 이미지를 전부 불러와서 dict 형으로 묶어버림
                                                # (str) 파일 경로
                                                in_path = PATH_BASE_IN_SUB
                                                # (선택, str) 파일 경로 - 하위폴더명
                                               ,in_path_sub = NAME_FOLDER_IMAGES
                                               )
            print("Pre-Degraded images loaded from:", PATH_BASE_IN_SUB + NAME_FOLDER_IMAGES)
            
            dict_dg_csv = csv_2_dict(path_csv = PATH_BASE_IN_SUB + HP_DG_CSV_NAME)
            print("Pre-Degrade option csv re-loaded from:", PATH_BASE_IN_SUB + HP_DG_CSV_NAME)
            
            flag_pre_degraded_images_loaded = True
            tmp_log_pre_degraded_images_load = "Degraded 이미지를 불러왔습니다."
        except:
            print("(exc) Pre-Degraded images load FAIL")
            flag_pre_degraded_images_loaded = False
            tmp_log_pre_degraded_images_load = "Degraded 이미지를 불러오지 않습니다."
        
        update_dict_v2("", ""
                      ,"", "Degraded 이미지 옵션: " + tmp_log_pre_degraded_images_load
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
        # scale_factor
        HP_DG_SCALE_FACTOR = kargs['HP_DG_SCALE_FACTOR']                # Dataloader 에서 쓰임
            
        update_dict_v2("", ""
                      ,"", "Degradation 관련"
                      ,"", "시행여부: " + "Train & Valid % Test 과정에 Degradation 시행 됨"
                      ,"", "DG 지정값 파일 경로: " + PATH_BASE_IN_SUB + HP_DG_CSV_NAME
                      ,"", "Scale Factor 고정값 = x" + str(HP_DG_SCALE_FACTOR)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )

            
        
    else:
        # "Train & Test 과정에 Degradation 시행 안됨"
        update_dict_v2("", ""
                      ,"", "Degradation 관련"
                      ,"", "시행여부: " + "Train & Valid & Test 과정에 Degradation 시행 안됨"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    
    # [data & model load]--------------------------
    try:
        PATH_ALTER_HR_IMAGE = kargs['PATH_ALTER_HR_IMAGE']
    except:
        PATH_ALTER_HR_IMAGE = None
    
    tmp_is_return_label    = False
    tmp_is_return_image_lr = False
    
    
    if TRAINER_MODE == "SS" or TRAINER_MODE == "SSSR":
        tmp_is_return_label = True
        if HP_AUGM_LITE:
            warnings.warn("라벨 데이터가 사용되는 경우, HP_AUGM_LITE 옵션 사용 불가능")
            HP_AUGM_LITE = False
        save_graph_ss = True
    else:
        save_graph_ss = False
    
    if model_type == "SR_C":
        tmp_is_return_label = True
    
    if TRAINER_MODE == "SR" or TRAINER_MODE == "SSSR":
        tmp_is_return_image_lr = True
        save_graph_sr = True
        try:
            will_niqe_calcurated = kargs['will_niqe_calcurated']    # 학습 초기 이후 Test에서 NIQE 측정여부
        except:
            will_niqe_calcurated = False
    else:
        save_graph_sr = False
        will_niqe_calcurated   = False
    
    
    #@@@<<<
    if tmp_is_return_label:
        HP_COLOR_MAP    = kargs['HP_COLOR_MAP']                         # (dict) gray -> rgb 컬러매핑
        
        HP_CHANNEL_HYPO = kargs['HP_CHANNEL_HYPO']                      # (int) 모델 출력 class (channal) 수
        
        update_dict_v2("", ""
                      ,"", "이미지 채널 수 (라벨): " + str(HP_CHANNEL_GRAY)
                      ,"", "이미지 채널 수 (모델출력물): " + str(HP_CHANNEL_HYPO)
                      ,"", "원본 데이터 라벨 수(void 포함): " + str(HP_LABEL_TOTAL)
                      ,"", "void 라벨 번호: " + str(HP_LABEL_VOID)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
        try:
            HP_LABEL_ONEHOT_ENCODE = kargs['HP_LABEL_ONEHOT_ENCODE']    # (bool) label one-hot encoding 시행여부
        except:
            HP_LABEL_ONEHOT_ENCODE = False
        
        #if TRAINER_MODE == "SSSR":
        #    print("현재 SRSS는 HP_LABEL_ONEHOT_ENCODE가 반드시 True")
        #    HP_LABEL_ONEHOT_ENCODE = True
        
        if HP_LABEL_ONEHOT_ENCODE:
            warnings.warn("Label one-hot encode: 시행함")
            update_dict_v2("", "Label one-hot encode: 시행함"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        else:
            warnings.warn("Label one-hot encode: 시행 안함")
            update_dict_v2("", "Label one-hot encode: 시행 안함"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        
        try:
            HP_LABEL_DILATED = kargs['HP_LABEL_DILATED']                # (bool) label dilation 시행여부
        except:
            HP_LABEL_DILATED = False
        
        if HP_LABEL_DILATED:
            warnings.warn("DILATION for Labels in train: Activated")
            update_dict_v2("", ""
                          ,"", "Label Dilation: 적용됨"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        else:
            warnings.warn("DILATION for Labels in train: Deactivated")
            update_dict_v2("", ""
                          ,"", "Label Dilation: 적용 안됨"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        
        if HP_LABEL_DILATED and not HP_LABEL_ONEHOT_ENCODE:
            _str = "If label is dilated, it must be in one-hot form."
            sys.exit(_str)
        
        if is_force_fix:
            try:
                HP_LABEL_VERIFY             = kargs['HP_LABEL_VERIFY']              # (bool) label 검증 여부
                HP_LABEL_VERIFY_TRY_CEILING = kargs['HP_LABEL_VERIFY_TRY_CEILING']  # (int) 최대 재시도 횟수
                HP_LABEL_VERIFY_CLASS_MIN   = kargs['HP_LABEL_VERIFY_CLASS_MIN']    # (int) 최소 class 종류
                HP_LABEL_VERIFY_RATIO_MAX   = kargs['HP_LABEL_VERIFY_RATIO_MAX']    # (float) 단일 class 최대 비율
            except:
                HP_LABEL_VERIFY             = False
                HP_LABEL_VERIFY_TRY_CEILING = None
                HP_LABEL_VERIFY_CLASS_MIN   = None
                HP_LABEL_VERIFY_RATIO_MAX   = None
            
            if HP_LABEL_VERIFY:
                update_dict_v2("", ""
                              ,"", "Label Verify in train: 적용됨"
                              ,"", "라벨 re-crop 시도 최대횟수: " + str(HP_LABEL_VERIFY_TRY_CEILING)
                              ,"", "라벨 내 유효 class 최소 종류 수: " + str(HP_LABEL_VERIFY_CLASS_MIN)
                              ,"", "라벨 내 최대 class 비율 상한 (0 ~ 1): " + str(HP_LABEL_VERIFY_RATIO_MAX)
                              ,in_dict = dict_log_init
                              ,in_print_head = "dict_log_init"
                              )
            else:
                update_dict_v2("", ""
                              ,"", "Label Verify in train: 적용 안됨"
                              ,in_dict = dict_log_init
                              ,in_print_head = "dict_log_init"
                              )
        
        
        
        
        
    else:
        HP_COLOR_MAP                = None
        HP_CHANNEL_HYPO             = None
        HP_LABEL_ONEHOT_ENCODE      = False
        HP_LABEL_DILATED            = False
        HP_LABEL_VERIFY             = False
        HP_LABEL_VERIFY_TRY_CEILING = None
        HP_LABEL_VERIFY_CLASS_MIN   = None
        HP_LABEL_VERIFY_RATIO_MAX   = None
    #@@@>>>
    
    
    
    # V6 : LR 이미지 생성 안하고 불러와서 씀
    dataset_train = Custom_Dataset_V6(# Return: dict key order -> 'file_name'
                                      #                         , 'pil_img_hr', 'ts_img_hr', 'info_augm'
                                      #                         , 'pil_lab_hr', 'ts_lab_hr'
                                      #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                      name_memo                     = 'train'
                                     ,in_path_dataset               = PATH_BASE_IN
                                     ,in_category                   = NAME_FOLDER_TRAIN
                                     ,in_name_folder_image          = NAME_FOLDER_IMAGES
                                     
                                      #--- options for train 
                                     ,is_train                      = True
                                      # below options can be skipped when above option is False
                                     ,opt_augm_lite                 = HP_AUGM_LITE
                                     ,opt_augm_crop_init_range      = HP_AUGM_RANGE_CROP_INIT
                                     ,opt_augm_rotate_max_degree    = HP_AUGM_ROTATION_MAX
                                     ,opt_augm_prob_flip            = HP_AUGM_PROB_FLIP
                                     ,opt_augm_prob_crop            = HP_AUGM_PROB_CROP
                                     ,opt_augm_prob_rotate          = HP_AUGM_PROB_ROTATE
                                     ,opt_augm_cj_brigntess         = HP_CJ_BRIGHTNESS
                                     ,opt_augm_cj_contrast          = HP_CJ_CONTRAST
                                     ,opt_augm_cj_saturation        = HP_CJ_SATURATION
                                     ,opt_augm_cj_hue               = HP_CJ_HUE
                                     ,opt_augm_random_scaler        = HP_AUGM_RANDOM_SCALER
                                     
                                      #--- options for HR image
                                     ,in_path_alter_hr_image        = PATH_ALTER_HR_IMAGE
                                     
                                      #--- options for HR label
                                     ,is_return_label               = tmp_is_return_label
                                      # below options can be skipped when above option is False
                                     ,in_name_folder_label          = NAME_FOLDER_LABELS
                                     ,label_number_total            = HP_LABEL_TOTAL
                                     ,label_number_void             = HP_LABEL_VOID
                                     ,is_label_dilated              = HP_LABEL_DILATED
                                     
                                     ,is_label_onehot_encode        = HP_LABEL_ONEHOT_ENCODE
                                     
                                     ,is_label_verify               = HP_LABEL_VERIFY
                                      #(선택) if is_label_verify is True
                                     ,label_verify_try_ceiling      = HP_LABEL_VERIFY_TRY_CEILING
                                     ,label_verify_class_min        = HP_LABEL_VERIFY_CLASS_MIN
                                     ,label_verify_ratio_max        = HP_LABEL_VERIFY_RATIO_MAX
                                     
                                      #--- options for LR image
                                     ,is_return_image_lr            = tmp_is_return_image_lr
                                      # below options can be skipped when above option is False
                                     ,scalefactor                   = HP_DG_SCALE_FACTOR
                                     ,in_path_dlc                   = PATH_BASE_IN_SUB
                                     ,in_name_dlc_csv               = HP_DG_CSV_NAME
                                     
                                      #--- increase dataset length
                                     ,in_dataset_loop               = 1
                                     
                                      #--- options for generate patch or force margin
                                     ,is_patch                      = is_use_patch
                                     ,patch_stride                  = HP_PATCH_STRIDES
                                     ,patch_crop_init_range         = HP_PATCH_CROP_INIT_COOR_RANGE
                                     ,model_input_patch_size        = (HP_MODEL_IMG_W, HP_MODEL_IMG_H)              #@@@ check required
                                     
                                     ,is_force_fix                  = is_force_fix
                                     ,force_fix_size_hr             = force_fix_size_hr
                                     
                                      #--- optionas for generate tensor
                                     ,transform_img                 = transform_to_ts_img                           #@@@ check required
                                     )
    
    dataset_val   = Custom_Dataset_V6(# Return: dict key order -> 'file_name'
                                      #                         , 'pil_img_hr', 'ts_img_hr'
                                      #                         , 'pil_lab_hr', 'ts_lab_hr'
                                      #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                      name_memo                     = ' val '
                                     ,in_path_dataset               = PATH_BASE_IN
                                     ,in_category                   = NAME_FOLDER_VAL
                                     ,in_name_folder_image          = NAME_FOLDER_IMAGES
                                     
                                      #--- options for train 
                                     ,is_train                      = False
                                     
                                      #--- options for HR image
                                     ,in_path_alter_hr_image        = PATH_ALTER_HR_IMAGE
                                     
                                      #--- options for HR label
                                     ,is_return_label               = tmp_is_return_label
                                      # below options can be skipped when above option is False
                                     ,in_name_folder_label          = NAME_FOLDER_LABELS
                                     ,label_number_total            = HP_LABEL_TOTAL
                                     ,label_number_void             = HP_LABEL_VOID
                                     ,is_label_dilated              = False
                                     
                                     ,is_label_onehot_encode        = HP_LABEL_ONEHOT_ENCODE
                                     
                                     ,is_label_verify               = False
                                     
                                      #--- options for LR image
                                     ,is_return_image_lr            = tmp_is_return_image_lr
                                      # below options can be skipped when above option is False
                                     ,scalefactor                   = HP_DG_SCALE_FACTOR
                                     ,in_path_dlc                   = PATH_BASE_IN_SUB
                                     ,in_name_dlc_csv               = HP_DG_CSV_NAME
                                     
                                      #--- options for generate patch or force margin
                                     ,is_patch                      = is_use_patch
                                     ,patch_stride                  = HP_PATCH_STRIDES
                                     ,patch_crop_init_range         = HP_PATCH_CROP_INIT_COOR_RANGE
                                     ,model_input_patch_size        = (HP_MODEL_IMG_W, HP_MODEL_IMG_H)              #@@@ check required
                                     
                                     ,is_force_fix                  = is_force_fix
                                     ,force_fix_size_hr             = force_fix_size_hr
                                     
                                      #--- optionas for generate tensor
                                     ,transform_img                 = transform_to_ts_img                           #@@@ check required
                                     )
    
    
    dataset_test  = Custom_Dataset_V6(# Return: dict key order -> 'file_name'
                                      #                         , 'pil_img_hr', 'ts_img_hr'
                                      #                         , 'pil_lab_hr', 'ts_lab_hr'
                                      #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                      name_memo                     = 'test '
                                     ,in_path_dataset               = PATH_BASE_IN
                                     ,in_category                   = NAME_FOLDER_TEST
                                     ,in_name_folder_image          = NAME_FOLDER_IMAGES
                                     
                                      #--- options for train 
                                     ,is_train                      = False
                                     
                                      #--- options for HR image
                                     ,in_path_alter_hr_image        = PATH_ALTER_HR_IMAGE
                                     
                                      #--- options for HR label
                                     ,is_return_label               = tmp_is_return_label
                                      # below options can be skipped when above option is False
                                     ,in_name_folder_label          = NAME_FOLDER_LABELS
                                     ,label_number_total            = HP_LABEL_TOTAL
                                     ,label_number_void             = HP_LABEL_VOID
                                     ,is_label_dilated              = False
                                     
                                     ,is_label_onehot_encode        = HP_LABEL_ONEHOT_ENCODE
                                     
                                     ,is_label_verify               = False
                                     
                                      #--- options for LR image
                                     ,is_return_image_lr            = tmp_is_return_image_lr
                                      # below options can be skipped when above option is False
                                     ,scalefactor                   = HP_DG_SCALE_FACTOR
                                     ,in_path_dlc                   = PATH_BASE_IN_SUB
                                     ,in_name_dlc_csv               = HP_DG_CSV_NAME
                                     
                                      #--- options for generate patch or force margin
                                     ,is_patch                      = False
                                     ,is_force_fix                  = False
                                     
                                      #--- optionas for generate tensor
                                     ,transform_img                 = transform_to_ts_img                           #@@@ check required
                                     )
    
    
    
    #https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    '''
    dataloader_train = torch.utils.data.DataLoader(dataset     = dataset_train
                                                  ,batch_size  = HP_BATCH_TRAIN
                                                  ,shuffle     = True
                                                  ,num_workers = 0
                                                  ,prefetch_factor = 2
                                                  ,drop_last = True
                                                  )
    '''
    
    dataloader_train = DataLoader_multi_worker_FIX(dataset     = dataset_train
                                                  ,batch_size  = HP_BATCH_TRAIN
                                                  ,shuffle     = True
                                                  ,num_workers = HP_NUM_WORKERS
                                                  ,prefetch_factor = 2
                                                  ,drop_last = True
                                                  )
    
    dataloader_val   = torch.utils.data.DataLoader(dataset     = dataset_val
                                                  ,batch_size  = HP_BATCH_VAL
                                                  ,shuffle     = False
                                                  ,num_workers = 0
                                                  ,prefetch_factor = 2
                                                  )
    
    dataloader_test  = DataLoader_multi_worker_FIX(dataset     = dataset_test
                                                  ,batch_size  = HP_BATCH_TEST
                                                  ,shuffle     = False
                                                  ,num_workers = HP_NUM_WORKERS
                                                  ,prefetch_factor = 2
                                                  )
    
    
    
    # [Train & Val & Test]-----------------------------
    
    #<<< Load check_point
    try:
        path_check_point = kargs['path_check_point']
    except:
        path_check_point = "False"
    
    try:
        prev_best = kargs['prev_best']
    except:
        prev_best = None
    
    if path_check_point == "False":
        print("\n<>-=[ Train from Scratch ]=-<>")
        flag_loaded_check_point = False
        loaded_epoch = 0
    else:
        print("\n<>-=[ init check_point loader ]=-<>")
        flag_loaded_check_point = True
        
        try:
            loaded_chech_point = torch.load(path_check_point)
        except:
            print("check_point can not be loaded")
            sys.exit(-1)
        
        loaded_epoch = loaded_chech_point['epoch']
        model.load_state_dict(loaded_chech_point['model_state_dict'])
        optimizer.load_state_dict(loaded_chech_point['optimizer_state_dict'])
        scheduler.load_state_dict(loaded_chech_point['scheduler_state_dict'])
        
        del loaded_chech_point
        
        print("Train will start from epoch:", (loaded_epoch + 1))
        
        for i_epoch in range(loaded_epoch):
            for dataloader_items in dataloader_train:
                print("\rLoading... ", (i_epoch + 1), end = '')
                break
            
    
    count_dataloader = 0
    #>>> Load check_point
    
    print("\nPause before init trainer")
    time.sleep(3)
        
    # 1 epoch 마다 시행할 mode list
    list_mode = ["train", "val", "test"]

    # 학습 전체 기록
    dict_log_total_train = {}
    dict_log_total_val = {}
    dict_log_total_test = {}

    # total log dict의 dict
    dict_dict_log_total = {list_mode[0]: dict_log_total_train
                          ,list_mode[1]: dict_log_total_val
                          ,list_mode[2]: dict_log_total_test
                          }

    for i_key in list_mode:
        if HP_DATASET_NAME == "CamVid":
            tmp_str_labels =  "0(Sky),1(Building),2(Column_pole),3(Road),4(Sidewalk),5(Tree),"
            tmp_str_labels += "6(SignSymbol),7(Fence),8(Car),9(Pedestrian),10(Bicyclist)"
            
        elif HP_DATASET_NAME == "MiniCity":
            tmp_str_labels =  "0(Road),1(Sidewalk),2(Building),3(Wall),4(Fence),"
            tmp_str_labels += "5(Pole),6(Traffic_light),7(Traffic_sign),8(Vegetation),9(Terrain),"
            tmp_str_labels += "10(Sky),11(Person),12(Rider),13(Car),14(Truck),"
            tmp_str_labels += "15(Bus),16(Train),17(Motorcycle),18(Bicycle)"
        
        update_dict_v2("epoch", "loss_(" + i_key + "),mIoU_(" +  i_key + ")," + tmp_str_labels
                      ,in_dict_dict = dict_dict_log_total
                      ,in_dict_key = i_key
                      ,in_print_head = "dict_log_total_" + i_key
                      )
    
    
    #<<< new_record_system
    # 학습 전체 기록
    d_log_total_train = {}
    d_log_total_val   = {}
    d_log_total_test  = {}
    
    # total log dict의 dict
    d_d_log_total = {list_mode[0]: d_log_total_train
                    ,list_mode[1]: d_log_total_val
                    ,list_mode[2]: d_log_total_test
                    }
    
    for i_key in list_mode:
        _str  = "loss_(" + i_key + "),PSNR_(" + i_key + "),SSIM_(" + i_key + "),NIQE_(" + i_key + "),"
        _str += "Pixel_Acc_(" + i_key + "),Class_Acc_(" + i_key + "),mIoU_(" + i_key + "),"
        
        if HP_DATASET_NAME == "CamVid":
            _str += "0(Sky),1(Building),2(Column_pole),3(Road),4(Sidewalk),5(Tree),"
            _str += "6(SignSymbol),7(Fence),8(Car),9(Pedestrian),10(Bicyclist)"
            
        elif HP_DATASET_NAME == "MiniCity":
            _str += "0(Road),1(Sidewalk),2(Building),3(Wall),4(Fence),"
            _str += "5(Pole),6(Traffic_light),7(Traffic_sign),8(Vegetation),9(Terrain),"
            _str += "10(Sky),11(Person),12(Rider),13(Car),14(Truck),"
            _str += "15(Bus),16(Train),17(Motorcycle),18(Bicycle)"
        
        update_dict_v2("epoch", _str
                      ,in_dict_dict = d_d_log_total
                      ,in_dict_key = i_key
                      ,in_print_head = "d_log_total_" + i_key
                      )
    
    #at .update_epoch(), set is_print_sub = True to see epoch info
    
    rb_train_lr   = RecordBox(name = "train_lr",        is_print = False)
    rb_train_loss = RecordBox(name = "train_loss",      is_print = False)
    rb_train_psnr = RecordBox(name = "train_psnr",      is_print = False, will_update_graph = save_graph_sr)
    rb_train_ssim = RecordBox(name = "train_ssim",      is_print = False, will_update_graph = save_graph_sr)
    rb_train_niqe = RecordBox(name = "train_niqe",      is_print = False, will_update_graph = save_graph_sr)
    rb_train_pa   = RecordBox(name = "train_pixel_acc", is_print = False, will_update_graph = save_graph_ss)
    rb_train_ca   = RecordBox(name = "train_class_acc", is_print = False, will_update_graph = save_graph_ss)
    rb_train_ious = RecordBox4IoUs(name = "train_ious", is_print = False, will_update_graph = save_graph_ss)
    
    rb_val_loss   = RecordBox(name = "val_loss",      is_print = False)
    rb_val_psnr   = RecordBox(name = "val_psnr",      is_print = False, will_update_graph = save_graph_sr)
    rb_val_ssim   = RecordBox(name = "val_ssim",      is_print = False, will_update_graph = save_graph_sr)
    rb_val_niqe   = RecordBox(name = "val_niqe",      is_print = False, will_update_graph = save_graph_sr)
    rb_val_pa     = RecordBox(name = "val_pixel_acc", is_print = False, will_update_graph = save_graph_ss)
    rb_val_ca     = RecordBox(name = "val_class_acc", is_print = False, will_update_graph = save_graph_ss)
    rb_val_ious   = RecordBox4IoUs(name = "val_ious", is_print = False, will_update_graph = save_graph_ss)
    
    rb_test_loss  = RecordBox(name = "test_loss",      is_print = False)
    rb_test_psnr  = RecordBox(name = "test_psnr",      is_print = False, will_update_graph = save_graph_sr)
    rb_test_ssim  = RecordBox(name = "test_ssim",      is_print = False, will_update_graph = save_graph_sr)
    rb_test_niqe  = RecordBox(name = "test_niqe",      is_print = False, will_update_graph = save_graph_sr)
    rb_test_pa    = RecordBox(name = "test_pixel_acc", is_print = False, will_update_graph = save_graph_ss)
    rb_test_ca    = RecordBox(name = "test_class_acc", is_print = False, will_update_graph = save_graph_ss)
    rb_test_ious  = RecordBox4IoUs(name = "test_ious", is_print = False, will_update_graph = save_graph_ss)
    
    # PSNR SSIM NIQE 측정기
    # 학습이 완료된 모델 출력 이미지 기준, ignite의 점수와 skimage의 점수는 
    # psnr은 거의 똑같다 (소수점 아래 5번째 자리까지 똑같다.)
    # ssim은 약간 다르다 (소수점 아래 2번째 자리까지 똑같다)
    # 또한, 이미지에 따라서 더 높은 점수가 나오는 평가방식이 다르다 -> 평균은 비슷할 것으로 추측함
    # 연산 효율을 위해 ignite 방식으로 교체하였다.
    #
    # niqe 측정은 cpu 방식밖에 못찾아서 학습 초기 이외엔 test phase 에서만 측정하기로 수정함
    def ignite_eval_step(engine, batch):
        return batch
    ignite_evaluator = ignite.engine.Engine(ignite_eval_step)
    ignite_psnr = ignite.metrics.PSNR(data_range=1.0, device=device)
    ignite_psnr.attach(ignite_evaluator, 'psnr')
    ignite_ssim = ignite.metrics.SSIM(data_range=1.0, device=device)
    ignite_ssim.attach(ignite_evaluator, 'ssim')
    
    calc_niqe = _calc_niqe()         # new niqe method
    #>>> new_record_system
    
    
    dict_2_txt_v2(in_file_path = PATH_OUT_LOG
                 ,in_file_name = "log_init.csv"
                 ,in_dict = dict_log_init
                 )
    
    
    try:
        timer_gpu_all_start  = torch.cuda.Event(enable_timing=True)     # Timer: Batch 당 소요시간 (Batch 준비기간 제외) 
        timer_gpu_all_finish = torch.cuda.Event(enable_timing=True)
        timer_gpu_start  = torch.cuda.Event(enable_timing=True)         # Timer: Model 입출력 + loss & optimizer update 소요시간
        timer_gpu_finish = torch.cuda.Event(enable_timing=True)
        
    except:
        timer_gpu_all_start  = None
        timer_gpu_all_finish = None
        timer_gpu_start  = None
        timer_gpu_finish = None
    
    timer_trainer_start_local = time.mktime(time.localtime())   # trainer 시작 시간 - 종료 예상시간 출력용
    timer_trainer_start = time.time()                           # trainer 시작 시간 - 경과시간 측정용
    timer_epoch = 0                                             # 시간 계산용 epoch 카운터
    
    for i_epoch in range(HP_EPOCH):
        if flag_loaded_check_point:
            i_epoch += loaded_epoch
        
        # train -> val -> test -> train ... 순환
        # epoch 단위 기록
        dict_log_epoch_train = {}
        dict_log_epoch_val = {}
        dict_log_epoch_test = {}
        
        # epoch log dict의 dict
        dict_dict_log_epoch = {list_mode[0]: dict_log_epoch_train
                              ,list_mode[1]: dict_log_epoch_val
                              ,list_mode[2]: dict_log_epoch_test
                              }
        
        #<<< new_record_system
        # epoch 단위 기록
        d_log_epoch_train = {}
        d_log_epoch_val   = {}
        d_log_epoch_test  = {}
        
        # epoch log dict의 dict
        d_d_log_epoch = {list_mode[0]: d_log_epoch_train
                        ,list_mode[1]: d_log_epoch_val
                        ,list_mode[2]: d_log_epoch_test
                        }
        
        #>>> new_record_system
        
        
        for i_mode in list_mode:
            print("--- init", i_mode, "---")
            if i_mode == "train":
                # 종료 예정시간 계산기
                if timer_epoch > 0:
                    _elapsed_time = time.time() - timer_trainer_start
                    try:
                        _estimated_time = (_elapsed_time / timer_epoch) * (HP_EPOCH - loaded_epoch)
                        _tmp = time.localtime(_estimated_time + timer_trainer_start_local)
                        print("Estimated Finish Time:", _tmp.tm_year, "y", _tmp.tm_mon, "m", _tmp.tm_mday, "d  ", _tmp.tm_hour, ":", _tmp.tm_min)
                    except:
                        print("Estimated Finish Time: FAIL")
                    
                    
                    
                timer_epoch = timer_epoch + 1
                
            
            
            if i_mode == "test" and i_epoch > 1:
                if TRAINER_MODE == "SSSR":
                    #tmp_is_best = rb_val_psnr.is_best_max or rb_val_ious.is_best_max
                    tmp_is_best = rb_val_ious.is_best_max
                    if prev_best is not None:
                        # prev best 값이 입력된 경우
                        if prev_best > rb_val_ious.total_max[-1]:
                            tmp_is_best = False
                
                elif TRAINER_MODE == "SR":
                    tmp_is_best = rb_val_psnr.is_best_max
                    if prev_best is not None:
                        # prev best 값이 입력된 경우
                        if prev_best > rb_val_psnr.total_max[-1]:
                            tmp_is_best = False
                
                elif TRAINER_MODE == "SS":
                    tmp_is_best = rb_val_ious.is_best_max
                    if prev_best is not None:
                        # prev best 값이 입력된 경우
                        if prev_best > rb_val_ious.total_max[-1]:
                            tmp_is_best = False
                
                if not tmp_is_best:
                    print("이번 epoch test 생략 ~")
                    continue
            
            
            
            # [공용 변수 초기화] ---
            # 오류 기록용 dict
            dict_log_error = {}
            update_dict_v2("", "오류 기록용 dict"
                          ,in_dict = dict_log_error
                          ,in_print_head = "dict_log_error"
                          ,is_print = False
                          )
            # 오류 발생여부 flag
            flag_error = 0
            
            #<<< #i_mode in list_mode
            # GPU 캐시 메모리 비우기
            torch.cuda.empty_cache()
            
            # 이번 epoch 첫 batch 여부 플래그
            flag_init_epoch = 0
            
            # 현재 batch 번호 (이미지 묶음 단위)
            i_batch = 0
            
            
            # 이번 epoch loss 총 합
            epoch_loss_sum = 0
            # 이번 epoch miou 총 합
            epoch_miou_sum = 0
            
            # epoch log dict 들의 머리글(표 최상단) 설정
            for i_key in list_mode:
                if HP_DATASET_NAME == "CamVid":
                    tmp_str_labels =  "0(Sky),1(Building),2(Column_pole),3(Road),4(Sidewalk),5(Tree),"
                    tmp_str_labels += "6(SignSymbol),7(Fence),8(Car),9(Pedestrian),10(Bicyclist)"
                    
                elif HP_DATASET_NAME == "MiniCity":
                    tmp_str_labels =  "0(Road),1(Sidewalk),2(Building),3(Wall),4(Fence),"
                    tmp_str_labels += "5(Pole),6(Traffic_light),7(Traffic_sign),8(Vegetation),9(Terrain),"
                    tmp_str_labels += "10(Sky),11(Person),12(Rider),13(Car),14(Truck),"
                    tmp_str_labels += "15(Bus),16(Train),17(Motorcycle),18(Bicycle)"
                
                update_dict_v2(i_key + "_"+ str(i_epoch + 1), "batch_num,file_name,loss_batch,miou_image," + tmp_str_labels
                              ,in_dict_dict = dict_dict_log_epoch
                              ,in_dict_key = i_key
                              ,in_print_head = "dict_log_epoch_" + i_key
                              ,is_print = False
                              )
                
                #<<< new_record_system
                #epoch 번호 - batch 번호, 파일 이름, Loss PSRN SSIM NIQE mIoU IoUs
                _str = "Loss,PSNR,SSIM,NIQE,Pixel_Acc,Class_Acc,mIoU,"
                update_dict_v2(i_key + "_"+ str(i_epoch + 1), "batch_num,file_name," + _str + tmp_str_labels
                              ,in_dict_dict = d_d_log_epoch
                              ,in_dict_key = i_key
                              ,in_print_head = "d_log_epoch_" + i_key
                              ,is_print = False
                              )
                
                #>>> new_record_system
                
            # miou 와 라벨별 iou의 유효 개수 카운트 & 누적 합 (tuple로 구성된 dict)
            dict_ious_accumulate = {}
            
            #<<<
            #[모드별 변수 초기화] ---
            if i_mode == "train":
                #현재 모드 batch size 재설정 (생성할 patch 개수를 의미)
                current_batch_size = HP_BATCH_TRAIN
                #dataloader 설정
                dataloader_input = dataloader_train
                #모델 모드 설정 (train / eval)
                model.train()
            elif i_mode == "val":
                #현재 모드 batch size 재설정
                # (trainer_dsrl에선 patch 생성 안함 = 전체 이미지를 단일 patch로 처리 -> 차량전방영상 Segm 특성때문에 patch 사용시 성능이 더 떨어짐)
                current_batch_size = HP_BATCH_VAL
                dataloader_input = dataloader_val
                model.eval()
            elif i_mode == "test":
                #현재 모드 batch size 재설정 (patch 생성 없이 원본 이미지 입력 시행)
                current_batch_size = HP_BATCH_TEST
                dataloader_input = dataloader_test
                model.eval()
            #>>>
            
            #전체 batch 개수
            i_batch_max = len(dataloader_input)
            
            print("\ncurrent_batch_size & total_batch_numbers", current_batch_size, i_batch_max)
            
            
            count_dataloader = 0
            
            # MP 함수용 버퍼 (초기화)
            try:
                del list_mp_buffer
            except:
                pass
            list_mp_buffer = []
            
            if timer_gpu_all_start is not None:
                timer_gpu_all_start.record()
            
            timer_cpu_start = time.time()           # Timer: Batch당 소요시간 Batch 준비기간 포함)
            
            # [train val test 공용 반복 구간] ---
            # x: 입력(이미지), y: 정답(라벨)
            for dataloader_items in dataloader_input:
                count_dataloader += 1
                # 이제 콘솔 출력 epoch 값과 실제 epoch 값이 동일함
                #print("", end = '\r')
                #print("\rin", i_mode, (i_epoch + 1), count_dataloader, "/", i_batch_max, " ", end = '')
                
                dl_str_file_name    = dataloader_items[0]       # (tuple) file name
                
                if i_mode == "test":
                    _bool = bool(set(MUST_SAVE_IMAGE) & set(dl_str_file_name))
                else:
                    _bool = False
                
                is_pil_needed = (i_batch % (i_batch_max//MAX_SAVE_IMAGES + 1) == 0  or _bool )  # pil 이미지가 사용될 경우 (label 제외)
                
                if TRAINER_MODE == "SS":
                    is_niqe_calcurated = False
                else:
                    #will_niqe_calcurated 로 설정입력, is_niqe_calcurated에 반영
                    if i_epoch < 0:
                        is_niqe_calcurated = True
                    elif i_mode == "test":
                        is_niqe_calcurated = will_niqe_calcurated
                    else:
                        is_niqe_calcurated = False
                
                # augm info for train
                dl_str_info_augm    = dataloader_items[3]
                
                # degraded info for LR images
                try:
                    dl_str_info_deg = dataloader_items[9] #@@@
                    if dl_str_info_deg[0] == False:
                        dl_str_info_deg = None
                except:
                    dl_str_info_deg = None
                
                # PIL (HR image & HR label & LR image) -> Full or Patch ver of RAW PIL (no Norm)
                if is_pil_needed:
                    dl_pil_img_hr       = tensor_2_list_pils_v1(in_tensor = dataloader_items[1])
                    try:
                        dl_pil_img_lr   = tensor_2_list_pils_v1(in_tensor = dataloader_items[7]) #@@@
                    except:
                        dl_pil_img_lr   = None
                else:
                    dl_pil_img_hr = None
                    dl_pil_img_lr = None
                
                try:
                    #gray 이미지 형태 [B, 1, H, W] -> mIoU 측정용으로만 사용 예정, (0~1) -> (0~255)를 위해 x255 후 반올림 시행
                    dl_ts_lab_hr_gray = torch.round(dataloader_items[4]*255).type(torch.uint8)
                except:
                    dl_ts_lab_hr_gray = None
                
                if is_pil_needed:
                    try:
                        # CamVid 기준 단순히 (Gray 이미지의) PIL -> Image Tensor -> PIL 작업이라 옵션지정 필요없음
                        dl_pil_lab_hr   = tensor_2_list_pils_v1(in_tensor = dataloader_items[4])
                    except:
                        dl_pil_lab_hr     = None
                
                # Tensor (HR image & HR label & LR image)
                dl_ts_img_hr        = dataloader_items[2].float()
                try:
                    dl_ts_lab_hr    = dataloader_items[5].float()   # class 만큼 채널 늘어난 형태 -> Dilation 적용될 수 있음
                    dl_ts_lab_hr_void = dataloader_items[6].float() #@@@
                except:
                    dl_ts_lab_hr    = None
                try:
                    dl_ts_img_lr    = dataloader_items[8].float() #@@@
                except:
                    dl_ts_img_lr    = None
                
                
                
                if i_mode == "train":
                    if TRAINER_MODE == "SSSR" or TRAINER_MODE == "SR":
                        dl_ts_img_lr = dl_ts_img_lr.requires_grad_(True)
                    elif TRAINER_MODE == "SS":
                        dl_ts_img_hr = dl_ts_img_hr.requires_grad_(True)
                
                dl_ts_img_hr = dl_ts_img_hr.to(device)
                if dl_ts_lab_hr is not None:
                    dl_ts_lab_hr = dl_ts_lab_hr.to(device)
                if dl_ts_img_lr is not None:
                    dl_ts_img_lr = dl_ts_img_lr.to(device)
                if dl_ts_lab_hr_gray is not None:
                    dl_ts_lab_hr_gray = dl_ts_lab_hr_gray.to(device)
                
                if i_mode == "train":
                    if i_batch == 0:
                        # 기울기 초기화
                        optimizer.zero_grad()
                        print("optimizer.zero_grad()")
                    
                    if timer_gpu_start is not None:
                        timer_gpu_start.record()
                    
                    with torch.cuda.amp.autocast(enabled=True):
                        #<<< AMP
                        in_batch_size, _, _, _ = dl_ts_img_hr.shape
                        if i_mode == "train" and in_batch_size != current_batch_size:
                            print("Batch size is not same with HyperParameter:", in_batch_size, current_batch_size)
                            sys.exit(-1)
                        
                        if TRAINER_MODE == "SSSR":
                            #<<< SSSR - TRAIN
                            if model_type == "SSSR_A" or model_type == "SSSR_B":
                                # (tensor) Segm 결과, SR 결과, Restored 결과
                                tensor_out_seg, tensor_out_sr, tensor_out_lr = model(dl_ts_img_lr)
                            elif model_type == "SSSR_C":
                                # (tensor) Segm 결과, SR 결과, Restored 결과 1,2
                                tensor_out_seg, tensor_out_sr, tensor_out_lr_1, tensor_out_lr_2 = model(dl_ts_img_lr)
                            elif model_type == "SSSR_D":
                                # Basic Block: 3개
                                tensor_out_lr_1, tensor_out_lr_2, tensor_out_sr, tensor_out_seg = model(dl_ts_img_lr)
                                tensor_out_if = tensor_out_lr_2
                            
                            # label 예측결과 softmax 시행
                            tensor_out_seg_softmax = F.softmax(tensor_out_seg, dim = 1)
                            # softmax 값을 바탕으로 label image 형태로 tensor 생성 (형태 변경 4D [B, C, H, W] -> 3D [B, H, W]) -> 이미지 생성에 사용됨
                            tensor_out_seg_label = torch.argmax(tensor_out_seg_softmax.clone().detach().requires_grad_(False), dim = 1)
                            
                            # loss 계산
                            if model_type == "SSSR_A":
                                loss = criterion.calc_v3(torch.clamp(tensor_out_seg_softmax - dl_ts_lab_hr_void.to(device), 0, 1)
                                                        ,tensor_out_sr
                                                        ,dl_ts_lab_hr
                                                        ,dl_ts_img_hr
                                                        ,is_AMP = True
                                                        #,cfa_sub_sample_rate = 8
                                                        )
                            elif model_type == "SSSR_B":
                                loss = criterion.calc_v4(torch.clamp(tensor_out_seg_softmax - dl_ts_lab_hr_void.to(device), 0, 1)
                                                        ,tensor_out_sr
                                                        ,dl_ts_lab_hr
                                                        ,dl_ts_img_hr
                                                        ,tensor_out_lr
                                                        ,is_AMP = True
                                                        #,cfa_sub_sample_rate = 8
                                                        )
                            elif model_type == "SSSR_C":
                                loss = criterion.calc_v5(torch.clamp(tensor_out_seg_softmax - dl_ts_lab_hr_void.to(device), 0, 1)
                                                        ,tensor_out_sr
                                                        ,dl_ts_lab_hr
                                                        ,dl_ts_img_hr
                                                        ,tensor_out_lr_1
                                                        ,tensor_out_lr_2
                                                        ,is_AMP = True
                                                        #,cfa_sub_sample_rate = 8
                                                        )
                            elif model_type == "SSSR_D":
                                # Basic Block: 3개
                                if HP_LABEL_ONEHOT_ENCODE:
                                    # one-hot encoding 시행
                                    loss = criterion.calc_v6(torch.clamp(tensor_out_seg_softmax - dl_ts_lab_hr_void.to(device), 0, 1)
                                                            ,tensor_out_sr
                                                            ,dl_ts_lab_hr
                                                            ,dl_ts_img_hr
                                                            ,pred_img_1 = tensor_out_lr_1
                                                            ,pred_img_2 = tensor_out_lr_2
                                                            ,is_AMP     = True
                                                            ,is_onehot  = True
                                                            )
                                else:
                                    # one-hot encoding 시행 안함
                                    _dl_ts_lab_hr_gray = dl_ts_lab_hr_gray.clone().detach()
                                    _dl_ts_lab_hr_gray = torch.squeeze(_dl_ts_lab_hr_gray, dim=1)
                                    _dl_ts_lab_hr_gray = _dl_ts_lab_hr_gray.type(torch.long)
                                    _dl_ts_lab_hr_gray = _dl_ts_lab_hr_gray.to(device)
                                    #loss = criterion(tensor_out_seg_softmax,  _dl_ts_lab_hr_gray)
                                    loss = criterion.calc_v6(tensor_out_seg_softmax
                                                            ,tensor_out_sr
                                                            ,_dl_ts_lab_hr_gray
                                                            ,dl_ts_img_hr
                                                            ,pred_img_1 = tensor_out_lr_1
                                                            ,pred_img_2 = tensor_out_lr_2
                                                            ,is_AMP     = True
                                                            ,is_onehot  = False
                                                            )
                            
                            
                            # SR 이미지텐서 역 정규화
                            if is_norm_in_transform_to_tensor:
                                tensor_out_sr = transform_ts_inv_norm(tensor_out_sr)
                            
                            #>>> SSSR - TRAIN
                            
                        elif TRAINER_MODE == "SR":
                            #<<< SR - TRAIN
                            
                            # Model 예측결과 생성 & Loss 계산
                            if model_type == "SR_A":
                                # (tensor) SR 결과 [stage 3, 2, 1]
                                tensor_out_sr_set = model(dl_ts_img_lr)
                                tensor_out_sr = tensor_out_sr_set[0]
                                loss = criterion(tensor_out_sr_set, dl_ts_img_hr)
                                
                            elif model_type == "SR_B":
                                tensor_out_sr = model(dl_ts_img_lr)
                                loss = criterion(tensor_out_sr, dl_ts_img_hr)
                            
                            elif model_type == "SR_C":
                                tensor_out_sr = model(dl_ts_img_lr)
                                loss = criterion(tensor_out_sr, dl_ts_img_hr, dl_ts_lab_hr)
                            
                            # SR 이미지텐서 역 정규화
                            if is_norm_in_transform_to_tensor:
                                tensor_out_sr = transform_ts_inv_norm(tensor_out_sr)
                            #>>> SR - TRAIN
                            
                        elif TRAINER_MODE == "SS":
                            #<<< SS - TRAIN
                            # Model 예측결과 생성 & Loss 계산 (모델 입력: HR 이미지)
                            if model_type == "SS_A":
                                tensor_out_seg = model(dl_ts_img_hr)
                            
                            # label 예측결과 softmax 시행
                            tensor_out_seg_softmax = F.softmax(tensor_out_seg, dim = 1)
                            # softmax 값을 바탕으로 label image 형태로 tensor 생성 (형태 변경 4D [B, C, H, W] -> 3D [B, H, W]) -> 이미지 생성에 사용됨
                            tensor_out_seg_label = torch.argmax(tensor_out_seg_softmax.clone().detach().requires_grad_(False), dim = 1)
                            
                            # loss 계산
                            if model_type == "SS_A":
                                if HP_LABEL_ONEHOT_ENCODE:
                                    # one-hot encoding 시행
                                    loss = criterion(torch.clamp(tensor_out_seg_softmax - dl_ts_lab_hr_void.to(device), 0, 1), dl_ts_lab_hr)
                                else:
                                    # one-hot encoding 시행 안함
                                    _dl_ts_lab_hr_gray = dl_ts_lab_hr_gray.clone().detach()
                                    _dl_ts_lab_hr_gray = torch.squeeze(_dl_ts_lab_hr_gray, dim=1)
                                    _dl_ts_lab_hr_gray = _dl_ts_lab_hr_gray.type(torch.long)
                                    _dl_ts_lab_hr_gray = _dl_ts_lab_hr_gray.to(device)
                                    loss = criterion(tensor_out_seg_softmax,  _dl_ts_lab_hr_gray)
                            
                            #>>> SS - TRAIN @@@
                        
                        
                        
                        #>>> AMP
                    
                    
                    batch_loss = loss.item()
                    epoch_loss_sum += batch_loss
                    
                    #<<< new_record_system
                    rb_train_loss.add_item(loss.item())
                    #>>> new_record_system
                    
                    try:
                        # loss overflow, underflow 오류 방지
                        if HP_DETECT_LOSS_ANOMALY:
                            with torch.autograd.detect_anomaly():
                                amp_scaler.scale(loss).backward(retain_graph=False)
                                if is_gradient_clipped: 
                                    amp_scaler.unscale_(optimizer)
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
                                #optimizer.step()  # 가중치 갱신 (batch 마다)
                                amp_scaler.step(optimizer)
                                #print("optimizer.step()", i_batch)
                                amp_scaler.update()
                        else:
                            amp_scaler.scale(loss).backward(retain_graph=False)
                            if is_gradient_clipped: 
                                amp_scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
                            amp_scaler.step(optimizer)
                            amp_scaler.update()
                        
                    except:
                        flag_error += 1
                        update_dict_v2("", "in " + str(i_epoch) + " " + str(i_batch)
                                      ,"", "loss.backward() 실패: " + str(loss.item())
                                      ,in_dict = dict_log_error
                                      ,in_print_head = "dict_log_error"
                                      ,is_print = False
                                      )
                    
                    if timer_gpu_finish is not None:
                        timer_gpu_finish.record()
                    
                    # 기울기 초기화
                    optimizer.zero_grad()
                    #print("optimizer.zero_grad()")
                    if HP_SCHEDULER_UPDATE_INTERVAL == "batch":
                        # 사용된 lr 기록 
                        rb_train_lr.add_item(optimizer.param_groups[0]['lr'])
                        rb_train_lr.update_batch()
                        # 스케쥴러 갱신
                        scheduler.step()
                        print("scheduler.step()")
                    
                
                else: # val or test
                    if timer_gpu_start is not None:
                        timer_gpu_start.record()
                    
                    with torch.no_grad():
                        #<<< no_grad
                        if TRAINER_MODE == "SSSR":
                            #<<< SSSR - Val & Test
                            if model_type == "SSSR_A" or model_type == "SSSR_B":
                                # (tensor) Segm 결과, SR 결과, Restored 결과
                                tensor_out_seg, tensor_out_sr, tensor_out_lr = model(dl_ts_img_lr)
                            elif model_type == "SSSR_C":
                                # (tensor) Segm 결과, SR 결과, Restored 결과 1,2
                                tensor_out_seg, tensor_out_sr, tensor_out_lr_1, tensor_out_lr_2 = model(dl_ts_img_lr)
                            elif model_type == "SSSR_D":
                                # Basic Block: 3개
                                tensor_out_lr_1, tensor_out_lr_2, tensor_out_sr, tensor_out_seg = model(dl_ts_img_lr)
                                tensor_out_if = tensor_out_lr_2
                            
                            #label 예측결과 softmax 시행
                            tensor_out_seg_softmax = F.softmax(tensor_out_seg, dim = 1)
                            # softmax 값을 바탕으로 label image 형태로 tensor 생성 (형태 변경 4D [B, C, H, W] -> 3D [B, H, W]) -> 이미지 생성에 사용됨
                            tensor_out_seg_label = torch.argmax(tensor_out_seg_softmax.clone().detach().requires_grad_(False), dim = 1)
                            
                            # loss 계산
                            if model_type == "SSSR_A":
                                loss = criterion.calc_v3(torch.clamp(tensor_out_seg_softmax - dl_ts_lab_hr_void.to(device), 0, 1)
                                                        ,tensor_out_sr
                                                        ,dl_ts_lab_hr
                                                        ,dl_ts_img_hr
                                                        ,is_AMP = True
                                                        #,cfa_sub_sample_rate = 8
                                                        )
                            elif model_type == "SSSR_B":
                                loss = criterion.calc_v4(torch.clamp(tensor_out_seg_softmax - dl_ts_lab_hr_void.to(device), 0, 1)
                                                        ,tensor_out_sr
                                                        ,dl_ts_lab_hr
                                                        ,dl_ts_img_hr
                                                        ,tensor_out_lr
                                                        ,is_AMP = True
                                                        #,cfa_sub_sample_rate = 8
                                                        )
                            elif model_type == "SSSR_C":
                                loss = criterion.calc_v5(torch.clamp(tensor_out_seg_softmax - dl_ts_lab_hr_void.to(device), 0, 1)
                                                        ,tensor_out_sr
                                                        ,dl_ts_lab_hr
                                                        ,dl_ts_img_hr
                                                        ,tensor_out_lr_1
                                                        ,tensor_out_lr_2
                                                        ,is_AMP = True
                                                        #,cfa_sub_sample_rate = 8
                                                        )
                            elif model_type == "SSSR_D":
                                # Basic Block: 3개
                                if HP_LABEL_ONEHOT_ENCODE:
                                    # one-hot encoding 시행
                                    loss = criterion.calc_v6(torch.clamp(tensor_out_seg_softmax - dl_ts_lab_hr_void.to(device), 0, 1)
                                                            ,tensor_out_sr
                                                            ,dl_ts_lab_hr
                                                            ,dl_ts_img_hr
                                                            ,pred_img_1 = tensor_out_lr_1
                                                            ,pred_img_2 = tensor_out_lr_2
                                                            ,is_AMP     = True
                                                            ,is_onehot  = True
                                                            )
                                else:
                                    # one-hot encoding 시행 안함
                                    _dl_ts_lab_hr_gray = dl_ts_lab_hr_gray.clone().detach()
                                    _dl_ts_lab_hr_gray = torch.squeeze(_dl_ts_lab_hr_gray, dim=1)
                                    _dl_ts_lab_hr_gray = _dl_ts_lab_hr_gray.type(torch.long)
                                    _dl_ts_lab_hr_gray = _dl_ts_lab_hr_gray.to(device)
                                    #loss = criterion(tensor_out_seg_softmax,  _dl_ts_lab_hr_gray)
                                    loss = criterion.calc_v6(tensor_out_seg_softmax
                                                            ,tensor_out_sr
                                                            ,_dl_ts_lab_hr_gray
                                                            ,dl_ts_img_hr
                                                            ,pred_img_1 = tensor_out_lr_1
                                                            ,pred_img_2 = tensor_out_lr_2
                                                            ,is_AMP     = True
                                                            ,is_onehot  = False
                                                            )
                            
                            
                            # SR 이미지텐서 역 정규화
                            if is_norm_in_transform_to_tensor:
                                tensor_out_sr = transform_ts_inv_norm(tensor_out_sr)
                            
                            #>>> SSSR - Val & Test
                        
                        elif TRAINER_MODE == "SR":
                            #<<< SR - Val & Test
                            
                            # Model 예측결과 생성 & Loss 계산
                            if model_type == "SR_A":
                                # (tensor) SR 결과 [stage 3, 2, 1]
                                tensor_out_sr_set = model(dl_ts_img_lr)
                                tensor_out_sr = tensor_out_sr_set[0]
                                loss = criterion(tensor_out_sr_set, dl_ts_img_hr)
                                
                            elif model_type == "SR_B":
                                tensor_out_sr = model(dl_ts_img_lr)
                                loss = criterion(tensor_out_sr, dl_ts_img_hr)
                            
                            elif model_type == "SR_C":
                                tensor_out_sr = model(dl_ts_img_lr)
                                loss = criterion(tensor_out_sr, dl_ts_img_hr, dl_ts_lab_hr)
                            
                            # SR 이미지텐서 역 정규화
                            if is_norm_in_transform_to_tensor:
                                tensor_out_sr = transform_ts_inv_norm(tensor_out_sr)
                            #>>> SR - Val & Test
                        
                        elif TRAINER_MODE == "SS":
                            #<<< SS - Val & Test
                            # Model 예측결과 생성 & Loss 계산 (모델 입력: HR 이미지)
                            if model_type == "SS_A":
                                tensor_out_seg = model(dl_ts_img_hr)
                            
                            # label 예측결과 softmax 시행
                            tensor_out_seg_softmax = F.softmax(tensor_out_seg, dim = 1)
                            # softmax 값을 바탕으로 label image 형태로 tensor 생성 (형태 변경 4D [B, C, H, W] -> 3D [B, H, W]) -> 이미지 생성에 사용됨
                            tensor_out_seg_label = torch.argmax(tensor_out_seg_softmax.clone().detach().requires_grad_(False), dim = 1)
                            
                            # loss 계산
                            if model_type == "SS_A":
                                if HP_LABEL_ONEHOT_ENCODE:
                                    # one-hot encoding 시행
                                    loss = criterion(torch.clamp(tensor_out_seg_softmax - dl_ts_lab_hr_void.to(device), 0, 1), dl_ts_lab_hr)
                                else:
                                    # one-hot encoding 시행 안함
                                    _dl_ts_lab_hr_gray = dl_ts_lab_hr_gray.clone().detach()
                                    _dl_ts_lab_hr_gray = torch.squeeze(_dl_ts_lab_hr_gray, dim=1)
                                    _dl_ts_lab_hr_gray = _dl_ts_lab_hr_gray.type(torch.long)
                                    _dl_ts_lab_hr_gray = _dl_ts_lab_hr_gray.to(device)
                                    loss = criterion(tensor_out_seg_softmax,  _dl_ts_lab_hr_gray)
                            
                            #>>> SS - Val & Test
                        
                        batch_loss = loss.item()
                        epoch_loss_sum += batch_loss
                        
                        #<<< new_record_system
                        if i_mode == "val":
                            rb_val_loss.add_item(loss.item())
                        elif i_mode == "test":
                            rb_test_loss.add_item(loss.item())
                        #>>> new_record_system
                        
                        #>>> no_grad
                    
                    if timer_gpu_finish is not None:
                        timer_gpu_finish.record()
                
                
                
                #VVV [Tensor -> model 예측결과 생성] -----------------------
                
                #AAA [Output Tensor -> PIL 이미지] -----------------------
                
                if TRAINER_MODE == "SSSR":
                    #<<< SSSR
                    if is_pil_needed:
                        list_out_pil_label = tensor_2_list_pils_v1(# 텐서 -> pil 이미지 리스트
                                                                   # (tensor) 변환할 텐서, 모델에서 다중 결과물이 생성되는 경우, 단일 출력물 묶음만 지정해서 입력 
                                                                   # (예: MPRNet -> in_tensor = tensor_sr_hypo[0])
                                                                   in_tensor = tensor_out_seg_label
                                                                   
                                                                   # (bool) 라벨 여부 (출력 pil 이미지 = 3ch, 라벨 = 1ch 고정) (default: False)
                                                                  ,is_label = True
                                                                   
                                                                   # (bool) pil 이미지 크기 변환 시행여부 (default: False)
                                                                  ,is_resized = False
                                                                  )
                    else:
                        list_out_pil_label = None
                    
                    if is_pil_needed or is_niqe_calcurated: 
                        list_out_pil_sr = tensor_2_list_pils_v1(in_tensor = tensor_out_sr
                                                               ,is_label = False
                                                               ,is_resized = False
                                                               )
                    else:
                        list_out_pil_sr = None
                    
                    if model_type == "SSSR_A" or model_type == "SSSR_B":
                        if is_pil_needed:
                            list_out_pil_lr = tensor_2_list_pils_v1(in_tensor = tensor_out_lr
                                                                   ,is_label = False
                                                                   ,is_resized = False
                                                                   )
                        else:
                            list_out_pil_lr = None
                    elif model_type == "SSSR_C":
                        if is_pil_needed:
                            list_out_pil_lr = tensor_2_list_pils_v1(in_tensor = tensor_out_lr_2
                                                                   ,is_label = False
                                                                   ,is_resized = False
                                                                   )
                        else:
                            list_out_pil_lr = None
                    elif model_type == "SSSR_D":
                        if is_pil_needed:
                            list_out_pil_lr = tensor_2_list_pils_v1(in_tensor = tensor_out_if
                                                                   ,is_label = False
                                                                   ,is_resized = False
                                                                   )
                        else:
                            list_out_pil_lr = None
                    
                    #>>> SSSR
                
                elif TRAINER_MODE == "SR":
                    #<<< SR
                    # MPRNet ESRT HAN IMDN 공용
                    if is_pil_needed or is_niqe_calcurated: 
                        list_out_pil_sr = tensor_2_list_pils_v1(in_tensor = tensor_out_sr
                                                               ,is_label = False
                                                               ,is_resized = False
                                                               )
                    else:
                        list_out_pil_sr = None
                    
                    #>>> SR
                
                elif TRAINER_MODE == "SS":
                    #<<< SS
                    if is_pil_needed:
                        list_out_pil_label = tensor_2_list_pils_v1(in_tensor = tensor_out_seg_label
                                                                  ,is_label = True
                                                                  ,is_resized = False
                                                                  )
                    else:
                        list_out_pil_label = None
                    
                    #>>> SS
                
                
                
                #VVV [Output Tensor -> PIL 이미지] -----------------------
                
                #AAA [batch 단위 이미지 평가 (Pixel Acc, Class Acc, mIoU, PSNR, SSIM, NIQE)] --------------------------
                # batch 이미지 들의 miou 누적변수
                batch_miou_sum = 0
                
                for i_image in range(current_batch_size):
                    
                    # 입력 x LR 이미지
                    # list_patch_pil_x[i_image] -> dl_pil_img_lr[i_image]
                    
                    # 입력 y HR 라벨 
                    # list_patch_pil_y[i_image] -> dl_pil_lab_hr[i_image]
                    
                    #라벨 예측결과를 원본 라벨 크기로 변환 ->  크기변환 없음
                    #pil_hypo_resized = list_out_pil_label[i_image]
                    
                    # Pixel Acc, Class Acc, mIoU 계산 (batch단위 평균값) (mIoU 연산에 사용한 이미지 수 = in_batch_size)
                    try:
                        if TRAINER_MODE == "SSSR" or TRAINER_MODE == "SS":
                            tmp_pa, tmp_ca, tmp_miou, dict_ious = calc_pa_ca_miou_gray_tensor(ts_ans    = dl_ts_lab_hr_gray[i_image][0]
                                                                                             ,ts_pred   = tensor_out_seg_label[i_image]
                                                                                             ,int_total = HP_LABEL_TOTAL
                                                                                             ,int_void  = HP_LABEL_VOID
                                                                                             ,device    = device
                                                                                             )
                            
                        else:
                            tmp_pa, tmp_ca, tmp_miou, dict_ious = dummy_calc_pa_ca_miou_gray_tensor(ts_ans    = None
                                                                                                   ,ts_pred   = None
                                                                                                   ,int_total = HP_LABEL_TOTAL
                                                                                                   ,int_void  = HP_LABEL_VOID
                                                                                                   )
                        
                    except:
                        print("(exc) Pixel Acc, Class Acc, mIoU 측정 실패")
                        tmp_pa, tmp_ca, tmp_miou, dict_ious = dummy_calc_pa_ca_miou_gray_tensor(ts_ans    = None
                                                                                               ,ts_pred   = None
                                                                                               ,int_total = HP_LABEL_TOTAL
                                                                                               ,int_void  = HP_LABEL_VOID
                                                                                               )
                    
                    batch_miou_sum += tmp_miou
                    
                    #<<< new_record_system
                    if i_mode == "train":
                        rb_train_pa.add_item(tmp_pa)
                        rb_train_ca.add_item(tmp_ca)
                        rb_train_ious.add_item(dict_ious)
                    elif i_mode == "val":
                        rb_val_pa.add_item(tmp_pa)
                        rb_val_ca.add_item(tmp_ca)
                        rb_val_ious.add_item(dict_ious)
                    elif i_mode == "test":
                        rb_test_pa.add_item(tmp_pa)
                        rb_test_ca.add_item(tmp_ca)
                        rb_test_ious.add_item(dict_ious)
                    #>>> new_record_system
                    
                    #--- PSNR SSIM NIQE
                    if TRAINER_MODE == "SSSR" or TRAINER_MODE == "SR":
                        try:
                            
                            # 이전 방식 = pil 이미지로 calc_psnr_ssim 사용
                            
                            #<<< ignite
                            with torch.no_grad():
                                if i_mode == "train":
                                    ignite_in_sr = torch.unsqueeze(tensor_out_sr[i_image].to(torch.float32), 0)
                                else:   # val & test
                                    ignite_in_sr = torch.unsqueeze(tensor_out_sr[i_image], 0)
                                
                                ignite_in_hr = torch.unsqueeze(dl_ts_img_hr[i_image], 0)
                                
                                ignite_in_sr = ignite_in_sr.to(device)
                                ignite_in_hr = ignite_in_hr.to(device)
                                
                                ignite_result = ignite_evaluator.run([[ignite_in_sr
                                                                      ,ignite_in_hr
                                                                    ]])
                                
                                out_psnr = ignite_result.metrics['psnr']
                                out_ssim = ignite_result.metrics['ssim']
                                
                            #>>> ignite
                        except:
                            print("(exc) PSRN SSIM calc FAIL")
                            out_psnr, out_ssim = -9, -9
                        
                        
                        if is_niqe_calcurated:
                            try:
                                out_niqe = calc_niqe.with_pil(list_out_pil_sr[i_image])
                            except:
                                print("(exc) NIQE calc FAIL")
                                out_niqe = -9
                            
                        else:
                            out_niqe = -9
                        
                    else:
                        out_psnr, out_ssim, out_niqe = -9, -9, -9
                    
                    #<<< new_record_system
                    if i_mode == "train":
                        rb_train_psnr.add_item(out_psnr)
                        rb_train_ssim.add_item(out_ssim)
                        rb_train_niqe.add_item(out_niqe)
                    elif i_mode == "val":
                        rb_val_psnr.add_item(out_psnr)
                        rb_val_ssim.add_item(out_ssim)
                        rb_val_niqe.add_item(out_niqe)
                    elif i_mode == "test":
                        rb_test_psnr.add_item(out_psnr)
                        rb_test_ssim.add_item(out_ssim)
                        rb_test_niqe.add_item(out_niqe)
                    
                    #>>> new_record_system
                    
                    
                    
                    # 이미지 단위 로그 갱신
                    # "","batch_num,file_name,loss_batch,miou_image,[라벨별 iou]"
                    tmp_file_name = dl_str_file_name[i_image]
                    
                    tmp_ious = ""
                    for i_key in dict_ious:
                        tmp_ious += "," + dict_ious[i_key]
                    
                    
                    update_dict_v2("", str(count_dataloader) + "," + tmp_file_name + "," + str(batch_loss) + "," + str(tmp_miou) + tmp_ious
                                  ,in_dict_dict = dict_dict_log_epoch
                                  ,in_dict_key = i_mode
                                  ,in_print_head = "dict_log_epoch_" + i_mode
                                  ,is_print = False
                                  )
                    
                    #<<< new_record_system
                    #epoch 번호 - batch 번호, 파일 이름, Loss PSRN SSIM NIQE mIoU IoUs
                    
                    tmp_str_contents = (str(count_dataloader) + "," + tmp_file_name + "," + str(batch_loss) 
                                       +"," + str(out_psnr) + "," + str(out_ssim) + "," + str(out_niqe)
                                       +"," + str(tmp_pa) + "," + str(tmp_ca) +"," + str(tmp_miou) + tmp_ious
                                       )
                    
                    update_dict_v2("", tmp_str_contents
                                  ,in_dict_dict = d_d_log_epoch
                                  ,in_dict_key = i_mode
                                  ,in_print_head = "d_log_epoch_" + i_mode
                                  ,is_print = False
                                  )
                    
                    #>>> new_record_system
                    
                    # 이미지별 miou & ious 누적기록 업데이트
                    accumulate_dict_ious(dict_ious_accumulate, tmp_miou, dict_ious)
                    
                    #<<< 예측결과를 이미지로 생성
                    
                    #print("debug", dl_str_file_name[i_image])
                    
                    if i_mode == "train":
                        if is_pil_needed:
                            # epoch 마다 n 배치 정도의 결과 이미지를 저장해봄
                            plt_title = "File name: " + dl_str_file_name[i_image]
                            plt_title += "\n" + dl_str_info_augm[i_image]
                            if option_apply_degradation and dl_str_info_deg is not None:
                                # 현재 patch의 degrad- 옵션 불러오기
                                plt_title += "\n" + dl_str_info_deg[i_image]
                            plt_title += "\nPSNR: " + str(round(out_psnr, 4))
                            plt_title += "  SSIM: " + str(round(out_ssim, 4))
                            plt_title += "  NIQE: " + str(round(out_niqe, 4))
                            plt_title += "\nmIoU: " + str(round(tmp_miou, 4))
                            
                            is_plt_will_saved = True
                        else:
                            is_plt_will_saved = False
                    
                    elif REDUCE_SAVE_IMAGES:  # Val & Test phase with reduced image saves
                        if is_pil_needed:
                            # epoch 마다 n 배치 정도의 결과 이미지를 저장해봄
                            plt_title = "File name: " + dl_str_file_name[i_image]
                            if option_apply_degradation and dl_str_info_deg is not None:
                                # 현재 patch의 degrad- 옵션 불러오기
                                plt_title += "\n" + dl_str_info_deg[i_image]
                            
                            plt_title += "\nPSNR: " + str(round(out_psnr, 4))
                            plt_title += "  SSIM: " + str(round(out_ssim, 4))
                            plt_title += "  NIQE: " + str(round(out_niqe, 4))
                            plt_title += "\nmIoU: " + str(round(tmp_miou, 4))
                            is_plt_will_saved = True
                        else:
                            is_plt_will_saved = False
                    
                    
                    else: # val or test
                        is_pil_needed = True
                        # 모든 이미지를 저장함
                        plt_title = "File name: " + dl_str_file_name[i_image]
                        if option_apply_degradation and dl_str_info_deg is not None:
                            # 현재 patch의 degrad- 옵션 불러오기
                            plt_title += "\n" + dl_str_info_deg[i_image]
                        
                        plt_title += "\nPSNR: " + str(round(out_psnr, 4))
                        plt_title += "  SSIM: " + str(round(out_ssim, 4))
                        plt_title += "  NIQE: " + str(round(out_niqe, 4))
                        plt_title += "\nmIoU: " + str(round(tmp_miou, 4))
                        is_plt_will_saved = True
                    
                    
                    # 결과 이미지 저장
                    if is_plt_will_saved:
                        tmp_file_name = (i_mode + "_" + str(i_epoch + 1) + "_" + str(i_batch + 1) + "_"
                                        +dl_str_file_name[i_image]
                                        )
                        if TRAINER_MODE == "SSSR":
                            #<<< SSSR
                            if model_type == "SSSR_A" or model_type == "SSSR_B" or model_type == "SSSR_C" or model_type == "SSSR_D":
                                
                                if len(list_mp_buffer) >= BUFFER_SIZE:
                                    # chunk full -> toss mp_buffer -> empty mp_buffer
                                    if i_mode == 'test':
                                        tmp_is_best = rb_val_ious.is_best_max   #chunk 단위 buffer 구조상 valid 기준으로 best여부 검사
                                    else:
                                        tmp_is_best = False
                                    plts_saver_sssr(list_mp_buffer, is_best = tmp_is_best, no_employ=(employ_threshold >= len(list_mp_buffer)))
                                    
                                    try:
                                        del list_mp_buffer
                                    except:
                                        pass
                                    list_mp_buffer = []
                                    
                                if i_image < 2: # batch 당 최대 2장 저장
                                    list_mp_buffer.append((# 0 (model name)
                                                           model_type
                                                           # 1 ~ 6 (pils)
                                                          ,dl_pil_img_hr[i_image], label_2_RGB(dl_pil_lab_hr[i_image], HP_COLOR_MAP)
                                                          ,label_2_RGB(list_out_pil_label[i_image], HP_COLOR_MAP), dl_pil_img_lr[i_image]
                                                          ,list_out_pil_lr[i_image], list_out_pil_sr[i_image]
                                                           # 7 ~ 12 (sub title)
                                                          ,"HR Image", "Label Answer", "Predicted", "LR Image", "Intermediate Feature", "SR Image"
                                                           # 13 (path plt)
                                                          ,PATH_OUT_IMAGE + i_mode + "/" + str(i_epoch + 1)
                                                           # 14 ~ 17 (data CF)
                                                          ,tensor_out_seg_softmax[i_image].clone().detach().cpu()
                                                          ,tensor_out_sr[i_image].clone().detach().cpu()
                                                          ,dl_ts_lab_hr[i_image].clone().detach().cpu()
                                                          ,dl_ts_img_hr[i_image].clone().detach().cpu()
                                                           # 18 (path CF)
                                                          ,PATH_OUT_IMAGE + i_mode + "/_CrossFeature/" + str(i_epoch + 1)
                                                           # 19 (path pil)
                                                          ,PATH_OUT_IMAGE + i_mode + "/_SR_Images/" + str(i_epoch + 1)
                                                           # 20 (plt title)
                                                          ,plt_title
                                                           # 21 (file name)
                                                          ,tmp_file_name
                                                          )
                                                         )
                            
                            #>>> SSSR
                        
                        elif TRAINER_MODE == "SR":
                            #<<< SR
                            if model_type == "SR_A" or model_type == "SR_B" or model_type == "SR_C":
                                # SR model train does not use mp_buffer
                                # RAM 할당량 보고 버퍼 사용여부 결정하기
                                # 버퍼 사용시, pil 저장을 위한 is_best 처리방식 수정해야됨
                                
                                if len(list_mp_buffer) >= BUFFER_SIZE and BUFFER_SIZE > 0:
                                    # chunk full -> toss mp_buffer -> empty mp_buffer
                                    if i_mode == 'test':
                                        tmp_is_best = rb_val_psnr.is_best_max   #chunk 단위 buffer 구조상 valid 기준으로 best여부 검사
                                    else:
                                        tmp_is_best = False
                                    
                                    plts_saver_sr(list_mp_buffer, is_best = tmp_is_best, no_employ=(employ_threshold >= len(list_mp_buffer)))
                                    
                                    try:
                                        del list_mp_buffer
                                    except:
                                        pass
                                    list_mp_buffer = []
                                
                                if i_image < 2: # batch 당 최대 2장 저장
                                    list_mp_buffer.append((# 0 (model name)
                                                           model_type
                                                           # 1 ~ 3 (pils)
                                                          ,dl_pil_img_hr[i_image]
                                                          ,dl_pil_img_lr[i_image]
                                                          ,list_out_pil_sr[i_image]
                                                           # 4 ~ 6 (sub title)
                                                          ,"HR Image", "LR Image", "SR Image"
                                                           # 7 (path for plt)
                                                          ,PATH_OUT_IMAGE + i_mode + "/" + str(i_epoch + 1)
                                                           # 8 (path for SR pil)
                                                          ,PATH_OUT_IMAGE + i_mode + "/_SR_Images/" + str(i_epoch + 1)
                                                           # 9 (plt title)
                                                          ,plt_title
                                                           # 10 (file name)
                                                          ,tmp_file_name
                                                          )
                                                         )
                            
                            #>>> SR
                        
                        elif TRAINER_MODE == "SS":
                            #<<< SS
                            if model_type == "SS_A":
                                if len(list_mp_buffer) >= BUFFER_SIZE and BUFFER_SIZE > 0:
                                    # chunk full -> toss mp_buffer -> empty mp_buffer
                                    if i_mode == 'test':
                                        tmp_is_best = rb_val_ious.is_best_max   #chunk 단위 buffer 구조상 valid 기준으로 best여부 검사
                                    else:
                                        tmp_is_best = False
                                    
                                    plts_saver_ss(list_mp_buffer, is_best = tmp_is_best, no_employ=(employ_threshold >= len(list_mp_buffer)))
                                    
                                    try:
                                        del list_mp_buffer
                                    except:
                                        pass
                                    list_mp_buffer = []
                                
                                if i_image < 2: # batch 당 최대 2장 저장
                                    list_mp_buffer.append((# 0 (model name)
                                                           model_type
                                                           # 1 ~ 3 (pils)
                                                          ,dl_pil_img_hr[i_image]
                                                          ,label_2_RGB(dl_pil_lab_hr[i_image], HP_COLOR_MAP)
                                                          ,label_2_RGB(list_out_pil_label[i_image], HP_COLOR_MAP)
                                                           # 4 ~ 6 (sub title)
                                                          ,"HR Image", "Label Answer", "Predicted"
                                                           # 7 (path for plt)
                                                          ,PATH_OUT_IMAGE + i_mode + "/" + str(i_epoch + 1)
                                                           # 8 (path for SS pil - Label)
                                                          ,PATH_OUT_IMAGE + i_mode + "/_SS_Labels/" + str(i_epoch + 1)
                                                           # 9 (plt title)
                                                          ,plt_title
                                                           # 10 (file name)
                                                          ,tmp_file_name
                                                          )
                                                         )
                            #>>> SS
                        
                    #>>> 예측결과를 이미지로 생성
                    
                # 이번 batch의 평균 mIoU
                batch_miou_mean = batch_miou_sum / current_batch_size
                # epoch 단위 누적기록 갱신
                epoch_miou_sum += batch_miou_mean
                
                #VVV [batch 단위 이미지 평가 (mIoU)] --------------------------
                
                
                #<<< new_record_system
                if i_mode == "train":
                    rb_train_loss.update_batch()
                    rb_train_psnr.update_batch()
                    rb_train_ssim.update_batch()
                    rb_train_niqe.update_batch()
                    rb_train_pa.update_batch()
                    rb_train_ca.update_batch()
                    rb_train_ious.update_batch()
                elif i_mode == "val":
                    rb_val_loss.update_batch()
                    rb_val_psnr.update_batch()
                    rb_val_ssim.update_batch()
                    rb_val_niqe.update_batch()
                    rb_val_pa.update_batch()
                    rb_val_ca.update_batch()
                    rb_val_ious.update_batch()
                elif i_mode == "test":
                    rb_test_loss.update_batch()
                    rb_test_psnr.update_batch()
                    rb_test_ssim.update_batch()
                    rb_test_niqe.update_batch()
                    rb_test_pa.update_batch()
                    rb_test_ca.update_batch()
                    rb_test_ious.update_batch()
                    
                #>>> new_record_system
                
                if timer_gpu_all_finish is not None:
                    timer_gpu_all_finish.record()
                
                try:
                    # GPU timer record
                    torch.cuda.synchronize()
                    timer_gpu_all_record = str(round(timer_gpu_all_start.elapsed_time(timer_gpu_all_finish) / 1000, 4))
                    timer_gpu_record = str(round(timer_gpu_start.elapsed_time(timer_gpu_finish) / 1000, 4))
                except:
                    timer_gpu_all_record = "FAIL"
                    timer_gpu_record = "FAIL"
                    
                try:
                    # CPU timer record: include batch preparation
                    timer_cpu_record = str(round(time.time() - timer_cpu_start, 4))
                except:
                    timer_cpu_record = "FAIL"
                
                
                print("\rin", i_mode, (i_epoch + 1), count_dataloader, "/", i_batch_max, " "
                     , "CPU:", timer_cpu_record, " GPU-Batch:", timer_gpu_all_record, " GPU-Model:", timer_gpu_record
                     ,end = '')
                
                if timer_gpu_all_start is not None:
                    # dataloader 시간 측정을 위함
                    timer_gpu_all_start.record()
                
                timer_cpu_start = time.time()
                
                i_batch += 1
                
            # End of "for path_x, path_y in dataloader_input:"
            # dataloader_input 종료됨
            
            # summary plt pil save
            if TRAINER_MODE == "SSSR":
                if model_type == "SSSR_A" or model_type == "SSSR_B" or model_type == "SSSR_C" or model_type == "SSSR_D":
                    if not list_mp_buffer:
                        print("(caution) list_mp_buffer is emply...")
                    else:
                        if i_mode == 'test':
                            tmp_is_best = rb_val_ious.is_best_max   #chunk 단위 buffer 구조상 valid 기준으로 best여부 검사
                        else:
                            tmp_is_best = False
                        plts_saver_sssr(list_mp_buffer, is_best = tmp_is_best, no_employ=(employ_threshold >= len(list_mp_buffer)))
            
            elif TRAINER_MODE == "SR":
                if model_type == "SR_A" or model_type == "SR_B" or model_type == "SR_C":
                    if not list_mp_buffer:
                        print("(caution) list_mp_buffer is emply...")
                    else:
                        if i_mode == 'test':
                            tmp_is_best = rb_val_psnr.is_best_max   #chunk 단위 buffer 구조상 valid 기준으로 best여부 검사
                        else:
                            tmp_is_best = False
                        plts_saver_sr(list_mp_buffer, is_best = tmp_is_best, no_employ=(employ_threshold >= len(list_mp_buffer)))
            
            elif TRAINER_MODE == "SS":
                if model_type == "SS_A":
                    if not list_mp_buffer:
                        print("(caution) list_mp_buffer is emply...")
                    else:
                        if i_mode == 'test':
                            tmp_is_best = rb_val_ious.is_best_max   #chunk 단위 buffer 구조상 valid 기준으로 best여부 검사
                        else:
                            tmp_is_best = False
                        
                        plts_saver_ss(list_mp_buffer, is_best = tmp_is_best, no_employ=(employ_threshold >= len(list_mp_buffer)))
            
            #<<< new_record_system
            path_log_graph = PATH_OUT_LOG                                   # log 그래프 저장 경로 (default = "./")    #@@@ 작성중
            
            if TRAINER_MODE == "SS":
                _graph_update_niqe = False          # niqe 그래프 업데이트 여부
            else:
                _graph_update_niqe = (i_epoch < 3) 
            
            if i_mode == "train":
                str_result_epoch_loss = str(rb_train_loss.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_psnr = str(rb_train_psnr.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_ssim = str(rb_train_ssim.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_niqe = str(rb_train_niqe.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True
                                                                      ,is_update_graph=_graph_update_niqe
                                                                      )
                                           )
                str_result_epoch_pa   = str(rb_train_pa.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_ca   = str(rb_train_ca.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_miou_ious = rb_train_ious.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True)
                
            elif i_mode == "val":
                str_result_epoch_loss = str(rb_val_loss.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_psnr = str(rb_val_psnr.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_ssim = str(rb_val_ssim.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_niqe = str(rb_val_niqe.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True
                                                                    ,is_update_graph=_graph_update_niqe
                                                                    )
                                           )
                str_result_epoch_pa   = str(rb_val_pa.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_ca   = str(rb_val_ca.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_miou_ious = rb_val_ious.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True)
                
            elif i_mode == "test":
                str_result_epoch_loss = str(rb_test_loss.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_psnr = str(rb_test_psnr.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_ssim = str(rb_test_ssim.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_niqe = str(rb_test_niqe.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_pa   = str(rb_test_pa.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_ca   = str(rb_test_ca.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True))
                str_result_epoch_miou_ious = rb_test_ious.update_epoch(is_return = True, path = path_log_graph, is_print_sub = True)
                
            
            # log total dict 업데이트
            tmp_str_contents = str_result_epoch_loss
            tmp_str_contents += "," + str_result_epoch_psnr + "," + str_result_epoch_ssim + "," + str_result_epoch_niqe
            tmp_str_contents += "," + str_result_epoch_pa   + "," + str_result_epoch_ca   + "," + str_result_epoch_miou_ious
            #epoch 번호 - Loss PSRN SSIM NIQE mIoU IoUs
            update_dict_v2(str(i_epoch + 1), tmp_str_contents
                          ,in_dict_dict = d_d_log_total
                          ,in_dict_key = i_mode
                          ,in_print_head = "d_log_total_" + i_mode
                          )
            print("\n")
            
            # log 기록 업데이트 (epoch 단위)
            dict_2_txt_v2(in_file_path = PATH_OUT_LOG + i_mode + "/"
                         ,in_file_name = "new_log_epoch_" + i_mode + "_" + str(i_epoch + 1) + ".csv"
                         ,in_dict_dict = d_d_log_epoch
                         ,in_dict_key = i_mode
                         )
            # log 기록 업데이트 (학습 전체 단위)
            dict_2_txt_v2(in_file_path = PATH_OUT_LOG
                         ,in_file_name = "new_log_total_" + i_mode + ".csv"
                         ,in_dict_dict = d_d_log_total
                         ,in_dict_key = i_mode
                         )
            
            #>>> new_record_system
            
            epoch_loss_mean = epoch_loss_sum / i_batch_max
            epoch_miou_mean = epoch_miou_sum / i_batch_max
            
            # 라벨별 iou 평균 계산
            tmp_str = ""
            for i_key in dict_ious_accumulate:
                tmp_count, tmp_iou_sum = dict_ious_accumulate[i_key][0], dict_ious_accumulate[i_key][-1]
                # print(tmp_count, tmp_iou_sum, str(tmp_iou_sum / tmp_count))
                try:
                    tmp_str += "," + str(tmp_iou_sum / tmp_count)
                except:
                    tmp_str += ",ERROR"
                
            
            # log total dict 업데이트
            if TRAINER_MODE != "SR" and RUN_WHERE != -1:
                update_dict_v2(str(i_epoch + 1), str(epoch_loss_mean) + tmp_str
                              ,in_dict_dict = dict_dict_log_total
                              ,in_dict_key = i_mode
                              ,in_print_head = "dict_log_total_" + i_mode
                              )
            print("\n")
            
            if TRAINER_MODE != "SR" and RUN_WHERE != -1:
                # 하드 접근횟수 감소를 위해 Colab에서는 구버전 log 저장 안함
                # log 기록 업데이트 (epoch 단위)
                dict_2_txt_v2(in_file_path = PATH_OUT_LOG + i_mode + "/"
                             ,in_file_name = "log_epoch_" + i_mode + "_" + str(i_epoch + 1) + ".csv"
                             ,in_dict_dict = dict_dict_log_epoch
                             ,in_dict_key = i_mode
                             )
            
            # log 기록 업데이트 (학습 전체 단위)
            dict_2_txt_v2(in_file_path = PATH_OUT_LOG
                         ,in_file_name = "log_total_" + i_mode + ".csv"
                         ,in_dict_dict = dict_dict_log_total
                         ,in_dict_key = i_mode
                         )
            
            
            # epoch 단위 scheduler 갱신 -> check_point 저장 
            if i_mode == "train":
                if HP_SCHEDULER_UPDATE_INTERVAL == "epoch":
                    # 사용된 lr 기록
                    rb_train_lr.add_item(optimizer.param_groups[0]['lr'])
                    rb_train_lr.update_batch()
                    # 스케쥴러 갱신
                    scheduler.step()
                    print("scheduler.step()")
                
                # 이번 epoch lr 기록 갱신
                rb_train_lr.update_epoch(is_return = False, path = path_log_graph, is_print_sub = True)
                
                if RUN_WHERE == -1:
                    save_interval = 50
                else:
                    save_interval = 100
                
                if (i_epoch + 1) % save_interval == 0:
                    # check_point 저장경로
                    tmp_path = PATH_OUT_MODEL + "check_points/"
                    if not os.path.exists(tmp_path):
                        os.makedirs(tmp_path)
                    # 모델 체크포인트 저장
                    print("\n[--- 체크포인트", str(i_epoch + 1), "저장됨 ---]\n")
                    torch.save({'epoch': (i_epoch + 1)                           # (int) 중단 시점 epoch 값
                               ,'model_state_dict': model.state_dict()           # (state_dict) model.state_dict()
                               ,'optimizer_state_dict': optimizer.state_dict()   # (state_dict) optimizer.state_dict()
                               ,'scheduler_state_dict': scheduler.state_dict()   # (state_dict) scheduler.state_dict()
                               }
                              ,tmp_path + str(i_epoch + 1) +'_check_point.tar'
                              )
                
            
            if i_mode == "val":
                if TRAINER_MODE == "SSSR":
                    #tmp_is_best = rb_val_psnr.is_best_max or rb_val_ious.is_best_max
                    tmp_is_best = rb_val_ious.is_best_max
                    if prev_best is not None:
                        # prev best 값이 입력된 경우
                        if prev_best > rb_val_ious.total_max[-1]:
                            print("\nprev_best 못넘음...")
                            tmp_is_best = False
                elif TRAINER_MODE == "SR":
                    tmp_is_best = rb_val_psnr.is_best_max
                    if prev_best is not None:
                        # prev best 값이 입력된 경우
                        if prev_best > rb_val_psnr.total_max[-1]:
                            print("\nprev_best 못넘음...")
                            tmp_is_best = False
                elif TRAINER_MODE == "SS":
                    tmp_is_best = rb_val_ious.is_best_max
                    if prev_best is not None:
                        # prev best 값이 입력된 경우
                        if prev_best > rb_val_ious.total_max[-1]:
                            print("\nprev_best 못넘음...")
                            tmp_is_best = False
                
                if tmp_is_best:
                    print("\n< Best Valid Epoch > Model State Dict 저장됨\n")
                    
                    # state_dict 저장경로
                    tmp_path = PATH_OUT_MODEL + "state_dicts/"
                    if not os.path.exists(tmp_path):
                        os.makedirs(tmp_path)
                    torch.save(model.state_dict()
                              ,tmp_path + str(i_epoch + 1) +'_model_state_dict.pt'
                              )
                else:
                    print("\n< Not a Best Valid Epoch >\n")
            
            # 에러 발생했던 경우, 로그 저장
            if flag_error != 0:
                dict_2_txt(in_file_path = PATH_OUT_LOG + "error/"
                          ,in_file_name = "log_error_" + i_mode + "_" + str(i_epoch) + "_" + str(flag_error) + ".csv"
                          ,in_dict = dict_log_error
                          )
                
            
            # [epoch 완료 -> 변수 초기화] ---
            
            print("epoch 완료")
            i_batch = 0
            
            

#=== End of trainer_



print("End of trainer_total.py")
