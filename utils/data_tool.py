#data_tool.py
import os
import numpy as np
import sys
import math
import random

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import cv2

import time

import warnings

#[클래스] ----------------------------------------------------------------
class ColorJitter_Double(torch.nn.Module):
    """
    torchvision.transforms.ColorJitter 기능 확장판
    2 장의 이미지를 넣으면 동일 랜덤 옵션으로 ColorJitter 기능 수행해줌
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self._CJ = torchvision.transforms.ColorJitter(brightness   = brightness
                                                     ,contrast     = contrast
                                                     ,saturation   = saturation
                                                     ,hue          = hue
                                                     )
    
    def forward(self, *args):
        # index, brightness_factor, contrast_factor, saturation_factor, hue_factor
        fn_idx, b_f, c_f, s_f, h_f = self._CJ.get_params(self._CJ.brightness
                                                        ,self._CJ.contrast
                                                        ,self._CJ.saturation
                                                        ,self._CJ.hue
                                                        )
        
        if len(args) == 1:
            img = args[0]
            for fn_id in fn_idx:
                if fn_id == 0 and b_f is not None:
                    img = torchvision.transforms.functional.adjust_brightness(img, b_f)
                    
                elif fn_id == 1 and c_f is not None:
                    img = torchvision.transforms.functional.adjust_contrast(img, c_f)
                    
                elif fn_id == 2 and s_f is not None:
                    img = torchvision.transforms.functional.adjust_saturation(img, s_f)
                    
                elif fn_id == 3 and h_f is not None:
                    img = torchvision.transforms.functional.adjust_hue(img, h_f)
            
            return img
            
        else:
            img = args[0]
            img_2 = args[1]
            for fn_id in fn_idx:
                if fn_id == 0 and b_f is not None:
                    img   = torchvision.transforms.functional.adjust_brightness(img, b_f)
                    img_2 = torchvision.transforms.functional.adjust_brightness(img_2, b_f)
                    
                elif fn_id == 1 and c_f is not None:
                    img   = torchvision.transforms.functional.adjust_contrast(img, c_f)
                    img_2 = torchvision.transforms.functional.adjust_contrast(img_2, c_f)
                    
                elif fn_id == 2 and s_f is not None:
                    img   = torchvision.transforms.functional.adjust_saturation(img, s_f)
                    img_2 = torchvision.transforms.functional.adjust_saturation(img_2, s_f)
                    
                elif fn_id == 3 and h_f is not None:
                    img   = torchvision.transforms.functional.adjust_hue(img, h_f)
                    img_2 = torchvision.transforms.functional.adjust_hue(img_2, h_f)
            
            return img, img_2

#=== End of ColorJitter_Double


class score_box():
    #각종 점수 저장용 클래스
    #안정성을 위해 new_item으로 int, float 형만 입력하길 권장
    
    #개시
    def __init__(self, name_item = "no_info"):
        #이 클래스에서 출력되는 모든 문장엔 다음의 안내문이 추가됨
        self.name_class = "[class score_box] ->"
        #클래스 하위 이름
        self.name_item = name_item
        print(self.name_class,"init",self.name_item)
        #(int, float) 이번 숫자값
        self.item = 0
        #입력된 숫자값 수
        self.count = 0
        #아이템 총 합
        self.sum = 0
        #아이템 평균값
        self.mean = 0
    
    #아이템 업데이트(이번 아이템, 아이템 개수)
    def update(self, new_item, number_of_new_items = 1):
        self.item = new_item
        self.count += number_of_new_items
        self.sum += new_item
        try:
            self.mean = self.sum / self.count
        except:
            print("(exc)", self.name_class, "item can not devided with 0")
    
    #현재 클래스에 저장된 정보 출력
    def info(self):
        print(self.name_class, "name:"
             ,self.name_item, ", last item:", self.item, ", count:"
             ,self.count, ", sum:", self.sum, ", mean:", self.mean
             )





#[함수] ---------------------------------------------------------------

#npArray를 이미지로 출력
def imshow_np(in_np, **kargs):
    
    try:
        plt.figure(figsize = kargs['figsize'])
    except:
        pass
    
    plt.imshow(in_np)
    plt.show()

#=== End of imshow_np

#PIL 이미지 출력
def imshow_pil(in_pil, **kargs):
    '''
    imshow_pil(#pil image show with plt function
               in_pil
               #(선택) (tuple) 출력 크기
              ,figsize = (,)
               #(선택) (bool) pil 이미지 정보 출력 여부 (default = True)
              ,print_info = 
              )
    '''
    
    try:
        plt.figure(figsize = kargs['figsize'])
    except:
        pass
    plt.imshow(np.array(in_pil))
    plt.show()
    
    try:
        print_info = kargs['print_info']
    except:
        print_info = True
    
    if print_info:
        try:
            print("Format:", in_pil.format, "  Mode:", in_pil.mode, "  Size (w,h):", in_pil.size)
        except:
            print("Format: No Info", "  Mode:", in_pil.mode, "  Size (w,h):", in_pil.size)
    

#=== End of imshow_pil

#Tensor 이미지 출력 (개선판)
def imshow_ts(in_ts, **kargs):
    try:
        plt.figure(figsize = kargs['figsize'])
    except:
        pass
    plt.imshow(np.array(transforms.functional.to_pil_image(in_ts)))
    plt.show()

#=== End of imshow_ts

def label_2_tensor(**kargs):
    func_name = "[label_2_tensor] ->"
    try:
        #라벨 채널이 Gray 인가? (default = True)
        is_gray = kargs['is_gray']
    except:
        is_gray = True
    
    if is_gray:
        in_pil = kargs['in_pil']
        in_np = np.array(in_pil)
    else:
        #RGB -> Gray 변환과정 (추가필요)
        pass
    
    #전체 라벨 수 (void 포함)
    label_total = kargs['label_total']
    
    #void 라벨 번호
    label_void = kargs['label_void']
    
    #one-hot 라벨 dilation 시행여부 (좀 더 넓은 영역을 해당 오브젝트 범위로 처리함)
    try:
        is_dilated = kargs['is_dilated']
    except:
        is_dilated = False
    
    if is_dilated:
        cv_kernel_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    
    #사용할 transform_to_ts_lab 함수 (별도 입력 없으면 ToTensor만 시행)
    try:
        label_to_ts_func = kargs['label_to_ts_func']
    except:
        label_to_ts_func = transforms.Compose([#PIL 이미지 or npArray -> pytorch 텐서
                                              transforms.ToTensor()
                                              ])
    
    
    flag_init_label_gen = 0
    for i_label in range(label_total):
        if i_label == label_void:
            #void 라벨 번호는 텐서변환 생략
            continue
        np_label_single = np.where(in_np == i_label, 1, 0).astype(np.uint8)
        #첫 라벨 생성
        if flag_init_label_gen == 0:
            flag_init_label_gen = 1
            if is_dilated:
                pil_dilated = Image.fromarray(cv2.dilate(np_label_single, cv_kernel_dilation))
                out_tensor = label_to_ts_func(pil_dilated)
            else:
                pil_onehot = Image.fromarray(np_label_single)
                out_tensor = label_to_ts_func(pil_onehot)
        #2번쨰 라벨부터 ~
        else:
            if is_dilated:
                pil_dilated = Image.fromarray(cv2.dilate(np_label_single, cv_kernel_dilation))
                out_tensor = torch.cat([out_tensor, label_to_ts_func(pil_dilated)], dim = 0)
            else:
                pil_onehot = Image.fromarray(np_label_single)
                out_tensor = torch.cat([out_tensor, label_to_ts_func(pil_onehot)], dim = 0)
    
    return out_tensor

#=== End of label_2_tensor

#label(gray)을 각 채널로 분리
#IN (**2 + 1): 
#               (PIL)  in_pil = GRAY 이미지, 
#               (int)  in_label_num = 선택된 라벨값(0~ "PR_LABEL - 1")
#               (bool) is_dilatate = (opencv) 모폴로지 팽창 연산 적용 여부
#OUT(1): (PIL) GRAY 이미지 (0 or 1)
def gray_2_onehot_v2(**kargs):
    
    in_pil = kargs['in_pil']
    in_label_num = kargs['in_label_num']
    try:
        is_dilatate = kargs['is_dilatate']
    except:
        is_dilatate = False
    
    #print("(func) init gray_2_onehot")
    in_np = np.array(in_pil)
    
    #in_h, in_w = in_np.shape
    #print("in_np", in_np.shape)
    
    out_np = np.where(in_np == in_label_num, 1, 0).astype(np.uint8)
    #out_np = np.where(in_np == in_label_num, 1, 0)
    
    if is_dilatate:
        #모폴로지 연산 (opencv - cv2) 시행
        #Label Relaxation (Improving Semantic Segmentation via Video Propagation and Label Relaxation) 적용
        #https://arxiv.org/abs/1812.01593
        cv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        out_cv = cv2.dilate(out_np, cv_kernel)
        out_pil = Image.fromarray(out_cv)
    else:
        out_pil = Image.fromarray(out_np)
    
    #print(out_pil, out_pil.size)
    
    #plt.imshow(np.array(out_pil))
    #plt.show()
    
    return out_pil

#===

#tensor reshape ([b*c, h, w] 형태 텐서를 [b, c, h, w]로 변환)
#IN (2): (ts)입력 텐서([b*c, h, w] 형태), (int) 출력 channel 크기 값
#OUT(1): (ts)출력 텐서([b, c, h, w] 형태)
def tensor_reshape(in_tensor, in_channel):
    #print("(func) init tensor_reshape")
    
    in_bc, in_h, in_w = in_tensor.shape
    
    #print(in_bc, in_h, in_w, in_tensor.shape)
    #print("to", in_bc // in_channel, in_channel, in_h, in_w)
    
    out_tensor = torch.reshape(in_tensor, (in_bc // in_channel, in_channel, in_h, in_w))
    
    #print("out_tensor", out_tensor.shape)
    
    return out_tensor

#===

#IN (2): 
#        (pil)  gray label
#        (dict) label_color_map
#OUT(1): (pil) RGB 이미지
def label_2_RGB(in_pil, in_label_rgb):
    
    in_np = np.array(in_pil)
    in_h, in_w = in_np.shape
    
    out_np_rgb = np.zeros((in_h, in_w, 3), dtype=np.uint8)
    
    for label, rgb in in_label_rgb.items():
        out_np_rgb[in_np == label, :] = rgb
    
    return Image.fromarray(out_np_rgb)

#===


def pil_resize(**kargs):
    #선택 옵션으로 resize (pil -> pil)
    """
        out_pil = pil_resize(in_pil = 
                            ,option = "NEAREST"
                            ,in_w = HP_ORIGIN_IMG_W
                            ,in_h = HP_ORIGIN_IMG_H
                            ,out_w = HP_MODEL_SS_IMG_W
                            ,out_h = HP_MODEL_SS_IMG_H
                            ,is_reverse = True
                            )
    """
    
    #동일 크기로의 변환은 입력 이미지를 그대로 return하는 방식 적용함
    
    func_name = "[pil_resize] ->"
    in_pil = kargs['in_pil']
    
    try:
        in_w = kargs['in_w']
        in_h = kargs['in_h']
        
        #역변환 여부 ((out_w, out_h) -> (in_w, in_h) 에 해당한 경우)
        #(bool)
        is_reverse = kargs['is_reverse']
    except:
        in_w, in_h = in_pil.size
        is_reverse = False
    
    out_w = kargs['out_w']
    out_h = kargs['out_h']
    
    if in_w == out_w and in_h == out_h:
        print(func_name, "no resize applied")
        return in_pil
    
    
    #(str) 이미지 보간법 방식
    #https://pillow.readthedocs.io/en/stable/handbook/concepts.html#PIL.Image.BILINEAR
    try:
        option = kargs['option']
    except:
        option = "NEAREST"
    
    
    #w, h = img.size
    #im_resized = im.resize((width, height))
    
    if option == "BILINEAR" or option == "bilinear":
        #print("option: BILINEAR")
        if is_reverse:
            out_pil = in_pil.resize((int(in_w), int(in_h)), Image.BILINEAR)
        else:
            out_pil = in_pil.resize((int(out_w), int(out_h)), Image.BILINEAR)
    
    elif option == "BICUBIC" or option == "bicubic":
        #print("option: BICUBIC")
        if is_reverse:
            out_pil = in_pil.resize((int(in_w), int(in_h)), Image.BICUBIC)
        else:
            out_pil = in_pil.resize((int(out_w), int(out_h)), Image.BICUBIC) 
    
    elif option == "LANCZOS" or option == "lanczos":
        #print("option: LANCZOS")
        if is_reverse:
            out_pil = in_pil.resize((int(in_w), int(in_h)), Image.LANCZOS)
        else:
            out_pil = in_pil.resize((int(out_w), int(out_h)), Image.LANCZOS) 
    
    else: #default: NEAREST
        #print("option: NEAREST")
        if is_reverse:
            out_pil = in_pil.resize((int(in_w), int(in_h)), Image.NEAREST)
        else:
            out_pil = in_pil.resize((int(out_w), int(out_h)), Image.NEAREST)
    
    return out_pil

#=== End of pil_resize

#Data Augm- 종합 함수 (좌우 flip, 랜덤 crop, 랜덤 rotate)
#사용예시
'''
in_pil_x_augm, in_pil_y_augm, str_used_option = pil_augm_v3(in_pil_x = in_pil_x_raw
                                                           ,in_option_resize_x = Image.LANCZOS
                                                           ,in_option_rotate_x = Image.BICUBIC
                                                           ,in_pil_y = in_pil_y_raw
                                                           ,in_option_resize_y = Image.NEAREST
                                                           ,in_option_rotate_y = Image.NEAREST
                                                           ,in_crop_wh_max = 10
                                                           ,in_crop_wh_min = 4
                                                           ,in_rotate_degree_max = 5
                                                           ,in_percent_flip = 50
                                                           ,in_percent_crop = 70
                                                           ,in_percent_rotate = 20
                                                           ,is_return_options = True
                                                           )
'''

'''
#random_in_ticket 계열값 전부 제거 후, random.uniform으로 대체함
#입력 이미지 크기와 출력 이미지 크기가 다른 경우, resize 기능 추가됨
def pil_augm_v3(**kargs):
    name_func = "[pil_augm_v3] -->"
    #[입력 x 관련]---
    #(pil) 입력 이미지 x (이미지)
    in_pil_x = kargs['in_pil_x']
    #최종 이미지 크기 확인용
    input_x_w, input_x_h = in_pil_x.size
    
    #https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters
    #resize 시행 시 옵션 (Image.NEAREST, PIL.Image.BILINEAR, ... , Image.LANCZOS)
    in_option_resize_x = kargs['in_option_resize_x']
    
    #rotate 시행 시 옵션 (Image.NEAREST, Image.BILINEAR, Image.BICUBIC 중 선택)
    in_option_rotate_x = kargs['in_option_rotate_x']
    
    #[입력 y 관련]---
    try:
        #(pil) 입력 이미지 y (라벨)
        in_pil_y = kargs['in_pil_y']
        in_option_resize_y = kargs['in_option_resize_y']
        in_option_rotate_y = kargs['in_option_rotate_y']
        is_y_in = True
        #최종 이미지 크기 확인용
        input_y_w, input_y_h = in_pil_y.size
    except:
        in_pil_y = in_pil_x
        in_option_resize_y = in_option_resize_x
        in_option_rotate_y = in_option_rotate_x
        is_y_in = False
    
    #[x y 공용]---
    #crop 시 잘라낼 영역의 너비 & 높이 최대 / 최소 값
    #in_crop_wh = kargs['in_crop_wh']
    in_crop_wh_max = kargs['in_crop_wh_max']
    in_crop_wh_min = kargs['in_crop_wh_min']
    
    #rotate 시 최대 각도
    in_rotate_degree_max = kargs['in_rotate_degree_max']
    
    #(int) flip을 시행할 확률 (0 ~ 100)
    in_percent_flip = kargs['in_percent_flip']
    #(int) crop을 시행할 확률 (0 ~ 100)
    in_percent_crop = kargs['in_percent_crop']
    #(int) flip을 시행할 확률
    in_percent_rotate = kargs['in_percent_rotate']
    
    #사용된 옵션 기록 return 여부
    try:
        is_return_options = kargs['is_return_options']
    except:
        is_return_options = False
    
    #return_option = 사용된 변환 옵션 기록한 str
    
    #----------
    #입력 이미지 크기
    in_w, in_h = in_pil_x.size
    
    #시행여부: flip
    if in_percent_flip > random.uniform(0, 100):
        is_flip = True
        return_option = "Flip O"
    else:
        is_flip = False
        return_option = "Flip X"
    
    #시행여부: crop
    if in_percent_crop > random.uniform(0, 100):
        is_crop = True
        #crop 구역 지정 (0 ~ 9)
        crop_zone = int(random.uniform(0, 100) % 10)
        #crop 시 제거될 영역의 너비
        in_crop_wh = int(random.uniform(in_crop_wh_min, in_crop_wh_max))
        return_option += " / Crop " + str(crop_zone) + " " + str(in_crop_wh)
    else:
        is_crop = False
        return_option += " / Crop X"
    
    #시행여부: rotate
    if in_percent_rotate > random.uniform(0, 100):
        is_rotate = True
        in_theta = int(random.uniform(0, in_rotate_degree_max - 1)) + 1
        if 50 > random.uniform(0, 100):
            in_theta = in_theta * (-1)
            
        return_option += " / Rotate " + str(in_theta)
    else:
        is_rotate = False
        return_option += " / Rotate X"
        
    #<<< flip 시행 (in_pil -> out_pil_x)
    if is_flip:
        out_pil_x = in_pil_x.transpose(Image.FLIP_LEFT_RIGHT)
        if is_y_in:
            out_pil_y = in_pil_y.transpose(Image.FLIP_LEFT_RIGHT)
    else: #no flip
        #print("pass flip")
        out_pil_x = in_pil_x.copy()
        if is_y_in:
            out_pil_y = in_pil_y.copy()
    #>>> flip 시행
    
    #<<< crop 시행 (out_pil_x -> out_pil_x)
    if is_crop:
        out_pil_x = out_pil_x.resize((int(in_w + in_crop_wh), int(in_h + in_crop_wh)), in_option_resize_x)
        if is_y_in:
            out_pil_y = out_pil_y.resize((int(in_w + in_crop_wh), int(in_h + in_crop_wh)), in_option_resize_y)
        
        #crop - 키패드 기준
        if crop_zone == 1: #crop (1)
            c_left  = 0
            c_right = in_w
            c_upper = 0 + in_crop_wh
            c_lower = in_h + in_crop_wh

        elif crop_zone == 2: #crop (2)
            c_left  = 0 + in_crop_wh / 2
            c_right = in_w + in_crop_wh / 2
            c_upper = 0 + in_crop_wh
            c_lower = in_h + in_crop_wh

        elif crop_zone == 3: #crop (3)
            c_left  = 0 + in_crop_wh
            c_right = in_w + in_crop_wh
            c_upper = 0 + in_crop_wh
            c_lower = in_h + in_crop_wh

        elif crop_zone == 4: #crop (4)
            c_left  = 0
            c_right = in_w
            c_upper = 0 + in_crop_wh / 2
            c_lower = in_h + in_crop_wh /2

        elif crop_zone == 6: #crop (6)
            c_left  = 0 + in_crop_wh
            c_right = in_w + in_crop_wh
            c_upper = 0 + in_crop_wh / 2
            c_lower = in_h + in_crop_wh / 2

        elif crop_zone == 7: #crop (7)
            c_left  = 0
            c_right = in_w
            c_upper = 0
            c_lower = in_h

        elif crop_zone == 8: #crop (8)
            c_left  = 0 + in_crop_wh / 2
            c_right = in_w + in_crop_wh / 2
            c_upper = 0
            c_lower = in_h

        elif crop_zone == 9: #crop (9)
            c_left  = 0 + in_crop_wh
            c_right = in_w + in_crop_wh
            c_upper = 0 
            c_lower = in_h

        else: #crop (0 & 5) -> center crop
            c_left  = 0 + in_crop_wh / 2
            c_right = in_w + in_crop_wh / 2
            c_upper = 0 + in_crop_wh / 2
            c_lower = in_h + in_crop_wh / 2

        out_pil_x = out_pil_x.crop((int(c_left), int(c_upper), int(c_right), int(c_lower)))
        if is_y_in:
            out_pil_y = out_pil_y.crop((int(c_left), int(c_upper), int(c_right), int(c_lower)))
        
    else: 
        #no crop
        pass
        #print("pass crop")
        #out_pil_x = out_pil_x
        #out_pil_y = out_pil_y
    #>>> crop 시행
    
    #<<< rotate 시행 (out_pil_x -> out_pil_x)
    if is_rotate:
        #<<< <<<
        #----플래그
        #입력 반전 시행 여부
        flag_flip = 0

        #---예외처리
        
        #역방향 (시계방향) 변환옵션 사용여부 확인
        if in_theta < 0:
            out_pil_x = out_pil_x.transpose(Image.FLIP_LEFT_RIGHT)
            if is_y_in:
                out_pil_y = out_pil_y.transpose(Image.FLIP_LEFT_RIGHT)
            
            in_theta = in_theta * (-1)
            flag_flip = 1
        
        if in_theta >= 360:
            in_theta = in_theta % 360

        check_theta = in_theta // 90

        #90도 이상 각도에 대한 처리 시행
        if check_theta > 0:
            #print("회전보정 시행")
            out_pil_x = out_pil_x.rotate(check_theta * 90, resample = in_option_rotate_x, expand = True)
            if is_y_in:
                out_pil_y = out_pil_y.rotate(check_theta * 90, resample = in_option_rotate_y, expand = True)
            in_theta = in_theta % 90
            #flag_rotate = check_theta
        
        #print("예외처리 완료", out_pil_x.size)

        #---계산 시행
        #입력 이미지 크기 확인
        in_r_w, in_r_h = out_pil_x.size
        #radian 단위로 변환
        theta = math.radians(in_theta)

        #원본 이미지 모서리 좌표 (제 1, 2, 4 사분면 지점, 원점 = 이미지 중앙)
        x_1, y_1 = in_r_w / 2 , in_r_h / 2
        x_2, y_2 = (-1) * x_1, y_1
        x_4, y_4 = x_1, (-1) * y_1
        
        #좌표 회전변환
        sin_theta, cos_theta = math.sin(theta), math.cos(theta)
        #제 1 사분면 좌표 변환
        a_1, b_1 = cos_theta * x_1 - sin_theta * y_1, sin_theta * x_1 + cos_theta * y_1
        #제 2 사분면 좌표 변환
        a_2, b_2 = cos_theta * x_2 - sin_theta * y_2, sin_theta * x_2 + cos_theta * y_2
        #제 4 사분면 좌표 변환
        a_4, b_4 = cos_theta * x_4 - sin_theta * y_4, sin_theta * x_4 + cos_theta * y_4
        
        #제 1 사분면 교점(right) 계산
        r_m_1 = y_1 / x_1
        r_m_2 = (b_4 - b_1) / (a_4 - a_1)
        r_k_2 = b_1 - r_m_2 * a_1
        
        node_x_1 = r_k_2 / (r_m_1 - r_m_2)
        node_y_1 = node_x_1 * r_m_1
        
        #제 2 사분면 교점 (left) 계산
        l_m_1 = y_2 / x_2
        l_m_2 = (b_2 - b_1) / (a_2 - a_1)
        l_k_2 = b_1 - l_m_2 * a_1
        
        node_x_2 = l_k_2 / (l_m_1 - l_m_2)
        node_y_2 = node_x_2 * l_m_1
        
        #원점으로부터 각 교점까지의 거리
        dist_right = math.sqrt(math.pow(node_x_1, 2) + math.pow(node_y_1, 2))
        dist_left  = math.sqrt(math.pow(node_x_2, 2) + math.pow(node_y_2, 2))
        
        #원점으로부터 원본 이미지의 한 모서리 점까지 거리
        dist_input = math.sqrt(math.pow(x_1, 2) + math.pow(y_1, 2))
        
        #원점으로부터 최종 이미지의 한 모서리 점까지 거리
        if dist_right < dist_left:
            dist_output = dist_right
        else:
            dist_output = dist_left
        
        #결과물 이미지 크기
        out_w, out_h = in_r_w * (dist_output / dist_input), in_r_h * (dist_output / dist_input)
        

        #---변환 시행
        out_pil_x_rotate = out_pil_x.rotate(in_theta, resample = in_option_rotate_x, expand = True)
        if is_y_in:
            out_pil_y_rotate = out_pil_y.rotate(in_theta, resample = in_option_rotate_y, expand = True)
        
        
        rotate_w, rotate_h = out_pil_x_rotate.size
        
        c_left  = int((rotate_w - out_w) / 2)
        c_right = int((rotate_w + out_w) / 2)
        c_upper = int((rotate_h - out_h) / 2)
        c_lower = int((rotate_h + out_h) / 2)
        
        out_pil_x = out_pil_x_rotate.crop((c_left, c_upper, c_right, c_lower))
        if is_y_in:
            out_pil_y = out_pil_y_rotate.crop((c_left, c_upper, c_right, c_lower))
        
        
        #반전 복원
        if flag_flip != 0:
            out_pil_x = out_pil_x.transpose(Image.FLIP_LEFT_RIGHT)
            if is_y_in:
                out_pil_y = out_pil_y.transpose(Image.FLIP_LEFT_RIGHT)
        #>>> >>>
        
    else: 
        #no rotate
        pass
        #print("pass rotate")
    #>>> rotate 시행
    
    #크기 보정 시행
    output_x_w, output_x_h = out_pil_x.size
    if input_x_w != output_x_w or input_x_h != output_x_h:
        #print(name_func, "input x size fixed (",output_x_w, output_x_h, ") -> (", input_x_w, input_x_h,")")
        out_pil_x = out_pil_x.resize((input_x_w, input_x_h), in_option_resize_x)
    
    if is_y_in:
        output_y_w, output_y_h = out_pil_y.size
        if input_y_w != output_y_w or input_y_h != output_y_h:
            #print(name_func, "input y size fixed (",output_y_w, output_y_h, ") -> (", input_y_w, input_y_h,")")
            out_pil_y = out_pil_y.resize((input_y_w, input_y_h), in_option_resize_y)
        
        if is_return_options:
            return out_pil_x, out_pil_y, return_option
        else:
            return out_pil_x, out_pil_y
    else:
        if is_return_options:
            return out_pil_x, return_option
        else:
            return out_pil_x
'''


#Data Augm- 종합 함수 (좌우 flip, 랜덤 crop, 랜덤 rotate)
#사용예시
'''
in_pil_x_augm, in_pil_x_lr_augm, in_pil_y_augm, str_used_option = pil_augm_v3(in_pil_x = in_pil_x_raw
                                                                             ,in_pil_x_lr = in_pil_x_lr_raw
                                                                             ,in_option_resize_x = Image.LANCZOS
                                                                             ,in_option_rotate_x = Image.BICUBIC
                                                                             ,in_pil_y = in_pil_y_raw
                                                                             ,in_option_resize_y = Image.NEAREST
                                                                             ,in_option_rotate_y = Image.NEAREST
                                                                             ,in_crop_wh_max = 10
                                                                             ,in_crop_wh_min = 4
                                                                             ,in_rotate_degree_max = 5
                                                                             ,in_percent_flip = 50
                                                                             ,in_percent_crop = 70
                                                                             ,in_percent_rotate = 20
                                                                             ,is_return_options = True
                                                                             )
'''

#random_in_ticket 계열값 전부 제거 후, random.uniform으로 대체함
#입력 이미지 크기와 출력 이미지 크기가 다른 경우, resize 기능 추가됨
# 이제 LR 이미지도 동시에 처리 가능함
# 2022-09-01: 0' 90' 135' 회전 시에 0으로 나누는 문제 발생 확인됨
# 0'    r_m_2 = (b_4 - b_1) / (a_4 - a_1)   -> 발생할 수 없는 각도
# 90'   l_m_2 = (b_2 - b_1) / (a_2 - a_1)   -> 각도 범위 제한을 통해 제어
# 135'  node_x_1 = r_k_2 / (r_m_1 - r_m_2)  -> 각도 범위 제한을 통해 제어
# 정사각형 이미지가 아닌 경우, 90'를 넘는 회전에 대해 이미지 가로세로 비율이 손상됨에 따라 각도범위를 90도 미만으로 제한함
def pil_augm_v3(**kargs):
    name_func = "[pil_augm_v3] -->"
    # [입력 x (image) 관련]---
    in_pil_x = kargs['in_pil_x']                                                    # (pil) 입력 이미지 x (HR 이미지)
    input_x_w, input_x_h = in_pil_x.size                                            # 최종 이미지 크기 확인용
    
    try:
        in_pil_x_lr = kargs['in_pil_x_lr']                                          # (pil) 입력 이미지 x (LR 이미지)
        input_x_lr_w, input_x_lr_h = in_pil_x_lr.size                               # 최종 이미지 크기 확인용
        # 연산 편의를 위해 크기를 HR과 동일하게 키움 (의도치않은 손실 최소화를 위해 LANCZOS 사용)
        in_pil_x_lr = in_pil_x_lr.resize((int(input_x_w), int(input_x_h)), Image.LANCZOS)
        is_x_lr_in = True
    except:
        is_x_lr_in = False
    
    #https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters
    # resize 시행 시 옵션 (Image.NEAREST, PIL.Image.BILINEAR, ... , Image.LANCZOS)
    in_option_resize_x = kargs['in_option_resize_x']
    
    # rotate 시행 시 옵션 (Image.NEAREST, Image.BILINEAR, Image.BICUBIC 중 선택)
    in_option_rotate_x = kargs['in_option_rotate_x']
    
    # [입력 y 관련]---
    try:
        in_pil_y = kargs['in_pil_y']                                                # (pil) 입력 이미지 y (라벨)
        in_option_resize_y = kargs['in_option_resize_y']
        in_option_rotate_y = kargs['in_option_rotate_y']
        is_y_in = True
        input_y_w, input_y_h = in_pil_y.size                                        # 최종 이미지 크기 확인용
    except:
        in_pil_y = in_pil_x
        in_option_resize_y = in_option_resize_x
        in_option_rotate_y = in_option_rotate_x
        is_y_in = False
    
    #[x y 공용]---
    in_crop_wh_max = kargs['in_crop_wh_max']                                        # crop 시 잘라낼 영역의 너비 & 높이 최대 값
    in_crop_wh_min = kargs['in_crop_wh_min']                                        # crop 시 잘라낼 영역의 너비 & 높이 최소 값
    
    in_rotate_degree_max = kargs['in_rotate_degree_max']                            # rotate 시 최대 각도
    
    if in_rotate_degree_max >= 90:
        print(name_func, "이 함수는 90' 미만의 회전각도만 지원합니다.")
        sys.exit(-9)
    
    in_percent_flip = kargs['in_percent_flip']                                      # (int) flip을 시행할 확률 (0 ~ 100)
    in_percent_crop = kargs['in_percent_crop']                                      # (int) crop을 시행할 확률 (0 ~ 100)
    in_percent_rotate = kargs['in_percent_rotate']                                  # (int) flip을 시행할 확률
    
    try:
        is_return_options = kargs['is_return_options']                              # 사용된 옵션 기록 return 여부
    except:
        is_return_options = False
    
    #----------
    in_w, in_h = in_pil_x.size                                                      # 입력 이미지 크기
    
    #시행여부: flip
    if in_percent_flip > random.uniform(0, 100):
        is_flip = True
        return_option = "Flip O"
    else:
        is_flip = False
        return_option = "Flip X"
    
    #시행여부: crop
    if in_percent_crop > random.uniform(0, 100):
        is_crop = True
        crop_zone = int(random.uniform(0, 100) % 10)                                # crop 구역 지정 (0 ~ 9)
        in_crop_wh = int(random.uniform(in_crop_wh_min, in_crop_wh_max))            # crop 시 제거될 영역의 너비
        return_option += " / Crop " + str(crop_zone) + " " + str(in_crop_wh)
    else:
        is_crop = False
        return_option += " / Crop X"
    
    #시행여부: rotate
    if in_percent_rotate > random.uniform(0, 100):
        is_rotate = True
        in_theta = int(random.uniform(0, in_rotate_degree_max - 1)) + 1         # 1 ~ max 값 생성
        if 50 > random.uniform(0, 100):
            in_theta = in_theta * (-1)
            
        return_option += " / Rotate " + str(in_theta)
    else:
        is_rotate = False
        return_option += " / Rotate X"
        
    #<<< flip 시행 (in_pil -> out_pil_x)
    if is_flip:
        out_pil_x = in_pil_x.transpose(Image.FLIP_LEFT_RIGHT)
        if is_x_lr_in:
            out_pil_x_lr = in_pil_x_lr.transpose(Image.FLIP_LEFT_RIGHT)
        if is_y_in:
            out_pil_y = in_pil_y.transpose(Image.FLIP_LEFT_RIGHT)
    else: #no flip
        out_pil_x = in_pil_x.copy()
        if is_x_lr_in:
            out_pil_x_lr = in_pil_x_lr.copy()
        if is_y_in:
            out_pil_y = in_pil_y.copy()
    #>>> flip 시행
    
    #<<< crop 시행 (out_pil_x -> out_pil_x)
    if is_crop:
        out_pil_x = out_pil_x.resize((int(in_w + in_crop_wh), int(in_h + in_crop_wh)), in_option_resize_x)
        if is_x_lr_in:
            out_pil_x_lr = out_pil_x_lr.resize((int(in_w + in_crop_wh), int(in_h + in_crop_wh)), in_option_resize_x)
        if is_y_in:
            out_pil_y = out_pil_y.resize((int(in_w + in_crop_wh), int(in_h + in_crop_wh)), in_option_resize_y)
        
        #crop - 키패드 기준
        if crop_zone == 1:                                                          # crop (1)
            c_left  = 0
            c_right = in_w
            c_upper = 0 + in_crop_wh
            c_lower = in_h + in_crop_wh

        elif crop_zone == 2:                                                        # crop (2)
            c_left  = 0 + in_crop_wh / 2
            c_right = in_w + in_crop_wh / 2
            c_upper = 0 + in_crop_wh
            c_lower = in_h + in_crop_wh

        elif crop_zone == 3:                                                        # crop (3)
            c_left  = 0 + in_crop_wh
            c_right = in_w + in_crop_wh
            c_upper = 0 + in_crop_wh
            c_lower = in_h + in_crop_wh

        elif crop_zone == 4:                                                        # crop (4)
            c_left  = 0
            c_right = in_w
            c_upper = 0 + in_crop_wh / 2
            c_lower = in_h + in_crop_wh /2

        elif crop_zone == 6:                                                        # crop (6)
            c_left  = 0 + in_crop_wh
            c_right = in_w + in_crop_wh
            c_upper = 0 + in_crop_wh / 2
            c_lower = in_h + in_crop_wh / 2

        elif crop_zone == 7:                                                        # crop (7)
            c_left  = 0
            c_right = in_w
            c_upper = 0
            c_lower = in_h

        elif crop_zone == 8:                                                        # crop (8)
            c_left  = 0 + in_crop_wh / 2
            c_right = in_w + in_crop_wh / 2
            c_upper = 0
            c_lower = in_h

        elif crop_zone == 9:                                                        # crop (9)
            c_left  = 0 + in_crop_wh
            c_right = in_w + in_crop_wh
            c_upper = 0 
            c_lower = in_h

        else:                                                                       # crop (0 & 5) -> center crop
            c_left  = 0 + in_crop_wh / 2
            c_right = in_w + in_crop_wh / 2
            c_upper = 0 + in_crop_wh / 2
            c_lower = in_h + in_crop_wh / 2

        out_pil_x = out_pil_x.crop((int(c_left), int(c_upper), int(c_right), int(c_lower)))
        if is_x_lr_in:
            out_pil_x_lr = out_pil_x_lr.crop((int(c_left), int(c_upper), int(c_right), int(c_lower)))
        if is_y_in:
            out_pil_y = out_pil_y.crop((int(c_left), int(c_upper), int(c_right), int(c_lower)))
        
    else:   # no crop
        pass
    #>>> crop 시행
    
    #<<< rotate 시행 (out_pil_x -> out_pil_x)
    if is_rotate:
        #<<< <<<
        #----플래그
        flag_flip = 0                                                               # 입력 반전 시행 여부
        
        #---예외처리
        #역방향 (시계방향) 변환옵션 사용여부 확인
        if in_theta < 0:
            out_pil_x = out_pil_x.transpose(Image.FLIP_LEFT_RIGHT)
            if is_x_lr_in:
                out_pil_x_lr = out_pil_x_lr.transpose(Image.FLIP_LEFT_RIGHT)
            if is_y_in:
                out_pil_y = out_pil_y.transpose(Image.FLIP_LEFT_RIGHT)
            
            in_theta = in_theta * (-1)
            flag_flip = 1
        
        if in_theta >= 360:
            in_theta = in_theta % 360

        check_theta = in_theta // 90

        #90도 이상 각도에 대한 처리 시행
        if check_theta > 0:
            #print("회전보정 시행")
            out_pil_x = out_pil_x.rotate(check_theta * 90, resample = in_option_rotate_x, expand = True)
            if is_x_lr_in:
                out_pil_x_lr = out_pil_x_lr.rotate(check_theta * 90, resample = in_option_rotate_x, expand = True)
            if is_y_in:
                out_pil_y = out_pil_y.rotate(check_theta * 90, resample = in_option_rotate_y, expand = True)
            in_theta = in_theta % 90
            #flag_rotate = check_theta
        
        #print("예외처리 완료", out_pil_x.size)

        #---계산 시행
        in_r_w, in_r_h = out_pil_x.size                                             # 입력 이미지 크기 확인
        
        # radian 단위로 변환
        theta = math.radians(in_theta)

        #원본 이미지 모서리 좌표 (제 1, 2, 4 사분면 지점, 원점 = 이미지 중앙)
        x_1, y_1 = in_r_w / 2 , in_r_h / 2
        x_2, y_2 = (-1) * x_1, y_1
        x_4, y_4 = x_1, (-1) * y_1
        
        #좌표 회전변환
        sin_theta, cos_theta = math.sin(theta), math.cos(theta)
        #제 1 사분면 좌표 변환
        a_1, b_1 = cos_theta * x_1 - sin_theta * y_1, sin_theta * x_1 + cos_theta * y_1
        #제 2 사분면 좌표 변환
        a_2, b_2 = cos_theta * x_2 - sin_theta * y_2, sin_theta * x_2 + cos_theta * y_2
        #제 4 사분면 좌표 변환
        a_4, b_4 = cos_theta * x_4 - sin_theta * y_4, sin_theta * x_4 + cos_theta * y_4
        
        #제 1 사분면 교점(right) 계산
        r_m_1 = y_1 / x_1
        r_m_2 = (b_4 - b_1) / (a_4 - a_1)
        r_k_2 = b_1 - r_m_2 * a_1
        
        node_x_1 = r_k_2 / (r_m_1 - r_m_2)
        node_y_1 = node_x_1 * r_m_1
        
        #제 2 사분면 교점 (left) 계산
        l_m_1 = y_2 / x_2
        l_m_2 = (b_2 - b_1) / (a_2 - a_1)
        l_k_2 = b_1 - l_m_2 * a_1
        
        node_x_2 = l_k_2 / (l_m_1 - l_m_2)
        node_y_2 = node_x_2 * l_m_1
        
        #원점으로부터 각 교점까지의 거리
        dist_right = math.sqrt(math.pow(node_x_1, 2) + math.pow(node_y_1, 2))
        dist_left  = math.sqrt(math.pow(node_x_2, 2) + math.pow(node_y_2, 2))
        
        #원점으로부터 원본 이미지의 한 모서리 점까지 거리
        dist_input = math.sqrt(math.pow(x_1, 2) + math.pow(y_1, 2))
        
        #원점으로부터 최종 이미지의 한 모서리 점까지 거리
        if dist_right < dist_left:
            dist_output = dist_right
        else:
            dist_output = dist_left
        
        #결과물 이미지 크기
        out_w, out_h = in_r_w * (dist_output / dist_input), in_r_h * (dist_output / dist_input)
        

        #---변환 시행
        out_pil_x_rotate = out_pil_x.rotate(in_theta, resample = in_option_rotate_x, expand = True)
        if is_x_lr_in:
            out_pil_x_lr_rotate = out_pil_x_lr.rotate(in_theta, resample = in_option_rotate_x, expand = True)
        if is_y_in:
            out_pil_y_rotate = out_pil_y.rotate(in_theta, resample = in_option_rotate_y, expand = True)
        
        rotate_w, rotate_h = out_pil_x_rotate.size
        
        c_left  = int((rotate_w - out_w) / 2)
        c_right = int((rotate_w + out_w) / 2)
        c_upper = int((rotate_h - out_h) / 2)
        c_lower = int((rotate_h + out_h) / 2)
        
        out_pil_x = out_pil_x_rotate.crop((c_left, c_upper, c_right, c_lower))
        if is_x_lr_in:
            out_pil_x_lr = out_pil_x_lr_rotate.crop((c_left, c_upper, c_right, c_lower))
        if is_y_in:
            out_pil_y = out_pil_y_rotate.crop((c_left, c_upper, c_right, c_lower))
        
        #반전 복원
        if flag_flip != 0:
            out_pil_x = out_pil_x.transpose(Image.FLIP_LEFT_RIGHT)
            if is_x_lr_in:
                out_pil_x_lr = out_pil_x_lr.transpose(Image.FLIP_LEFT_RIGHT)
            if is_y_in:
                out_pil_y = out_pil_y.transpose(Image.FLIP_LEFT_RIGHT)
        #>>> >>>
        
    else: 
        #no rotate
        pass
    #>>> rotate 시행
    
    #크기 보정 시행
    output_x_w, output_x_h = out_pil_x.size
    if input_x_w != output_x_w or input_x_h != output_x_h:
        #print(name_func, "input x size fixed (",output_x_w, output_x_h, ") -> (", input_x_w, input_x_h,")")
        out_pil_x = out_pil_x.resize((input_x_w, input_x_h), in_option_resize_x)
    
    if is_x_lr_in:
        # 본래 크기로 축소
        out_pil_x_lr = out_pil_x_lr.resize((int(input_x_lr_w), int(input_x_lr_h)), Image.LANCZOS)
    
    if is_y_in:
        output_y_w, output_y_h = out_pil_y.size
        if input_y_w != output_y_w or input_y_h != output_y_h:
            #print(name_func, "input y size fixed (",output_y_w, output_y_h, ") -> (", input_y_w, input_y_h,")")
            out_pil_y = out_pil_y.resize((input_y_w, input_y_h), in_option_resize_y)
    
    
    if is_return_options:
        if is_x_lr_in and is_y_in:
            return out_pil_x, out_pil_x_lr, out_pil_y, return_option
        elif is_x_lr_in:
            return out_pil_x, out_pil_x_lr, return_option
        elif is_y_in:
            return out_pil_x, out_pil_y, return_option
        else:
            return out_pil_x, return_option
    else:
        if is_x_lr_in and is_y_in:
            return out_pil_x, out_pil_x_lr, out_pil_y
        elif is_x_lr_in:
            return out_pil_x, out_pil_x_lr
        elif is_y_in:
            return out_pil_x, out_pil_y
        else:
            return out_pil_x
    
    

#=== end of pil_augm_v3

def pil_augm_lite(in_pil_hr, in_pil_lr, get_info=False):
    # 일반적인 SR 모델 학습법에 사용되는 Random Horizontal Flip + 90' 단위 Rotation 구현
    # 일반적인 경우를 가정하고 작성하였으므로 정사각형 이미지에 대해서만 가동을 보장함
    
    w_hr, h_hr = in_pil_hr.size
    w_lr, h_lr = in_pil_lr.size
    
    if w_hr != h_hr or w_lr != h_lr:
        print("(exc) [pil_augm_lite] -> 입력 이미지가 정사각형이 아닙니다!")
        sys.exit(-9)
    
    mirror = bool(int(np.random.choice([0, 1])))
    
    if mirror:
        in_pil_hr = in_pil_hr.transpose(Image.FLIP_LEFT_RIGHT)
        in_pil_lr = in_pil_lr.transpose(Image.FLIP_LEFT_RIGHT)
        return_option = "Flip O"
    else:
        return_option = "Flip X"
    
    degree = int(np.random.choice([0, 90, 180, 270]))
    
    if get_info:
        if degree == 0:
            return_option += " / Rotate X"
            return in_pil_hr, in_pil_lr, return_option
        else:
            return_option += " / Rotate " + str(degree)
            return in_pil_hr.rotate(degree), in_pil_lr.rotate(degree), return_option
    else:
        if degree == 0:
            return in_pil_hr, in_pil_lr
        else:
            return in_pil_hr.rotate(degree), in_pil_lr.rotate(degree)


#@@@
def degradation_total_v7(**kargs):
    func_name = "[degradation_total_v7] -->"
    #--- 변경사항
    # 
    #   1. Degradation 1회만 시행
    #   2. Degradation 결과물을 원본 크기로 복원하는 단계 시행 안함
    #      -> scale_factor 만큼 그대로 작아진 결과물 배출
    #   3. 노이즈 (Color / Gray) 생성 방식은 Real-ESRGAN 방식 그대로 적용
    #      -> Color: RGB 채널에 서로 다른 노이즈 생성 
    #      -> Gray : RGB 채널에 서로 같은 노이즈 생성
    #---
    
    '''
    사용 예시
    pil_img = degradation_total_v7(in_pil =
                                  ,is_return_options = 
                                  #--블러
                                  ,in_option_blur = "Gaussian"
                                  #-- 다운 스케일
                                  ,in_scale_factor = 
                                  ,in_option_resize = 
                                  #--노이즈
                                  ,in_option_noise = "Gaussian"
                                  #노이즈 시그마값 범위 (tuple)
                                  ,in_range_noise_sigma = 
                                  #Gray 노이즈 확룔 (int)
                                  ,in_percent_gray_noise = 
                                  #노이즈 고정값 옵션
                                  ,is_fixed_noise = 
                                  ,in_fixed_noise_channel =
                                  ,in_fixed_noise_sigma   = 
                                  )
    '''

    #IN(**):
    #       (선택, str)        in_path                 : "/..."
    #       (대체 옵션, pil)    in_pil                  : pil_img

    #고정값   (str)             in_option_blur         : "Gaussian"
    #       (선택, 2d npArray) kernel_blur            : np커널
    #
    #중지됨   (int or tuple)    in_resolution          : (사용금지) in_scale_factor로 변경됨
    #       (int or tuple)    in_scale_factor        : 스케일 팩터 (1 ~ ) -> tuple 입력시, 범위 내 균등추출 (소수점 1자리까지 사용)
    #       (str)             in_option_resize       : "AREA", "BILINEAR", "BICUBIC"
    #
    #고정값   (str)             in_option_noise        : "Gaussian"
    #       (tuple)           in_range_noise_sigma   : ((float), (float))
    #       (int)             in_percent_gray_noise  : Gray 노이즈 확률 (그 외엔 Color 노이즈로 생성), 최대 100

    #       (선택, bool)       is_fixed_noise         : 노이즈 옵션지정여부 (val & test용 , default = False)
    #       (선택, str)        in_fixed_noise_channel : 노이즈 발생 채널 지정 (val & test용) ("Color" or "Gray")
    #       (선택, str)        in_fixed_noise_sigma   : 노이즈 시그마값 지정  (val & test용)

    #       (bool)            is_return_options      : degrad- 옵션 return 여부

    #OUT(1):
    #       (PIL)             이미지
    #       (선택, str)             Degradation 옵션
    #--- --- ---

    #degrad- 옵션 return 여부
    try:
        is_return_options = kargs['is_return_options']
    except:
        is_return_options = False
    
    #(str) 사용된 degrad 옵션 저장
    return_option = ""
    
    #(str) 파일 경로 or (pil) 이미지 입력받음
    try:
        in_path = kargs['in_path']
        in_cv = cv2.imread(in_path)
    except:
        in_cv = cv2.cvtColor(np.array(kargs['in_pil']), cv2.COLOR_RGB2BGR)
    
    #입력 이미지 크기
    in_h, in_w, _ = in_cv.shape
    
    #***--- degradation_blur
    #(str) blur 방식
    in_option_blur = kargs['in_option_blur']
    
    return_option += "Blur = " + in_option_blur
    
    #평균 필터
    if in_option_blur == "Mean" or in_option_blur == "mean":
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size) / (kernel_size * kernel_size))
        out_cv_blur = cv2.filter2D(in_cv, -1, kernel)
    
    #가우시안 필터
    elif in_option_blur == "Gaussian" or in_option_blur == "gaussian":
        kernel_size = 3 #홀수만 가능
        kernel_sigma = 0.1
        kernel = cv2.getGaussianKernel(kernel_size, kernel_sigma) * cv2.getGaussianKernel(kernel_size, kernel_sigma).T
        out_cv_blur = cv2.filter2D(in_cv, -1, kernel)
    
    #기타 필터 (sinc 용)
    elif in_option_blur == "Custom" or in_option_blur == "custom":
        kernel = kargs['kernel_blur']
        out_cv_blur = cv2.filter2D(in_cv, -1, kernel)
    
    #***--- degradation_resolution
    
    #scale factor (float, 소수점 2자리 까지 사용)
    if type(kargs['in_scale_factor']) == type((0, 1)):
        in_scale_factor = round(random.uniform(kargs['in_scale_factor'][0], kargs['in_scale_factor'][-1]), 2)
    else:
        in_scale_factor = round(kargs['in_scale_factor'], 2)
    
    #최소값 clipping
    min_scale_factor = 0.25
    if in_scale_factor < min_scale_factor:
        print(func_name, "scale factor clipped to", min_scale_factor)
        in_scale_factor = min_scale_factor
    
    #(str) resize 옵션 ("AREA", "BILINEAR", "BICUBIC" / 소문자 가능)
    try:
        in_option_resize = kargs['in_option_resize']
    except:
        #default: BILINEAR
        in_option_resize = "BILINEAR"
    
    tmp_s_f = 1 / in_scale_factor
    if in_option_resize == "AREA" or in_option_resize == "area":
        tmp_interpolation = cv2.INTER_AREA
    elif in_option_resize == "BILINEAR" or in_option_resize == "bilinear":
        tmp_interpolation = cv2.INTER_LINEAR
    elif in_option_resize == "BICUBIC" or in_option_resize == "bicubic":
        tmp_interpolation = cv2.INTER_CUBIC
    
    out_cv_resize = cv2.resize(out_cv_blur, dsize=(0,0), fx=tmp_s_f, fy=tmp_s_f
                                  ,interpolation = tmp_interpolation
                                  )
    
    #감소된 크기 계산
    out_h, out_w, _ = out_cv_resize.shape
    
    return_option += ", Downscale(x" + str(in_scale_factor) + ") = " + in_option_resize
    
    #***--- degradation 노이즈 추가 (Color or Gray)
    
    #채널 분할
    in_cv_b, in_cv_g, in_cv_r = cv2.split(out_cv_resize)
    
    #노이즈 옵션 고정값 사용여부
    try:
        is_fixed_noise = kargs['is_fixed_noise']
    except:
        is_fixed_noise = False
    
    #노이즈 종류 선택 (Gaussian or Poisson) -> Poisson 사용 안함
    try:
        in_option_noise = kargs['in_option_noise']
    except:
        in_option_noise = "Gaussian"
    
    #노이즈 생성 (Gaussian)
    if in_option_noise == "Gaussian":
        in_noise_mu = 0 #뮤 =고정값 적용
        
        #노이즈 옵션이 지정된 경우
        if is_fixed_noise:
            #노이즈 발생 채널
            in_noise_channel = kargs['in_fixed_noise_channel']
            #시그마 값
            in_noise_sigma = int(kargs['in_fixed_noise_sigma'])
        #노이즈 옵션이 지정되지 않은 경우
        else:
            #노이즈 발생 채널 추첨
            in_percent_gray_noise = kargs['in_percent_gray_noise']
            in_noise_channel = random.choices(["Color", "Gray"]
                                             ,weights = [(100 - in_percent_gray_noise), in_percent_gray_noise]
                                             ,k = 1
                                             )[0]
            #시그마 값
            in_noise_sigma = int(random.uniform(kargs['in_range_noise_sigma'][0], kargs['in_range_noise_sigma'][-1]))
        
        #Color 노이즈 발생 (채널별 다른 노이즈 발생)
        if in_noise_channel == "Color":
            noise_r = np.random.normal(in_noise_mu, in_noise_sigma, size=(out_h, out_w))
            noise_g = np.random.normal(in_noise_mu, in_noise_sigma, size=(out_h, out_w))
            noise_b = np.random.normal(in_noise_mu, in_noise_sigma, size=(out_h, out_w))
        #Gray 노이즈 발생 (모든 채널 동일 노이즈 발생)
        elif in_noise_channel == "Gray":
            noise_r = np.random.normal(in_noise_mu, in_noise_sigma, size=(out_h, out_w))
            noise_g = noise_r
            noise_b = noise_r
        
        out_cv_r = np.uint8(np.clip(in_cv_r + noise_r, 0, 255))
        out_cv_g = np.uint8(np.clip(in_cv_g + noise_g, 0, 255))
        out_cv_b = np.uint8(np.clip(in_cv_b + noise_b, 0, 255))
    
    #채널 재 병합
    out_cv_noise = cv2.merge((out_cv_r, out_cv_g, out_cv_b))
    
    #생성옵션 기록갱신
    return_option += ", Noise = (" + in_option_noise + ", " + in_noise_channel
    return_option += ", mu = " + str(in_noise_mu) + ", sigma = " + str(in_noise_sigma) + ")"
    
    if is_return_options:
        #(PIL), (str)
        return Image.fromarray(out_cv_noise) , return_option
    else:
        #(PIL)
        return Image.fromarray(out_cv_noise)


#=== end of degradation_total_v7

def pil_2_patch_v5(**kargs):
    #이미지 resize 후 crop (hr & lr patch dict 생성)
    '''
    dict_patch_hr, dict_patch_lr = pil_2_patch_v5(in_pil_hr = in_pil_y_input
                                                 ,in_pil_lr = in_pil_x_input
                                                 ,in_scale_factor = HP_DG_SCALE_FACTOR
                                                 #val 모드의 경우, center crop 1장만 생성
                                                 ,batch_size = current_batch_size
                                                 ,strides = HP_SR_STRIDES
                                                 ,patch_size = (HP_SR_PATCH_IMG_W, HP_SR_PATCH_IMG_H)
                                                 ,crop_init_coor = (crop_init_coor_w,crop_init_coor_h)
                                                 ,is_val = tmp_is_val
                                                 )
    '''
    
    name_func = "[pil_2_patch_v5] ->"
    in_pil_hr = kargs['in_pil_hr'] #(pil) High Resolution 이미지
    in_pil_lr = kargs['in_pil_lr'] #(pil) Low Resolution 이미지
    
    try:
    #scale factor
        #나눗셈 연산 안정성을 위해 1 또는 짝수 (2, 4, ...) 권장
        in_sf = kargs['in_scale_factor']
    except:
        in_sf = 1
    
    try:
        batch_size = kargs['batch_size'] #(int) Return할 Patch 개수 (test 모드에선 사용 안함)
    except:
        batch_size = -1 #해당 기능이 사용되지 않은 경우의 설정값 = -1
    
    strides = kargs['strides'] #(tuple) stride 값 (w, h)
    patch_size = kargs['patch_size'] #(tuple) 이미지 크기 (w, h)
    
    try:
        crop_init_center_w, crop_init_center_h = kargs['crop_init_coor'] #(tuple) crop 시작좌표 (w, h)
    except:
        crop_init_center_w, crop_init_center_h = 0, 0
    
    try:
        is_val = kargs['is_val'] #(bool) val 여부 (center crop만 시행)
    except:
        is_val = False
    
    
    
    #입력 이미지 크기 (나눗셈 관련  misalignment 문제 고려하여 Small size - LR image 기준으로 생성)
    in_w, in_h = in_pil_lr.size
    
    #patch 이미지 크기
    p_w, p_h = int(patch_size[0]), int(patch_size[-1])
    
    #stride 값
    strides_w, strides_h = int(strides[0]), int(strides[-1])
    
    diff_w = in_w - p_w
    diff_h = in_h - p_h
    
    
    #--- 이미지 patch 단일 데이터 dict 생성 (val)
    if is_val:
        #(dict) 이미지 patch 묶음 (Higb Resolution, Low Resolution)
        dict_patch_hr = {}
        dict_patch_lr = {}
        #center crop 좌표 지정
        tuple_crop_range_hr = (int(in_sf * (in_w//2 - p_w//2))
                              ,int(in_sf * (in_h//2 - p_h//2))
                              ,int(in_sf * (in_w//2 + p_w//2))
                              ,int(in_sf * (in_h//2 + p_h//2))
                              )
        
        tuple_crop_range_lr = (int(in_w//2 - p_w//2)
                              ,int(in_h//2 - p_h//2)
                              ,int(in_w//2 + p_w//2)
                              ,int(in_h//2 + p_h//2)
                              )
        
        
        dict_patch_hr[0] = in_pil_hr.crop(tuple_crop_range_hr)
        dict_patch_lr[0] = in_pil_lr.crop(tuple_crop_range_lr)
        
        return dict_patch_hr, dict_patch_lr
    
    
    #---------------------------- train 모드의 경우
    
    #입력 이미지 크기 (나눗셈 관련  misalignment 문제 고려하여 Small size - LR image 기준으로 생성)
    in_w, in_h = in_pil_lr.size
    
    #patch 크기 절반
    p_half_w = p_w // 2
    p_half_h = p_h // 2
    
    #<<< @@@
    #유효 crop area 리스트
    list_crop_area = []
    
    tmp_center_h = crop_init_center_h
    while tmp_center_h < in_h:
        tmp_center_w = crop_init_center_w
        while tmp_center_w < in_w:
            
            if tmp_center_h - p_half_h >= 0 and tmp_center_h + p_half_h <= in_h:
                if tmp_center_w - p_half_w >= 0 and tmp_center_w + p_half_w <= in_w:
                    #HR crop area
                    center_coor_hr = (int(in_sf * (tmp_center_w - p_half_w))
                                     ,int(in_sf * (tmp_center_h - p_half_h))
                                     ,int(in_sf * (tmp_center_w + p_half_w))
                                     ,int(in_sf * (tmp_center_h + p_half_h))
                                     )
                    #LR crop area
                    center_coor_lr = (int(tmp_center_w - p_half_w)
                                     ,int(tmp_center_h - p_half_h)
                                     ,int(tmp_center_w + p_half_w)
                                     ,int(tmp_center_h + p_half_h)
                                     )
                    
                    #print("HR:", center_coor_hr)
                    #print("LR:", center_coor_lr)
                    
                    #(HR crop area, LR crop area)
                    list_crop_area.append((center_coor_hr, center_coor_lr))
                    
            tmp_center_w += strides_w
        tmp_center_h += strides_h
    
    
    #유효 좌표 리스트 셔플
    random.shuffle(list_crop_area) #학습 효율을 위해 셔플
    
    
    #유효 좌표 리스트 중에 앞 n개만 patch로 생성 후 dict로 묶어서 return
    dict_patch_hr = {}
    dict_patch_lr = {}
    tmp_count_patch = 0
    for i_coor in list_crop_area:
        #coor 불러오기
        tuple_crop_range_hr = i_coor[0]
        tuple_crop_range_lr = i_coor[-1]
        #patch 생성
        dict_patch_hr[tmp_count_patch] = in_pil_hr.crop(tuple_crop_range_hr)
        dict_patch_lr[tmp_count_patch] = in_pil_lr.crop(tuple_crop_range_lr)
        
        tmp_count_patch += 1
        #현재 tmp_count_patch 값 = 생성된 patch 수
        
        #print("tmp_count_patch after +1:", tmp_count_patch)
        
        if tmp_count_patch >= batch_size:
            break
        
    
    #OUT(1) (dict) patch 묶음 
    return dict_patch_hr, dict_patch_lr 
    
    
    #>>> @@@

#=== End of pil_2_patch_v5

def pil_2_patch_v6(**kargs):
    #이미지 resize 후 crop (hr & lr patch dict 생성)
    '''
    dict_patch_hr, dict_patch_hr_label, dict_patch_lr = pil_2_patch_v6(in_pil_hr = in_pil_image_hr
                                                                       # 선택
                                                                      ,in_pil_hr_label = in_pil_label_hr
                                                                       # 선택
                                                                      ,in_pil_lr = in_pil_image_lr
                                                                       
                                                                      ,in_scale_factor = HP_DG_SCALE_FACTOR
                                                                       # val 모드의 경우, center crop 1장만 생성
                                                                      ,batch_size = current_batch_size
                                                                      ,strides = HP_SR_STRIDES
                                                                      ,patch_size = (HP_SR_PATCH_IMG_W, HP_SR_PATCH_IMG_H)
                                                                      ,crop_init_coor = (crop_init_coor_w,crop_init_coor_h)
                                                                      ,is_val = tmp_is_val
                                                                      )
    '''
    
    name_func = "[pil_2_patch_v6] ->"
    in_pil_hr = kargs['in_pil_hr']                                                  # (pil) High Resolution 이미지
    
    try:
        in_pil_hr_label = kargs['in_pil_hr_label']                                  # (pil) High Resolution 라벨
        is_return_lab_hr = True
    except:
        is_return_lab_hr = False
    
    try:
        in_pil_lr = kargs['in_pil_lr']                                              # (pil) Low Resolution 이미지
        is_return_img_lr = True
    except:
        is_return_img_lr = False
    
    
    try:
    #scale factor
        #나눗셈 연산 안정성을 위해 1 또는 짝수 (2, 4, ...) 권장
        in_sf = kargs['in_scale_factor']
    except:
        in_sf = 1
    
    try:
        batch_size = kargs['batch_size']                                            # (int) Return할 Patch 개수 (test 모드에선 사용 안함)
    except:
        batch_size = -1 #해당 기능이 사용되지 않은 경우의 설정값 = -1 -> 입력이 필요한데 생략된 경우, 연산오류 발생시킴
    
    strides = kargs['strides'] #(tuple) stride 값 (w, h)
    patch_size = kargs['patch_size'] #(tuple) 이미지 크기 (w, h)
    
    try:
        crop_init_center_w, crop_init_center_h = kargs['crop_init_coor']            # (tuple) crop 시작좌표 (w, h)
    except:
        crop_init_center_w, crop_init_center_h = 0, 0
    
    try:
        is_val = kargs['is_val']                                                    # (bool) val 여부 (center crop만 시행)
    except:
        is_val = False
    
    
    
    #입력 이미지 크기 (나눗셈 관련  misalignment 문제 고려하여 Small size - LR image 기준으로 생성)
    if is_return_img_lr:
        in_w, in_h = in_pil_lr.size
    else:
        # lr 이미지가 입력되지 않은 경우, HR = LR 로 가정하고 크기 계산 시행
        in_sf = 1
        in_w, in_h = in_pil_hr.size
    
    #patch 이미지 크기
    p_w, p_h = int(patch_size[0]), int(patch_size[-1])
    
    #stride 값
    strides_w, strides_h = int(strides[0]), int(strides[-1])
    
    diff_w = in_w - p_w
    diff_h = in_h - p_h
    
    
    #--- 이미지 patch 단일 데이터 dict 생성 (val)
    if is_val:
        # (dict) 이미지 patch 묶음 (Higb Resolution (image, label), Low Resolution(image))
        dict_patch_hr = {}
        dict_patch_hr_label = {}
        dict_patch_lr = {}
        
        # center crop 좌표 지정
        tuple_crop_range_hr = (int(in_sf * (in_w//2 - p_w//2))
                              ,int(in_sf * (in_h//2 - p_h//2))
                              ,int(in_sf * (in_w//2 + p_w//2))
                              ,int(in_sf * (in_h//2 + p_h//2))
                              )
        
        tuple_crop_range_lr = (int(in_w//2 - p_w//2)
                              ,int(in_h//2 - p_h//2)
                              ,int(in_w//2 + p_w//2)
                              ,int(in_h//2 + p_h//2)
                              )
        
        
        dict_patch_hr[0] = in_pil_hr.crop(tuple_crop_range_hr)
        
        if is_return_lab_hr:
            dict_patch_hr_label[0] = in_pil_hr_label.crop(tuple_crop_range_hr)
        
        if is_return_img_lr:
            dict_patch_lr[0] = in_pil_lr.crop(tuple_crop_range_lr)
        
        if is_return_lab_hr and is_return_img_lr:
            return dict_patch_hr, dict_patch_hr_label, dict_patch_lr
        elif is_return_lab_hr:
            return dict_patch_hr, dict_patch_hr_label
        elif is_return_img_lr:
            return dict_patch_hr, dict_patch_lr
        else:
            return dict_patch_hr
        
    #---------------------------- train 모드의 경우
    
    #입력 이미지 크기 (나눗셈 관련  misalignment 문제 고려하여 Small size - LR image 기준으로 생성)
    if is_return_img_lr:
        in_w, in_h = in_pil_lr.size
    else:
        # lr 이미지가 입력되지 않은 경우, HR = LR 로 가정하고 크기 계산 시행
        in_sf = 1
        in_w, in_h = in_pil_hr.size
    
    #patch 크기 절반
    p_half_w = p_w // 2
    p_half_h = p_h // 2
    
    #<<< @@@
    #유효 crop area 리스트
    list_crop_area = []
    
    tmp_center_h = crop_init_center_h
    while tmp_center_h < in_h:
        tmp_center_w = crop_init_center_w
        while tmp_center_w < in_w:
            
            if tmp_center_h - p_half_h >= 0 and tmp_center_h + p_half_h <= in_h:
                if tmp_center_w - p_half_w >= 0 and tmp_center_w + p_half_w <= in_w:
                    #HR crop area
                    center_coor_hr = (int(in_sf * (tmp_center_w - p_half_w))
                                     ,int(in_sf * (tmp_center_h - p_half_h))
                                     ,int(in_sf * (tmp_center_w + p_half_w))
                                     ,int(in_sf * (tmp_center_h + p_half_h))
                                     )
                    #LR crop area
                    center_coor_lr = (int(tmp_center_w - p_half_w)
                                     ,int(tmp_center_h - p_half_h)
                                     ,int(tmp_center_w + p_half_w)
                                     ,int(tmp_center_h + p_half_h)
                                     )
                    
                    #print("HR:", center_coor_hr)
                    #print("LR:", center_coor_lr)
                    
                    #(HR crop area, LR crop area)
                    list_crop_area.append((center_coor_hr, center_coor_lr))
                    
            tmp_center_w += strides_w
        tmp_center_h += strides_h
    
    
    #유효 좌표 리스트 셔플
    random.shuffle(list_crop_area) #학습 효율을 위해 셔플
    
    
    #유효 좌표 리스트 중에 앞 n개만 patch로 생성 후 dict로 묶어서 return
    
    # (dict) 이미지 patch 묶음 (Higb Resolution (image, label), Low Resolution(image))
    dict_patch_hr = {}
    dict_patch_hr_label = {}
    dict_patch_lr = {}
    
    tmp_count_patch = 0
    for i_coor in list_crop_area:
        #coor 불러오기
        tuple_crop_range_hr = i_coor[0]
        tuple_crop_range_lr = i_coor[-1]
        #patch 생성
        dict_patch_hr[tmp_count_patch] = in_pil_hr.crop(tuple_crop_range_hr)
        if is_return_lab_hr:
            dict_patch_hr_label[tmp_count_patch] = in_pil_hr_label.crop(tuple_crop_range_hr)
        if is_return_img_lr:
            dict_patch_lr[tmp_count_patch] = in_pil_lr.crop(tuple_crop_range_lr)
        
        tmp_count_patch += 1
        #현재 tmp_count_patch 값 = 생성된 patch 수
        
        #print("tmp_count_patch after +1:", tmp_count_patch)
        
        if tmp_count_patch >= batch_size:
            break
        
    
    if len(dict_patch_hr) != batch_size:
        print(name_func, "dict_patch_hr has been generated fewer than required.")
        sys.exit(9)
    
    if is_return_lab_hr and len(dict_patch_hr_label) != batch_size:
        print(name_func, "dict_patch_hr_label has been generated fewer than required.")
        sys.exit(9)
        
    if is_return_img_lr and len(dict_patch_lr) != batch_size:
        print(name_func, "dict_patch_lr has been generated fewer than required.")
        sys.exit(9)
    
    # OUT (dict) patch 묶음 
    if is_return_lab_hr and is_return_img_lr:
        return dict_patch_hr, dict_patch_hr_label, dict_patch_lr
    elif is_return_lab_hr:
        return dict_patch_hr, dict_patch_hr_label
    elif is_return_img_lr:
        return dict_patch_hr, dict_patch_lr
    else:
        return dict_patch_hr
    
    
    #>>> @@@

#=== End of pil_2_patch_v6


def pil_marginer_v1(**kargs):
    # pil 이미지에 margin 추가
    '''
    pil_img_hr, pil_lab_hr, pil_img_lr = pil_marginer_v1(in_pil_hr          =
                                                        ,target_size_hr     =
                                                        ,img_background     =
                                                        ,is_random          =
                                                         # 선택 (HR Label)
                                                        ,in_pil_hr_label    = 
                                                        ,lab_background     =
                                                         # 선택 (LR Image)
                                                        ,in_pil_lr          =
                                                        ,in_scale_factor    = 
                                                        ,target_size_lr     =
                                                        )
    '''
    
    is_random            = kargs['is_random']           # (bool) margin 랜덤여부 결정
    
    # HR Image
    in_pil_hr           = kargs['in_pil_hr']            # (pil)
    _hr_w, _hr_h        = in_pil_hr.size
    _w, _h              = kargs['target_size_hr']       # (tuple with int) width, height
    target_size_hr      = (int(_w), int(_h))
    
    _diff_hr_w          = target_size_hr[0]  - _hr_w
    _diff_hr_h          = target_size_hr[-1] - _hr_h
    _r, _g, _b          = kargs['img_background']       # (tuple with int) (R, G, B) Range: 0 ~ 255
    img_background      = (int(min(255, max(0, _r))), int(min(255, max(0, _g))), int(min(255, max(0, _b))))
    
    
    # HR Label
    try:
        in_pil_hr_label = kargs['in_pil_hr_label']      # (pil)
        _n              = kargs['lab_background']       # (tuple with int) (n)
        lab_background  = (int(min(255, max(0, _n))))
        return_hr_label = True
    except:
        in_pil_hr_label = None
        lab_background  = None
        return_hr_label = False
    
    # LR Image
    try:
        in_pil_lr       = kargs['in_pil_lr']            # (pil)
        _lr_w, _lr_h    = in_pil_lr.size
        in_scale_factor = kargs['in_scale_factor']      # (int) scale factor
        _w, _h          = kargs['target_size_lr']       # (tuple with int) width, height
        target_size_lr  = (int(_w), int(_h))
        _diff_lr_w      = target_size_lr[0]  - _lr_w
        _diff_lr_h      = target_size_lr[-1] - _lr_h
        return_lr_image = True
    except:
        in_pil_lr       = None
        in_scale_factor = 1
        target_size_lr  = None
        return_lr_image = False
    
    
    if return_lr_image:
        if is_random:
            _margin_lr_left = int(random.uniform(0, _diff_lr_w))
            _margin_lr_up   = int(random.uniform(0, _diff_lr_h))
        else:
            _margin_lr_left = int(_diff_lr_w // 2)
            _margin_lr_up   = int(_diff_lr_h // 2)
        
        _margin_hr_left     = int(_margin_lr_left * in_scale_factor)
        _margin_hr_up       = int(_margin_lr_up * in_scale_factor)
        
        canvas_lr_image = Image.new(in_pil_lr.mode, target_size_lr, img_background)
        canvas_lr_image.paste(in_pil_lr, (_margin_lr_left, _margin_lr_up))
        
    else:
        if is_random:
            _margin_hr_left = int(random.uniform(0, _diff_hr_w))
            _margin_hr_up   = int(random.uniform(0, _diff_hr_h))
        else:
            _margin_hr_left = int(_diff_hr_w // 2)
            _margin_hr_up   = int(_diff_hr_h // 2)
    
    canvas_hr_image = Image.new(in_pil_hr.mode, target_size_hr, img_background)
    canvas_hr_image.paste(in_pil_hr, (_margin_hr_left, _margin_hr_up))
    
    
    if return_hr_label:
        # 현재 gray 라벨 이미지만 사용 가능
        canvas_hr_label = Image.new(in_pil_hr_label.mode, target_size_hr, lab_background)
        canvas_hr_label.paste(in_pil_hr_label, (_margin_hr_left, _margin_hr_up))
    
    
    if return_hr_label and return_lr_image:
        return canvas_hr_image, canvas_hr_label, canvas_lr_image
    elif return_hr_label:
        return canvas_hr_image, canvas_hr_label
    elif return_lr_image:
        return canvas_hr_image, canvas_lr_image
    
#=== End of pil_marginer_v1


def pil_marginer_v2(**kargs):
    # pil 이미지에 scaler 적용 후 margin 추가 혹은 crop 시행
    '''
    pil_img_hr, pil_lab_hr, pil_img_lr = pil_marginer_v2(in_pil_hr          =
                                                        ,target_size_hr     =
                                                        ,img_background     =
                                                        # (선택) 세부옵션 (각각 default 값 있음)
                                                        ,scaler             =
                                                        ,is_random          =
                                                        ,itp_opt_img    = Image.LANCZOS
                                                        ,itp_opt_lab    = Image.NEAREST
                                                         # 선택 (HR Label 관련)
                                                        ,in_pil_hr_label    = 
                                                        ,lab_background     =
                                                         # 선택 (LR Image 관련)
                                                        ,in_pil_lr          =
                                                        ,in_scale_factor    = 
                                                        ,target_size_lr     =
                                                        )
    '''
    try:
        scaler              = kargs['scaler']                                           # (float) 입력 이미지 배율
    except:
        scaler              = 1.0
    
    try:
        is_random           = kargs['is_random']                                        # (bool) margin 랜덤여부 결정
    except:
        is_random           = False
    
    try:
        itp_opt_img         = kargs['itp_opt_img']                                      # (Image.-) PIL Image Interpolate Option for 이미지
    except:
        itp_opt_img         = Image.LANCZOS
    
    try:
        itp_opt_lab         = kargs['itp_opt_lab']                                      # (Image.-) PIL Image Interpolate Option for 라벨
    except:
        itp_opt_lab         = Image.NEAREST
    
    
    # HR Image
    in_pil_hr               = kargs['in_pil_hr']                                        # (pil) HR Image
    ori_hr_w, ori_hr_h      = in_pil_hr.size                                            # (int int) original input size (W, H)
    
    itp_hr_w, itp_hr_h      = int(ori_hr_w*scaler), int(ori_hr_h*scaler)                # (int int) interpolated input size (W, H)
    
    if scaler != 1:
        itp_size_hr         = (itp_hr_w, itp_hr_h)
        itp_pil_hr          = in_pil_hr.resize(itp_size_hr, itp_opt_img)                # (pil) scaler 따라서 크기 변경된 HR Image
    else:
        itp_pil_hr          = in_pil_hr
    
    _w, _h                  = kargs['target_size_hr']                                   # (tuple with int) HR output size (W,H)
    tgt_hr_w, tgt_hr_h      = int(_w), int(_h)
    
    canvas_size_hr          = (                                                         # (tuple with int) HR 캔버스 크기
                               max(itp_hr_w, tgt_hr_w)
                              ,max(itp_hr_h, tgt_hr_h)
                              )
    
    
    _r, _g, _b              = kargs['img_background']                                   # (tuple with int) (R, G, B) Range: 0 ~ 255
    img_back                = (int(min(255, max(0, _r)))
                              ,int(min(255, max(0, _g)))
                              ,int(min(255, max(0, _b)))
                              )
    
    canvas_hr_img           = Image.new(in_pil_hr.mode, canvas_size_hr, img_back)       # (pil) empty canvas for HR Image
    
    # HR Label
    try:
        in_pil_hr_lab       = kargs['in_pil_hr_label']                                  # (pil) HR Label
        if scaler != 1:
            itp_size_hr     = (itp_hr_w, itp_hr_h)
            itp_pil_hr_lab  = in_pil_hr_lab.resize(itp_size_hr, itp_opt_lab)            # (pil) scaler 따라서 크기 변경된 HR Label
        else:
            itp_pil_hr_lab  = in_pil_hr_lab
        
        _n                  = kargs['lab_background']                                   # (tuple with int) (n)
        lab_back            = (int(min(255, max(0, _n))))
        
        canvas_hr_lab       = Image.new(in_pil_hr_lab.mode, canvas_size_hr, lab_back)   # (pil) empty canvas for HR Image
        
        return_hr_label     = True
    except:
        in_pil_hr_lab       = None
        lab_back            = None
        return_hr_label     = False
    
    
    # LR Image
    try:
        in_pil_lr           = kargs['in_pil_lr']                                        # (pil) LR Image
        ori_lr_w, ori_lr_h  = in_pil_lr.size                                            # (int int) original input size (W, H)
        in_scale_factor     = int(kargs['in_scale_factor'])                             # (int) scale factor between HR and LR
        
        itp_lr_w, itp_lr_h  = int(ori_lr_w*scaler), int(ori_lr_h*scaler)                # (int int) interpolated input size (W, H)
        
        if scaler != 1:
            itp_pil_lr      = in_pil_lr.resize((itp_lr_w, itp_lr_h), itp_opt_img)       # (pil)scaler 따라서 크기 변경된 LR Image
        else:
            itp_pil_lr      = in_pil_lr
        
        _w, _h              = kargs['target_size_lr']                                   # (tuple with int) LR output size (W,H)
        tgt_lr_w, tgt_lr_h  = int(_w), int(_h)                                          # -> HR 이미지와 in_scale_factor 배율이 맞도록 설정
        
        canvas_size_lr      = (                                                         # (tuple with int) LR 캔버스 크기
                               max(itp_lr_w, tgt_lr_w)
                              ,max(itp_lr_h, tgt_lr_h)
                              )
        
        canvas_lr_img       = Image.new(in_pil_lr.mode, canvas_size_lr, img_back)       # (pil) empty canvas for LR Image
        
        return_lr_image     = True
    except:
        in_pil_lr           = None
        in_scale_factor     = 1
        return_lr_image     = False
    
    
    # 캔버스에 덧칠하기
    if return_lr_image:
        diff_lr_w           = int(canvas_size_lr[0] - itp_lr_w)
        diff_lr_h           = int(canvas_size_lr[1] - itp_lr_h)
        
        if is_random:
            mrg_lr_left     = int(random.uniform(0, diff_lr_w))
            mrg_lr_up       = int(random.uniform(0, diff_lr_h))
        else:
            mrg_lr_left     = int(diff_lr_w // 2)
            mrg_lr_up       = int(diff_lr_h // 2)
        
        mrg_hr_left         = int(in_scale_factor*mrg_lr_left)
        mrg_hr_up           = int(in_scale_factor*mrg_lr_up)
        
        canvas_lr_img.paste(itp_pil_lr, (mrg_lr_left, mrg_lr_up))
        canvas_hr_img.paste(itp_pil_hr, (mrg_hr_left, mrg_hr_up))
        
        if return_hr_label:
            canvas_hr_lab.paste(itp_pil_hr_lab, (mrg_hr_left, mrg_hr_up))
        
    else:
        diff_hr_w           = int(canvas_size_hr[0] - itp_hr_w)
        diff_hr_h           = int(canvas_size_hr[1] - itp_hr_h)
        
        if is_random:
            mrg_hr_left     = int(random.uniform(0, diff_hr_w))
            mrg_hr_up       = int(random.uniform(0, diff_hr_h))
        else:
            mrg_hr_left     = int(diff_hr_w // 2)
            mrg_hr_up       = int(diff_hr_h // 2)
        
        canvas_hr_img.paste(itp_pil_hr, (mrg_hr_left, mrg_hr_up))
        
        if return_hr_label:
            canvas_hr_lab.paste(itp_pil_hr_lab, (mrg_hr_left, mrg_hr_up))
    
    # 캔버스 자르기
    if return_lr_image:
        if canvas_size_lr[0] > tgt_lr_w:
            # width 잘라야 됨
            if is_random:
                coor_lr_left= int(random.uniform(0, int(canvas_size_lr[0] - tgt_lr_w)))
            else:
                coor_lr_left= int((canvas_size_lr[0] - tgt_lr_w) // 2)
            coor_lr_right   = coor_lr_left + tgt_lr_w
            
            coor_hr_left    = int(coor_lr_left * in_scale_factor)
            coor_hr_right   = coor_hr_left + tgt_hr_w
            
        else:
            # width 자를 필요 없음
            coor_lr_left    = 0
            coor_lr_right   = tgt_lr_w
            
            coor_hr_left    = 0
            coor_hr_right   = tgt_hr_w
            
        
        if canvas_size_lr[1] > tgt_lr_h:
            # height 잘라야 됨
            if is_random:
                coor_lr_up  = int(random.uniform(0, int(canvas_size_lr[1] - tgt_lr_h)))
            else:
                coor_lr_up  = int((canvas_size_lr[1] - tgt_lr_h) // 2)
            coor_lr_down    = coor_lr_up + tgt_lr_h
            
            coor_hr_up      = int(coor_lr_up * in_scale_factor)
            coor_hr_down    = coor_hr_up + tgt_hr_h
            
        else:
            # height 자를 필요 없음
            coor_lr_up      = 0
            coor_lr_down    = tgt_lr_h
            
            coor_hr_up      = 0
            coor_hr_down    = tgt_hr_h
        
        out_pil_lr          = canvas_lr_img.crop((coor_lr_left, coor_lr_up, coor_lr_right, coor_lr_down))
        out_pil_hr          = canvas_hr_img.crop((coor_hr_left, coor_hr_up, coor_hr_right, coor_hr_down))
        
        if return_hr_label:
            out_pil_hr_lab  = canvas_hr_lab.crop((coor_hr_left, coor_hr_up, coor_hr_right, coor_hr_down))
            return out_pil_hr, out_pil_hr_lab, out_pil_lr
        else:
            return out_pil_hr, out_pil_lr
        
    else:
        if canvas_size_hr[0] > tgt_hr_w:
            # width 잘라야 됨
            if is_random:
                coor_hr_left= int(random.uniform(0, int(canvas_size_hr[0] - tgt_hr_w)))
            else:
                coor_hr_left= int((canvas_size_hr[0] - tgt_hr_w) // 2)
            coor_hr_right   = coor_hr_left + tgt_hr_w
            
        else:
            # width 자를 필요 없음
            coor_hr_left    = 0
            coor_hr_right   = tgt_hr_w
        
        if canvas_size_hr[1] > tgt_hr_h:
            # height 잘라야 됨
            if is_random:
                coor_hr_up  = int(random.uniform(0, int(canvas_size_hr[1] - tgt_hr_h)))
            else:
                coor_hr_up  = int((canvas_size_hr[1] - tgt_hr_h) // 2)
            coor_hr_down    = coor_hr_up + tgt_hr_h
            
        else:
            # height 자를 필요 없음
            coor_hr_up      = 0
            coor_hr_down    = tgt_hr_h
        
        out_pil_hr          = canvas_hr_img.crop((coor_hr_left, coor_hr_up, coor_hr_right, coor_hr_down))
        
        if return_hr_label:
            out_pil_hr_lab  = canvas_hr_lab.crop((coor_hr_left, coor_hr_up, coor_hr_right, coor_hr_down))
            return out_pil_hr, out_pil_hr_lab
        else:
            return out_pil_hr
    
    
#=== End of pil_marginer_v2


def pil_marginer_v3(**kargs):
    # pil 이미지에 scaler 적용 후 margin 추가 혹은 crop 시행
    # 라벨 정보 (전체 class 수, void class 번호) 요구됨
    '''
    pil_img_hr, pil_lab_hr, pil_img_lr = pil_marginer_v3(in_pil_hr          =
                                                        ,target_size_hr     =
                                                        ,img_background     =
                                                        # (선택) 세부옵션 (각각 default 값 있음)
                                                        ,scaler             = 1.0
                                                        ,is_random          = False
                                                        ,itp_opt_img        = Image.LANCZOS
                                                        ,itp_opt_lab        = Image.NEAREST
                                                         # 선택 (HR Label 관련)
                                                        ,in_pil_hr_label    =
                                                        ,lab_total          =
                                                        ,lab_background     =
                                                        
                                                        ,is_lab_verify         =
                                                        # 선택 - 선택 (Label 검증 관련, is_lab_verify=True에만 사용)
                                                        ,lab_try_ceiling    = 10
                                                        ,lab_class_min      =
                                                        ,lab_ratio_max      =
                                                         # 선택 (LR Image 관련)
                                                        ,in_pil_lr          =
                                                        ,in_scale_factor    = 
                                                        ,target_size_lr     =
                                                        )
    '''
    
    name_func = "(pil_marginer_v3) -> "
    
    try:
        scaler              = kargs['scaler']                                           # (float) 입력 이미지 배율
    except:
        scaler              = 1.0
    
    try:
        is_random           = kargs['is_random']                                        # (bool) margin 랜덤여부 결정
    except:
        is_random           = False
    
    try:
        itp_opt_img         = kargs['itp_opt_img']                                      # (Image.-) PIL Image Interpolate Option for 이미지
    except:
        itp_opt_img         = Image.LANCZOS
    
    try:
        itp_opt_lab         = kargs['itp_opt_lab']                                      # (Image.-) PIL Image Interpolate Option for 라벨
    except:
        itp_opt_lab         = Image.NEAREST
    
    
    # HR Image
    in_pil_hr               = kargs['in_pil_hr']                                        # (pil) HR Image
    ori_hr_w, ori_hr_h      = in_pil_hr.size                                            # (int int) original input size (W, H)
    
    itp_hr_w, itp_hr_h      = int(ori_hr_w*scaler), int(ori_hr_h*scaler)                # (int int) interpolated input size (W, H)
    
    if scaler != 1:
        itp_size_hr         = (itp_hr_w, itp_hr_h)
        itp_pil_hr          = in_pil_hr.resize(itp_size_hr, itp_opt_img)                # (pil) scaler 따라서 크기 변경된 HR Image
    else:
        itp_pil_hr          = in_pil_hr
    
    _w, _h                  = kargs['target_size_hr']                                   # (tuple with int) HR output size (W,H)
    tgt_hr_w, tgt_hr_h      = int(_w), int(_h)
    
    canvas_size_hr          = (max(itp_hr_w, tgt_hr_w), max(itp_hr_h, tgt_hr_h))        # (tuple with int) HR 캔버스 크기
    
    _r, _g, _b              = kargs['img_background']                                   # (tuple with int) (R, G, B) Range: 0 ~ 255
    img_back                = (int(min(255, max(0, _r))), int(min(255, max(0, _g))), int(min(255, max(0, _b))))
    
    canvas_hr_img           = Image.new(in_pil_hr.mode, canvas_size_hr, img_back)       # (pil) empty canvas for HR Image
    
    # HR Label
    try:
        in_pil_hr_lab       = kargs['in_pil_hr_label']                                  # (pil) HR Label
        if scaler != 1:
            itp_size_hr     = (itp_hr_w, itp_hr_h)
            itp_pil_hr_lab  = in_pil_hr_lab.resize(itp_size_hr, itp_opt_lab)            # (pil) scaler 따라서 크기 변경된 HR Label
        else:
            itp_pil_hr_lab  = in_pil_hr_lab
        
        lab_total           = kargs['lab_total']                                        # (int) 전체 class 수
        lab_back            = int(min(255, max(0, kargs['lab_background'])))            # (int) void class 번호
        
        if kargs['lab_background'] != lab_back:
            _str = name_func + "void class 번호는 0 ~ 255 int 값만 지원합니다."
            sys.exit(_str)
        else:
            pass
            #_str = name_func + "입력된 void class 번호는 " + str(lab_back) + " 입니다."
            #warnings.warn(_str)
        
        is_lab_verify       = kargs['is_lab_verify']                                    # (bool) 라벨 검증 시행여부
        
        if is_lab_verify and is_random:
            # crop 영역에 대해 입력값 기준 (적은 수의 class만 존재 / class 비율 중 최대값이 초과되게) crop 시 Re-Crop
            # Val 의 경우 (random하지 않은 경우)엔 적용 안됨
            try:
                lab_try_ceiling = kargs['lab_try_ceiling']                              # (int) Crop Retry 최대 횟수
            except:
                lab_try_ceiling = 10
            
            lab_class_min   = kargs['lab_class_min']                                    # (int) void 포함 최소 class 종류
            lab_ratio_max   = kargs['lab_ratio_max']                                    # (float) class 비율값의 상한값 (0 ~ 1.0)
            #_str = name_func + "Label 검증 시행됨 (void 포함 최소 class 종류: " + str(lab_class_min) + ", class 비율값 상한: " + str(lab_ratio_max) + ")"
            #warnings.warn(_str)
        else:
            # random 하지 않으면 label 검증 시행 안함
            is_lab_verify = False
            #_str = name_func + "Label 검증 시행 안함"
            #warnings.warn(_str)
        
        canvas_hr_lab       = Image.new(in_pil_hr_lab.mode, canvas_size_hr, (lab_back)) # (pil) empty canvas for HR Label
        
        return_hr_label     = True
        
    except:
        in_pil_hr_lab       = None
        lab_back            = None
        lab_total           = None
        is_lab_verify       = False
        return_hr_label     = False
        
    
    # LR Image
    try:
        in_pil_lr           = kargs['in_pil_lr']                                        # (pil) LR Image
        ori_lr_w, ori_lr_h  = in_pil_lr.size                                            # (int int) original input size (W, H)
        in_scale_factor     = int(kargs['in_scale_factor'])                             # (int) scale factor between HR and LR
        
        itp_lr_w, itp_lr_h  = int(ori_lr_w*scaler), int(ori_lr_h*scaler)                # (int int) interpolated input size (W, H)
        
        if scaler != 1:
            itp_pil_lr      = in_pil_lr.resize((itp_lr_w, itp_lr_h), itp_opt_img)       # (pil)scaler 따라서 크기 변경된 LR Image
        else:
            itp_pil_lr      = in_pil_lr
        
        _w, _h              = kargs['target_size_lr']                                   # (tuple with int) LR output size (W,H)
        tgt_lr_w, tgt_lr_h  = int(_w), int(_h)                                          # -> HR 이미지와 in_scale_factor 배율이 맞도록 설정
        
        canvas_size_lr      = (                                                         # (tuple with int) LR 캔버스 크기
                               max(itp_lr_w, tgt_lr_w)
                              ,max(itp_lr_h, tgt_lr_h)
                              )
        
        canvas_lr_img       = Image.new(in_pil_lr.mode, canvas_size_lr, img_back)       # (pil) empty canvas for LR Image
        
        return_lr_image     = True
    except:
        in_pil_lr           = None
        in_scale_factor     = 1
        return_lr_image     = False
    
    
    # 캔버스에 덧칠하기
    if return_lr_image:
        diff_lr_w           = int(canvas_size_lr[0] - itp_lr_w)
        diff_lr_h           = int(canvas_size_lr[1] - itp_lr_h)
        
        if is_random:
            mrg_lr_left     = int(random.uniform(0, diff_lr_w))
            mrg_lr_up       = int(random.uniform(0, diff_lr_h))
        else:
            mrg_lr_left     = int(diff_lr_w // 2)
            mrg_lr_up       = int(diff_lr_h // 2)
        
        mrg_hr_left         = int(in_scale_factor*mrg_lr_left)
        mrg_hr_up           = int(in_scale_factor*mrg_lr_up)
        
        canvas_lr_img.paste(itp_pil_lr, (mrg_lr_left, mrg_lr_up))
        canvas_hr_img.paste(itp_pil_hr, (mrg_hr_left, mrg_hr_up))
        
        if return_hr_label:
            canvas_hr_lab.paste(itp_pil_hr_lab, (mrg_hr_left, mrg_hr_up))
        
    else:
        diff_hr_w           = int(canvas_size_hr[0] - itp_hr_w)
        diff_hr_h           = int(canvas_size_hr[1] - itp_hr_h)
        
        if is_random:
            mrg_hr_left     = int(random.uniform(0, diff_hr_w))
            mrg_hr_up       = int(random.uniform(0, diff_hr_h))
        else:
            mrg_hr_left     = int(diff_hr_w // 2)
            mrg_hr_up       = int(diff_hr_h // 2)
        
        canvas_hr_img.paste(itp_pil_hr, (mrg_hr_left, mrg_hr_up))
        
        if return_hr_label:
            canvas_hr_lab.paste(itp_pil_hr_lab, (mrg_hr_left, mrg_hr_up))
    
    
    _need_cut       = False     # 자르기 작업이 시행될 것인가?
    _label_verified = False     # 잘라진 라벨이 주어진 조건을 충족하였는가?
    _retry_counter  = 0
    
    while(True):
        _retry_counter += 1
        # 좌표 계산 
        if return_lr_image:
            # (HR, LR)
            if canvas_size_lr[0] > tgt_lr_w:
                # width 잘라야 됨
                _need_cut = True
                if is_random:
                    coor_lr_left= int(random.uniform(0, int(canvas_size_lr[0] - tgt_lr_w)))
                else:
                    coor_lr_left= int((canvas_size_lr[0] - tgt_lr_w) // 2)
                coor_lr_right   = coor_lr_left + tgt_lr_w
                
                coor_hr_left    = int(coor_lr_left * in_scale_factor)
                coor_hr_right   = coor_hr_left + tgt_hr_w
                
            else:
                # width 자를 필요 없음
                coor_lr_left    = 0
                coor_lr_right   = tgt_lr_w
                
                coor_hr_left    = 0
                coor_hr_right   = tgt_hr_w
                
            
            if canvas_size_lr[1] > tgt_lr_h:
                # height 잘라야 됨
                _need_cut = True
                if is_random:
                    coor_lr_up  = int(random.uniform(0, int(canvas_size_lr[1] - tgt_lr_h)))
                else:
                    coor_lr_up  = int((canvas_size_lr[1] - tgt_lr_h) // 2)
                coor_lr_down    = coor_lr_up + tgt_lr_h
                
                coor_hr_up      = int(coor_lr_up * in_scale_factor)
                coor_hr_down    = coor_hr_up + tgt_hr_h
                
            else:
                # height 자를 필요 없음
                coor_lr_up      = 0
                coor_lr_down    = tgt_lr_h
                
                coor_hr_up      = 0
                coor_hr_down    = tgt_hr_h
            
            
        else:
            # (HR)
            if canvas_size_hr[0] > tgt_hr_w:
                # width 잘라야 됨
                _need_cut = True
                if is_random:
                    coor_hr_left= int(random.uniform(0, int(canvas_size_hr[0] - tgt_hr_w)))
                else:
                    coor_hr_left= int((canvas_size_hr[0] - tgt_hr_w) // 2)
                coor_hr_right   = coor_hr_left + tgt_hr_w
                
            else:
                # width 자를 필요 없음
                coor_hr_left    = 0
                coor_hr_right   = tgt_hr_w
            
            if canvas_size_hr[1] > tgt_hr_h:
                # height 잘라야 됨
                _need_cut = True
                if is_random:
                    coor_hr_up  = int(random.uniform(0, int(canvas_size_hr[1] - tgt_hr_h)))
                else:
                    coor_hr_up  = int((canvas_size_hr[1] - tgt_hr_h) // 2)
                coor_hr_down    = coor_hr_up + tgt_hr_h
                
            else:
                # height 자를 필요 없음
                coor_hr_up      = 0
                coor_hr_down    = tgt_hr_h
            
        
        
        if is_lab_verify:
            # 검증용 crop 시행
            sample_label = canvas_hr_lab.crop((coor_hr_left, coor_hr_up, coor_hr_right, coor_hr_down))  # sample pil label
            sample_np = np.array(sample_label)                                                          # pil -> numpy
            sample_bin = np.bincount(sample_np.reshape(-1), minlength=lab_total)                        # class 별 pixel 카운트
            sample_total_labels = int(np.sum(sample_bin > 0))                                           # 유효 class 수 (void 포함)
            sample_ratio_max = float(np.max(sample_bin) / np.sum(sample_bin))                           # class 최대 비율 (0 ~ 1.0 사이값)
            
            if (sample_total_labels >= lab_class_min) and (sample_ratio_max <= lab_ratio_max):
                # 라벨 검증을 통과한 경우
                _label_verified = True
        else:
            # 라벨 검증을 시행하지 않는 경우
            _label_verified = True
        
        if not _need_cut:
            # 잘라내지 않았다면 무조건 pass
            _label_verified = True
        
        if _label_verified or (_retry_counter > lab_try_ceiling):
            break
        #else:
        #    print(name_func + "Retry...")
        
        #---End of While
    
    # 최종 crop 시행
    if return_lr_image:
        # (HR, LR)
        out_pil_lr          = canvas_lr_img.crop((coor_lr_left, coor_lr_up, coor_lr_right, coor_lr_down))
        out_pil_hr          = canvas_hr_img.crop((coor_hr_left, coor_hr_up, coor_hr_right, coor_hr_down))
        
        if return_hr_label:
            out_pil_hr_lab  = canvas_hr_lab.crop((coor_hr_left, coor_hr_up, coor_hr_right, coor_hr_down))
            return out_pil_hr, out_pil_hr_lab, out_pil_lr
        else:
            return out_pil_hr, out_pil_lr
    else:
        # (HR)
        out_pil_hr          = canvas_hr_img.crop((coor_hr_left, coor_hr_up, coor_hr_right, coor_hr_down))
        
        if return_hr_label:
            out_pil_hr_lab  = canvas_hr_lab.crop((coor_hr_left, coor_hr_up, coor_hr_right, coor_hr_down))
            return out_pil_hr, out_pil_hr_lab
        else:
            return out_pil_hr
    
    
    
#=== End of pil_marginer_v3


def list_pils_2_tensor_v1(**kargs): #@@@ 검증필요
    name_func = "[list_pils_2_tensor_v1] ->"
    '''
    tensor_out = list_pils_2_tensor_v1(#(list - pil) list로 묶인 PIL 이미지 묶음을 pytorch tensor로 변환해줌
                                       #label should be GRAY
                                       list_pils = 
                                      
                                       #(int) pil 이미지 채널 수 (Color = 3, Gray = 1)
                                      ,pil_channels = 
                                      
                                       #(bool) 라벨 여부 (one-hot 변환여부 결정) (default: False)
                                      ,is_label = 
                                       #(귀속) (int) 전체 라벨 수 (void 포함)
                                      ,label_total = HP_LABEL_TOTAL
                                       #(귀속) (int) void 라벨 번호
                                      ,label_void = HP_LABEL_VOID
                                       #(귀속) (bool) 라벨 스무딩 적용여부
                                      ,is_label_dilated = True
                                      
                                       #(bool) pil 이미지 크기 변환 시행여부 (default: False)
                                      ,is_resized = 
                                       #(귀속) (tuple) pil 변환결과 크기 (w, h)
                                      ,resized_size = 
                                       #(귀속) (str) interpolation 방식 ("NEAREST", "BILINEAR", "BICUBIC", "LANCZOS")
                                      ,resized_method = 
                                      
                                       #(transforms) pil to tensor 함수 (label은 자동으로 ToTensor 고정)
                                      ,transforms_to_tensor = 
                                       #(bool) requires grad 여부
                                      ,is_requires_grad = 
                                      )
    '''
    
    list_pils = kargs['list_pils']
    pil_channels = int(kargs['pil_channels'])
    
    try:
        is_label = kargs['is_label']
    except:
        is_label = False
    if is_label == True:
        #현재 GRAY 라벨만 지원됨
        if not pil_channels == 1:
            print(name_func, "현재 GRAY 라벨만 지원됩니다.")
        pil_channels = 1
        
        label_total = kargs['label_total']
        label_void = kargs['label_void']
        is_label_dilated = kargs['is_label_dilated']
        if is_label_dilated:
            cv_kernel_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    
    try:
        is_resized = kargs['is_resized']
    except:
        is_resized = False
    if is_resized:
        resized_size = kargs['resized_size']
        resized_method = kargs['resized_method']
    
    transforms_to_tensor = kargs['transforms_to_tensor']
    if is_label == True:
        transforms_to_tensor = transforms.Compose([#PIL 이미지 or npArray -> pytorch 텐서
                                                  transforms.ToTensor()
                                                  ])
    
    is_requires_grad = kargs['is_requires_grad']
    
    for i_image in range(len(list_pils)):
        tmp_pil = list_pils[i_image]
        tmp_w, tmp_h = tmp_pil.size
        if not is_resized:
            in_pil = tmp_pil
            in_w = tmp_w
            in_h = tmp_h
        else: #resized
            in_w = resized_size[0]
            in_h = resized_size[-1]
            if tmp_w == in_w and tmp_h == in_h: #size same
                print(name_func, "resize process skip: target is same size", in_w, in_h)
                in_pil = tmp_pil
            else: #size different: input <-> target
                if resized_method == "NEAREST":
                    in_pil = tmp_pil.resize((int(in_w), int(in_h)), Image.NEAREST)
                elif resized_method == "BILINEAR":
                    in_pil = tmp_pil.resize((int(in_w), int(in_h)), Image.BILINEAR)
                elif resized_method == "BICUBIC":
                    in_pil = tmp_pil.resize((int(in_w), int(in_h)), Image.BICUBIC)
                elif resized_method == "LANCZOS":
                    in_pil = tmp_pil.resize((int(in_w), int(in_h)), Image.LANCZOS)
        
        
        if is_label:
            #pil -> nparray
            in_np = np.array(list_pils[i_image])
            
            #단일라벨텐서 생성
            flag_init_label = 0
            for i_label in range(label_total):
                if i_label == label_void:
                    continue
                #단일 클래스 라벨만 추출
                np_label_single = np.where(in_np == i_label, 1, 0).astype(np.uint8)
                
                #np -> pil (+ 선택적 라벨 스무딩)
                if is_label_dilated:
                    pil_onehot = Image.fromarray(cv2.dilate(np_label_single, cv_kernel_dilation))
                elif not is_label_dilated:
                    pil_onehot = Image.fromarray(np_label_single)
                
                #to tensor (single)
                if flag_init_label == 0:
                    flag_init_label = 1
                    out_tensor_single = transforms_to_tensor(pil_onehot)
                else:
                    out_tensor_single = torch.cat([out_tensor_single, transforms_to_tensor(pil_onehot)], dim = 0)
            
            #to tensor (total)
            if i_image == 0:
                #[c, h, w] 크기 torch.tensor 생성 (c = label_total - label_void)
                in_tensor = out_tensor_single
            else: #2번쨰 라벨 이미지 이후 ~
                #[c*b, h, w] 크기 torch.tensor 생성 (c = label_total - label_void, b = 현재 이미지 번호)
                in_tensor = torch.cat([in_tensor, out_tensor_single], dim = 0)
        
        else: #not label
            #to tensor (total)
            if i_image == 0:
                #[c, h, w] 크기 torch.tensor 생성 (c = pil_channels)
                in_tensor = transforms_to_tensor(in_pil)
            else: #2번쨰 이미지 이후 ~
                #["이미지 수", c, h, w] 크기 torch.tensor로 갱신 (c = pil_channels)
                in_tensor = torch.cat([in_tensor, transforms_to_tensor(in_pil)], dim = 0)
        
    in_bc, in_h, in_w = in_tensor.shape
    
    if is_label == True:
        tmp_channels = label_total - 1
        
    else: #not label
        tmp_channels = pil_channels
    
    #최종 텐서 생성
    out_tensor = torch.reshape(in_tensor, (in_bc // tmp_channels, tmp_channels, in_h, in_w))
    
    if is_requires_grad:
        return out_tensor.float().requires_grad_(True)
    else:
        return out_tensor.float()

#=== End of list_pils_2_tensor_v1 #@@@ 검증필요

def tensor_2_list_pils_v1(**kargs): #@@@ 검증필요
    name_func = "[tensor_2_list_pils_v1] ->"
    '''
    tensor_2_list_pils_v1(#텐서 -> pil 이미지 리스트
                         #(tensor) 변환할 텐서, 모델에서 다중 결과물이 생성되는 경우, 단일 출력물 묶음만 지정해서 입력 
                         #(예: MPRNet -> in_tensor = tensor_sr_hypo[0])
                         in_tensor = 
                         
                          #(bool) 라벨 여부 (출력 pil 이미지 = 3ch, 라벨 = 1ch 고정) (default: False)
                         ,is_label = 
                        
                          #(bool) pil 이미지 크기 변환 시행여부 (default: False)
                         ,is_resized = 
                          #(귀속) (tuple) pil 변환결과 크기 (w, h)
                         ,resized_size = 
                          #(귀속) (str) interpolation 방식 ("NEAREST", "BILINEAR", "BICUBIC", "LANCZOS")
                         ,resized_method = 
                         )
    '''
    out_list_pils = []
    
    in_tensor = kargs['in_tensor'].clone().detach()
    
    try:
        is_label = kargs['is_label']
    except:
        is_label = False
    
    try:
        is_resized = kargs['is_resized']
    except:
        is_resized = False
    if is_resized:
        resized_size = kargs['resized_size']
        resized_method = kargs['resized_method']
    
    in_tensor_size = in_tensor.shape
    if len(in_tensor_size) == 3:
        #라벨 입력에 해당
        in_c = 1
        in_b, in_h, in_w = in_tensor.shape
    elif len(in_tensor_size) == 4:
        in_b, in_c, in_h, in_w = in_tensor.shape
    
    for i_b in range(in_b):
        #tensor -> pil (single)
        if is_label:
            tmp_pil = Image.fromarray(in_tensor.cpu().numpy()[i_b].astype('uint8'))
        else: #not label
            tmp_pil = to_pil_image(torch.clamp(in_tensor[i_b], min=0, max=1))
        
        tmp_w, tmp_h = tmp_pil.size
        
        #pil -> pil (선택적 resize)
        if not is_resized:
            in_pil = tmp_pil
            in_w = tmp_w
            in_h = tmp_h
        else: #resized
            in_w = resized_size[0]
            in_h = resized_size[-1]
            if tmp_w == in_w and tmp_h == in_h: #size same
                print(name_func, "resize process skip: target is same size", in_w, in_h)
                in_pil = tmp_pil
            else: #size different: input <-> target
                if resized_method == "NEAREST":
                    in_pil = tmp_pil.resize((int(in_w), int(in_h)), Image.NEAREST)
                elif resized_method == "BILINEAR":
                    in_pil = tmp_pil.resize((int(in_w), int(in_h)), Image.BILINEAR)
                elif resized_method == "BICUBIC":
                    in_pil = tmp_pil.resize((int(in_w), int(in_h)), Image.BICUBIC)
                elif resized_method == "LANCZOS":
                    in_pil = tmp_pil.resize((int(in_w), int(in_h)), Image.LANCZOS)
        
        #list append
        out_list_pils.append(in_pil)
    
    return out_list_pils

#=== End of tensor_2_list_pils_v1 #@@@ 검증필요



def merge_patch_v1(**kargs):
    #patch 병합 후 단일 이미지 생성
    '''
    pil_merged = merge_patch_v1(in_list_patch = list_patch_predicted    #(list) patch 묶음
                                                                        # 또는
                                in_dict_patch = dict_patch_predicted    #(dict) patch 묶음
                                                                        # 중 한 가지 patch 묶음만 입력 (list 권장)
                               ,original_pil_size = (480, 360)          #(tuple) 원본 이미지 크기 (w, h)
                               ,patch_pil_size =    (256, 256)          #(tuple) patch 이미지 크게 (w, h)
                               )
    '''

    
    try:
        #(list) patch 묶음
        in_list_patch = kargs['in_list_patch']
    except:
        #(dict) patch 묶음
        in_dict_patch = kargs['in_dict_patch']
        in_list_patch = []
        for i_key in in_dict_patch:
            in_list_patch.append(in_dict_patch[i_key])
    
    #(tuple) 원본 이미지 크기 (w, h)
    ori_w, ori_h = kargs['original_pil_size']
    #(tuple) patch 이미지 크게 (w, h)
    pat_w, pat_h = kargs['patch_pil_size']
    
    #(str) 이미지 모드 확인
    in_pil_mode = in_list_patch[0].mode
    
    #모서리 여백 길이 확인
    margin_w = int(ori_w - (ori_w // pat_w) * pat_w)
    margin_h = int(ori_h - (ori_h // pat_h) * pat_h)
    
    #print("여백 길이 (w,h): (", margin_w, ", ", margin_h, ")")
    
    if margin_w == 0:
        amount_patch_x = int(ori_w // pat_w)
    elif margin_w > 0:
        amount_patch_x = int(ori_w // pat_w) + 1
    
    if margin_h == 0:
        amount_patch_y = int(ori_h // pat_h)
    elif margin_h > 0:
        amount_patch_y = int(ori_h // pat_h) + 1
    
    #print("이미지 총 수량 예측 (가로, 세로): (", amount_patch_x, "x", amount_patch_y, ")")
    
    #빈 이미지 생성 ((str)이미지 모드, (w,h)이미지 크기, (생략 가능)픽셀 컬러)
    out_pil = Image.new(in_pil_mode, (ori_w, ori_h))
    
    for i_h in range(amount_patch_y):
        for i_w in range(amount_patch_x):
            #현재 이미지
            #print("i_w + i_h*amount_patch_x:", i_w + i_h*amount_patch_x)
            tmp_pil_patch = in_list_patch[i_w + i_h*amount_patch_x]
            
            #이어붙일 좌표
            tuple_coor = (i_w*pat_w, i_h*pat_h)
            #자를 이미지의 초기 크기
            cut_w, cut_h = tmp_pil_patch.size
            #이미지가 잘릴 좌표
            cut_coor = [0,0,cut_w,cut_h]
            
            #우측에 자투리 이미지가 있는 경우에 맨 우측 이미지를 다루는 상황
            if i_w + 1 == amount_patch_x and margin_w != 0:
                cut_coor[0] = pat_w - margin_w

            #하단에 자투리 이미지가 있는 경우에 맨 하단 이미지를 다루는 상황
            if i_h + 1 == amount_patch_y and margin_h != 0:
                cut_coor[1] = pat_h - margin_h
            
            if cut_coor[0] != 0 or cut_coor[1] != 0:
                 out_pil.paste(tmp_pil_patch.crop((cut_coor[0], cut_coor[1], cut_coor[2], cut_coor[3])), tuple_coor)
            else:
                out_pil.paste(tmp_pil_patch, tuple_coor)
    
    return out_pil

#=== End of merge_patch_v1

#IN (* 짝수 개수):
#           홀: (str) key
#           짝: (str) value
#IN (**1):
#           in_dict  : (dict) 변수
#IN(** 생략가능)
#           in_dict_name : (str) 변수 이름
#           is_print: (bool) 결과 출력여부 (default = True)
#dict 형 요소 추가
def update_dict(*args, **kargs):
    name_func = "[update_dict] ->"
    #dict = {key : value}, 별도 함수 없이 append & 요소 수정 가능 / dict 변수 생성은 불가능 (사전선언 필수)
    in_dict = kargs['in_dict']

    try:
        dict_name = kargs['in_dict_name']
    except:
        dict_name = "False"
    
    try:
        is_print = kargs['is_print']
    except:
        is_print = True
    
    flag_elem = 0
    
    for elem in args:
        if type(elem) != type(" "):
            print(name_func, "입력값은 str 형태만 가능합니다.")
            sys.exit(1)
            
        if flag_elem == 0:
            in_key = elem
            if in_key == "" or in_key == " ":
                in_key = str(len(in_dict))
            flag_elem = 1
        else:
            in_value = elem
            in_dict[in_key] = in_value
            if dict_name == "False":
                if is_print:
                    print(name_func, "{", in_key, ":", in_value, "}")
            else:
                if is_print:
                    print(name_func, dict_name, "{", in_key, ":", in_value, "}")
            flag_elem = 0


#=== End of update_dict

'''
update_dict_v2("", "추가할 내용 2-1"
              ,"", "추가할 내용 2-2"
              ,in_dict_dict = {"a":dict_a, "b":dict_b, "c":dict_c}
              ,in_dict_key = "c"
              ,print_head = "sample"
              ,is_print = True
              )
'''
#IN (* 짝수 개수):
#           홀: (str) key
#           짝: (str) value
#IN (**1 or 2):
#           <case 1>
#           in_dict  : (dict) 내용을 추가할 변수
#           <case 2>
#           in_dict_dict : (dict) in_dict 후보로 구성된 dict
#           in_dict_key  : (str)  이번에 갱신할 dict의 key
#
#IN(** 생략가능)
#           in_print_head : (str) 출력문 말머리 (변경됨: in_dict_name -> in_print_head)
#           is_print: (bool) 결과 출력여부 (default = True)
#dict 형 요소 추가
def update_dict_v2(*args, **kargs):
    name_func = "[update_dict_v2] ->"
    #dict = {key : value}, 별도 함수 없이 append & 요소 수정 가능 / dict 변수 생성은 불가능 (사전선언 필수)
    
    try:
        in_dict = kargs['in_dict']
    except:
        #후보 dict의 dict
        in_dict_dict = kargs['in_dict_dict']
        #이번에 update 할 dict의 key (str)
        in_dict_key = kargs['in_dict_key']
        #dict 결정
        in_dict = in_dict_dict[in_dict_key]
        
    try:
        print_head = kargs['in_print_head']
    except:
        print_head = "False"
    
    try:
        is_print = kargs['is_print']
    except:
        is_print = True
    
    flag_elem = 0
    
    for elem in args:
        if type(elem) != type(" "):
            print(name_func, "입력값은 str 형태만 가능합니다.")
            sys.exit(1)
            
        if flag_elem == 0:
            in_key = elem
            if in_key == "" or in_key == " ":
                in_key = str(len(in_dict))
            flag_elem = 1
        else:
            in_value = elem
            in_dict[in_key] = in_value
            if print_head == "False":
                if is_print:
                    print(name_func, "{", in_key, ":", in_value, "}")
            else:
                if is_print:
                    print(name_func, print_head, "{", in_key, ":", in_value, "}")
            flag_elem = 0

#=== End of update_dict_v2





if __name__ == '__main__':
    print("EoF: data_tool.py")