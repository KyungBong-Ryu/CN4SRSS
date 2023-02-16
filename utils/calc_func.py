#calc_func.py
import numpy as np
from PIL import Image
import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import sys

#pytorch-ignite
#import ignite
#------------------------------------------------------------------------------------------------------
'''
#loss 연산
#IN (4): 
#       (ts) 정답 라벨(one-hot), 
#       (ts) 예측값(0~1 softmax), 
#       (int)void 라벨 번호, 
#       (np) 라벨별 가중치
#OUT(1):
#       (ts) 연산결과 loss 값
def calc_loss_v1(in_ts_y, in_ts_h, in_num_void, in_np_weight):
    #MSE(Mean Squared Error) 공식 활용
    #정답과 예측의 차이를 제곱하여 합한 다음, 평균내는 loss
    #전체 Train 데이터셋의 라벨별 비율의 역수를 가중치로 적용
    #void 라벨에 대한 오차는 반영하지 않음
    
    #print("(func) init calc_loss")
    in_b, in_c, in_h, in_w = in_ts_y.shape
    
    #print(in_b, in_c, in_h, in_w, in_ts_y.shape)
    
    flag_init_func = 0
    
    for i_bat in range(in_b):
        for i_lab in range(in_c):
            #void 라벨 제외
            if i_lab != in_num_void:
                #tmp = torch.sum(torch.exp(torch.sub(in_ts_y[i_bat][i_lab], in_ts_h[i_bat][i_lab])))
                tmp = torch.sum(torch.pow(torch.sub(in_ts_y[i_bat][i_lab], in_ts_h[i_bat][i_lab]), 2))
                if flag_init_func == 0:
                    flag_init_func += 1
                    #print("init")
                    out_ts_loss_sum = tmp / (in_h * in_w * in_np_weight[i_lab])
                else:
                    out_ts_loss_sum += tmp / (in_h * in_w * in_np_weight[i_lab])
                #print("weight", in_np_weight[i_lab])
    
    out_ts_loss = out_ts_loss_sum / (in_b * (in_c - 1))
    
    #print("out_ts_loss", out_ts_loss)
    
    return out_ts_loss
'''
#===

#IN (*2):[pil_origin: (pil) 원본 이미지]
#       [pil_compare:(pil) 비교대상 이미지]
def calc_psnr(**kargs):
    #pil -> opencv
    #https://www.zinnunkebi.com/python-opencv-pil-convert/
    try:
        in_cv_origin  = cv2.cvtColor(np.array(kargs['pil_origin']),  cv2.COLOR_RGB2BGR)
        im_cv_compare = cv2.cvtColor(np.array(kargs['pil_compare']), cv2.COLOR_RGB2BGR)
        
        #cv2.imshow("in_cv_origin", in_cv_origin)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #cv2.imshow("im_cv_compare", im_cv_compare)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        return cv2.PSNR(in_cv_origin, im_cv_compare)
        
    except:
        print("(except) image load FAIL")
        return -1

#=== End of calc_psnr

#psnr & ssim 계산 (높을수록 원본에 비슷함을 의미, 단위 = dB)
#IN (2):
#       pil_original = (pil) 원본 이미지
#       pil_contrast = (pil) 대조 이미지
#IN (*)
#       is_gray = (bool) 입력 이미지 gray 여부 (default = False)
#       is_return_ssim_img = (bool) SSIM Image return 여부 (default = False)
#OUT (2):
#       (float) PSNR 값
#       (float) SSIM 값
#OUT (*):
#       (ndarray) full ssim image

'''
out_psnr, out_ssim = calc_psnr_ssim(pil_original = 
                                   ,pil_contrast = 
                                   )

'''

def calc_psnr_ssim(**kargs):
    #from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    #PSNR: https://scikit-image.org/docs/dev/api/skimage.metrics.html?highlight=peak_signal_noise_ratio#peak-signal-noise-ratio
    #SSIM: https://scikit-image.org/docs/dev/api/skimage.metrics.html?highlight=structural_similarity#skimage.metrics.structural_similarity
    #(pil) 원본 이미지
    in_pil_original = kargs['pil_original']
    
    #(pil) 대조 이미지
    in_pil_contrast = kargs['pil_contrast']
    
    #(bool) 입력 이미지 gray 여부 (default = False)
    try:
        is_gray = kargs['is_gray']
    except:
        is_gray = False
        #multichannel = True로 설정
        #위 argument는 후에 channel_axis로 변경될 예정
    
    #(bool) SSIM Image return 여부 (default = False)
    try:
        is_return_ssim_img = kargs['is_return_ssim_img']
    except:
        is_return_ssim_img = False
    
    np_original = np.array(in_pil_original)
    np_contrast = np.array(in_pil_contrast)
    
    #print(np.max(np_original),np.max(np_contrast),np.min(np_original),np.min(np_contrast))
    
    mse = np.mean((np_original - np_contrast) **2)
    
    #psnr 연산
    if mse == 0:
        out_psnr = 100
    else:
        out_psnr = peak_signal_noise_ratio(np_original, np_contrast)
    
    #ssim 연산
    if is_gray:
        if is_return_ssim_img:
            out_ssim, out_ssim_img = structural_similarity(np_original, np_contrast, full = True)
        else:
            out_ssim = structural_similarity(np_original, np_contrast)
    else:
        if is_return_ssim_img:
            #out_ssim, out_ssim_img = structural_similarity(np_original, np_contrast, channel_axis = 2, full = True)
            out_ssim, out_ssim_img = structural_similarity(np_original, np_contrast, multichannel = True, full = True)
        else:
            #out_ssim = structural_similarity(np_original, np_contrast, channel_axis = 2)
            out_ssim = structural_similarity(np_original, np_contrast, multichannel = True)
    
    #(float) out_psnr, (float) out_ssim
    
    if is_return_ssim_img:
        return out_psnr, out_ssim, out_ssim_img
    else:
        return out_psnr, out_ssim

#=== End of calc_psnr_ssim


def calc_miou_gray(**kargs):
    # PIL 이미지 (gray) mIoU 계산
    # IN (**3):
    #       (pil) pil_gray_answer
    #           단일채널 정답 이미지
    #       (pil) pil_gray_predict
    #           단일채널 에측 이미지
    #       (int) int_total_labels
    #           전체 라벨 수 (void 포함)
    # IN (*):
    #       (int) int_void_label
    #           void 라벨값 (기본값 = -1)
    #
    # OUT (2):
    #       (float) mIoU
    #       (dict)  클래스 별 IoU
    #       
    #print("(func) calc_miou")
    #[INPUT]--------------------------------------------------------
    try:
        #(np) 정답 이미지
        in_np_ans = np.array(kargs['pil_gray_answer'])
        #(np) 예측 이미지
        in_np_pred = np.array(kargs['pil_gray_predict'])
        #(int) 전체 라벨 수
        in_int_total_labels = kargs['int_total_labels']
    except:
        print("(except) in calc_miou")
        print("more inputs required")
        sys.exit(9)
    
    try:
        #(int) void 라벨 번호 (-1 : void 없음)
        in_int_void = kargs['int_void_label']
    except:
        in_int_void = -1
    
    #[CALC]---------------------------------------------------------
    #전체 클래스 수 
    int_total_labels = in_int_total_labels
    
    #각 클래스별 빈도 수 (1d)
    np_count_ans = np.bincount(in_np_ans.reshape(-1), minlength = int_total_labels)
    np_count_pred = np.bincount(in_np_pred.reshape(-1), minlength = int_total_labels)
    
    #이미지를 1차원 벡터로
    np_1d_ans = in_np_ans.reshape(-1)
    np_1d_pred = in_np_pred.reshape(-1)
    
    #(ans,pred) 조합을 번호로 변경하여 1차원 배열로 저장 (label 수를 기준으로 n진법 형태로 변환)
    np_1d_category = int_total_labels * np_1d_ans + np_1d_pred
    
    #confusion matrix 생성 (2d)
    np_1d_confusion_matrix = np.bincount(np_1d_category, minlength = int_total_labels*int_total_labels)
    np_2d_confusion_matrix_raw = np_1d_confusion_matrix.reshape(int_total_labels, int_total_labels)
    
    
    if(in_int_void == 0) :
        np_2d_confusion_matrix = np.append(np.zeros((1,in_int_total_labels))
                                          ,np.delete(np_2d_confusion_matrix_raw, in_int_void, axis = 0)
                                          ,axis = 0
                                          )
    elif (in_int_void == in_int_total_labels - 1):
        np_2d_confusion_matrix = np.append(np.delete(np_2d_confusion_matrix_raw, in_int_void, axis = 0)
                                          ,np.zeros((1,in_int_total_labels))
                                          ,axis = 0
                                          )
    else:
        #void 라벨 번호 입력 오류
        sys.exit(9)
    
    #클래스 별 IoU 계산
    np_1d_intersection = np.diag(np_2d_confusion_matrix)
    
    np_1d_union = (np_2d_confusion_matrix.sum(axis = 0) 
                   + np_2d_confusion_matrix.sum(axis = 1) 
                   - np_1d_intersection)
    
    #연산에 사용한 라벨의 수
    int_available_labels = 0
    #유효한 iou 총 합
    float_sum_iou = 0
    #유효한 iou 딕셔너리
    #dict_iou = {"label":"iou"}
    dict_iou = {}
    
    for i_label in range(int_total_labels):
        #void 라벨이 아닌 경우
        if(i_label != in_int_void):
            #라벨의 픽셀 수가 0이 아닌 경우
            if(np_count_ans[i_label] != 0):
                int_available_labels += 1
                #print("IoU", i_label, np_1d_intersection[i_label] / np_1d_union[i_label])
                float_sum_iou += np_1d_intersection[i_label] / np_1d_union[i_label]
                dict_iou[str(i_label)] = str(np_1d_intersection[i_label] / np_1d_union[i_label])
            #라벨의 픽셀 수가 0인 경우
            else:
                dict_iou[str(i_label)] = "NaN"
    #print("calc labels:", int_available_labels)
    
    try:
        float_miou = float_sum_iou / int_available_labels
    except:
        float_miou = -9
    
    return float_miou, dict_iou

#=== End of calc_miou_gray



def calc_pa_ca_miou_gray(**kargs):
    # PIL 이미지 (gray) mIoU, Pixel Acc, Class Acc 계산 -> 손계산으로 검증 완료
    # Pixel Acc, Class Acc 공식 참고: https://ieeexplore.ieee.org/document/9311609
    # IN (**3):
    #       (pil) pil_gray_answer
    #           단일채널 정답 이미지
    #       (pil) pil_gray_predict
    #           단일채널 에측 이미지
    #       (int) int_total_labels
    #           전체 라벨 수 (void 포함)
    # IN (*):
    #       (int) int_void_label
    #           void 라벨값 (기본값 = -1)
    #
    # OUT (4):
    #       (float) Pixel_ACC, Class_Acc, mIoU
    #       (dict)  클래스 별 IoU
    #       
    #[INPUT]--------------------------------------------------------
    try:
        #(np) 정답 이미지
        in_np_ans = np.array(kargs['pil_gray_answer'])
        #(np) 예측 이미지
        in_np_pred = np.array(kargs['pil_gray_predict'])
        #(int) 전체 라벨 수
        in_int_total_labels = kargs['int_total_labels']
    except:
        print("(except) in calc_miou")
        print("more inputs required")
        sys.exit(9)
    
    try:
        #(int) void 라벨 번호 (-1 : void 없음)
        in_int_void = kargs['int_void_label']
    except:
        in_int_void = -1
    
    #[CALC]---------------------------------------------------------
    #전체 클래스 수 
    int_total_labels = in_int_total_labels
    
    #각 클래스별 빈도 수 (1d)
    np_count_ans = np.bincount(in_np_ans.reshape(-1), minlength = int_total_labels)
    #np_count_pred = np.bincount(in_np_pred.reshape(-1), minlength = int_total_labels)
    
    #이미지를 1차원 벡터로
    np_1d_ans = in_np_ans.reshape(-1)
    np_1d_pred = in_np_pred.reshape(-1)
    
    #(ans,pred) 조합을 번호로 변경하여 1차원 배열로 저장 (label 수를 기준으로 n진법 형태로 변환)
    np_1d_category = int_total_labels * np_1d_ans + np_1d_pred
    
    #confusion matrix 생성 (2d)
    np_1d_confusion_matrix = np.bincount(np_1d_category, minlength = int_total_labels*int_total_labels)
    np_2d_confusion_matrix_raw = np_1d_confusion_matrix.reshape(int_total_labels, int_total_labels)
    
    
    if(in_int_void == 0) :
        np_2d_confusion_matrix = np.append(np.zeros((1,in_int_total_labels))
                                          ,np.delete(np_2d_confusion_matrix_raw, in_int_void, axis = 0)
                                          ,axis = 0
                                          )
    elif (in_int_void == in_int_total_labels - 1):
        np_2d_confusion_matrix = np.append(np.delete(np_2d_confusion_matrix_raw, in_int_void, axis = 0)
                                          ,np.zeros((1,in_int_total_labels))
                                          ,axis = 0
                                          )
    else:
        #void 라벨 번호 입력 오류
        sys.exit(9)
    
    #클래스 별 IoU 계산
    np_1d_intersection = np.diag(np_2d_confusion_matrix)                # TP
    #np_1d_FP = np_2d_confusion_matrix.sum(axis=0) - np_1d_intersection  # FP
    #np_1d_FN = np_2d_confusion_matrix.sum(axis=1) - np_1d_intersection  # FN
    np_1d_TP_n_FP = np_2d_confusion_matrix.sum(axis=0)                  # TP + FP
    
    #print("np_1d_TP_n_FP", np_1d_TP_n_FP)
    
    np_1d_union = (np_1d_TP_n_FP
                  +np_2d_confusion_matrix.sum(axis=1)
                  -np_1d_intersection
                  )
    
    #연산에 사용한 라벨의 수
    int_available_labels = 0
    #유효한 iou 총 합
    float_sum_iou = 0
    #유효한 iou 딕셔너리
    #dict_iou = {"label":"iou"}
    dict_iou = {}
    
    _pa_up   = 0.0
    _pa_down = 0.0
    _ca_sum  = 0.0
    
    for i_label in range(int_total_labels):
        #void 라벨이 아닌 경우
        if(i_label != in_int_void):
            #라벨의 픽셀 수가 0이 아닌 경우
            if(np_count_ans[i_label] != 0):
                int_available_labels += 1
                
                #-- Pixel Acc & Class_Acc
                _pa_up   += np_1d_intersection[i_label]
                _pa_down += np_1d_TP_n_FP[i_label]
                
                if np_1d_TP_n_FP[i_label] != 0:
                    # 정답에 존재하는 class이나, 예측 결과에 그 class 라벨이 없으면 해당 class의 acc는 0점으로 처리
                    _ca_sum  += np_1d_intersection[i_label] / np_1d_TP_n_FP[i_label]
                
                
                #--- mIoU
                float_sum_iou += np_1d_intersection[i_label] / np_1d_union[i_label]
                dict_iou[str(i_label)] = str(np_1d_intersection[i_label] / np_1d_union[i_label])
            #라벨의 픽셀 수가 0인 경우
            else:
                dict_iou[str(i_label)] = "NaN"
    #print("calc labels:", int_available_labels)
    
    
    try:
        pixel_acc = float(_pa_up / _pa_down)
    except:
        pixel_acc = -9
    
    try:
        class_acc = float(_ca_sum / int_available_labels)
    except:
        class_acc = -9
    
    try:
        float_miou = float(float_sum_iou / int_available_labels)
    except:
        float_miou = -9
    
    return pixel_acc, class_acc, float_miou, dict_iou

#=== End of calc_pa_ca_miou_gray


def calc_pa_ca_miou_gray_tensor(**kargs):
    # Tensor ([H,W])를 통해 mIoU, Pixel Acc, Class Acc 계산 -> calc_pa_ca_miou_gray의 입력 자료형만 변경
    # Pixel Acc, Class Acc 공식 참고: https://ieeexplore.ieee.org/document/9311609
    # IN (**3):
    #       (Tensor) pil_gray_answer -> ts_ans
    #           단일채널 정답 이미지 -> 텐서
    #       (Tensor) pil_gray_predict -> ts_pred
    #           단일채널 에측 이미지 -> 텐서
    #       (int) int_total
    #           전체 라벨 수 (void 포함)
    #       (torch) device
    #           현재 사용 중인 device 정보
    # IN (*):
    #       (int) int_void
    #           void 라벨값 (기본값 = -1)
    #
    # OUT (4):
    #       (float) Pixel_ACC, Class_Acc, mIoU
    #       (dict)  클래스 별 IoU
    #       
    #[INPUT]--------------------------------------------------------
    device = kargs['device']
    
    try:
        # tensor 값은 해당 pixel의 class 값 (int)
        #(ts) 정답 이미지 텐서 in_np_ans -> ts_ans
        ts_ans = kargs['ts_ans'].type(torch.int64)
        #(ts) 예측 이미지 텐서 in_np_pred -> ts_pred
        ts_pred = kargs['ts_pred'].type(torch.int64)
        #(int) 전체 라벨 수 int_total_labels -> int_total
        int_total = kargs['int_total']
    except:
        print("(except) in calc_miou")
        print("more inputs required")
        sys.exit(9)
    
    try:
        # (int) void 라벨 번호 (-1 : void 없음) in_int_void -> int_void
        # 0번 혹은 끝번만 입력 가능
        int_void = kargs['int_void']
    except:
        int_void = -1
    
    name_func = "(calc_pa_ca_miou_gray_tensor) -> "
    #[CALC]---------------------------------------------------------
    
    #이미지를 1차원 벡터로
    #np_1d_ans -> ts_ans_1d
    #np_1d_pred -> ts_pred_1d
    
    ts_ans_1d  = ts_ans.reshape(-1)
    ts_pred_1d = ts_pred.reshape(-1)
    
    #각 클래스별 빈도 수 (1d)
    #np_count_ans -> count_ans
    #np_count_pred -> count_pred
    
    count_ans  = torch.bincount(ts_ans_1d,  minlength = int_total) > 0
    #count_pred = torch.bincount(ts_pred_1d, minlength = int_total)
    
    #(ans,pred) 조합을 번호로 변경하여 1차원 배열로 저장 (label 수를 기준으로 n진법 형태로 변환)
    #np_1d_category -> category_1d
    
    category_1d = int_total * ts_ans_1d + ts_pred_1d
    
    #confusion matrix 생성 (2d)
    #np_1d_confusion_matrix -> c_f_1d
    #np_2d_confusion_matrix_raw -> c_f_2d_raw
    
    c_f_1d = torch.bincount(category_1d, minlength = int_total*int_total)
    c_f_2d_raw = c_f_1d.reshape(int_total, int_total)
    
    if(int_void == 0) :
        #np_2d_confusion_matrix -> c_f_2d
        c_f_2d = torch.cat([torch.zeros((1,int_total), dtype=torch.int64).to(device)
                           ,c_f_2d_raw[1:,:]
                           ]
                          ,dim=0
                          )
    
    elif (int_void == int_total - 1):
        #np_2d_confusion_matrix -> c_f_2d
        c_f_2d = torch.cat([c_f_2d_raw[:-1,:]
                           ,torch.zeros((1,int_total), dtype=torch.int64).to(device)
                           ]
                          ,dim=0
                          )
    
    else:
        #void 라벨 번호 입력 오류
        print(name_func, "Void 번호는 0 또는", int_total - 1, "만 사용 가능합니다.")
        sys.exit(9)
    
    #클래스 별 IoU 계산
    #np_1d_intersection -> ts_1d_intersection
    #np_1d_TP_n_FP -> ts_1d_TP_n_FP
    
    ts_1d_intersection = torch.diag(c_f_2d)         # TP
    ts_1d_TP_n_FP = c_f_2d.sum(axis=0)              # TP + FP
    
    #np_1d_union -> ts_1d_union
    
    ts_1d_union = (ts_1d_TP_n_FP
                  +c_f_2d.sum(axis=1)
                  -ts_1d_intersection
                  )
    
    
    #연산에 사용한 라벨의 수
    int_available_labels = 0
    #유효한 iou 총 합
    float_sum_iou = 0
    #유효한 iou 딕셔너리 dict_iou = {"label":"iou"}
    dict_iou = {}
    
    _pa_up   = 0.0
    _pa_down = 0.0
    _ca_sum  = 0.0
    
    for i_label in range(int_total):
        #void 라벨이 아닌 경우
        if(i_label != int_void):
            #라벨의 픽셀 수가 0이 아닌 경우
            if(count_ans[i_label]):
                int_available_labels += 1
                
                #-- Pixel Acc & Class_Acc
                _pa_up   += ts_1d_intersection[i_label]
                _pa_down += ts_1d_TP_n_FP[i_label]
                
                if ts_1d_TP_n_FP[i_label] != 0:
                    # 정답에 존재하는 class이나, 예측 결과에 그 class 라벨이 없으면 해당 class의 acc는 0점으로 처리
                    _ca_sum  += ts_1d_intersection[i_label] / ts_1d_TP_n_FP[i_label]
                
                
                #--- mIoU
                float_iou = float(ts_1d_intersection[i_label] / ts_1d_union[i_label])
                float_sum_iou += float_iou
                dict_iou[str(i_label)] = str(float_iou)
            #라벨의 픽셀 수가 0인 경우
            else:
                dict_iou[str(i_label)] = "NaN"
    #print("calc labels:", int_available_labels)
    
    
    try:
        pixel_acc = float(_pa_up / _pa_down)
    except:
        pixel_acc = -9
    
    try:
        class_acc = float(_ca_sum / int_available_labels)
    except:
        class_acc = -9
    
    try:
        float_miou = float(float_sum_iou / int_available_labels)
    except:
        float_miou = -9
    
    return pixel_acc, class_acc, float_miou, dict_iou

#=== End of calc_pa_ca_miou_gray_tensor


#https://leedakyeong.tistory.com/entry/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4-%ED%95%A8%EC%88%98-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-in-%ED%8C%8C%EC%9D%B4%EC%8D%AC-softmax-in-python
def calc_np_softmax(in_np):
    in_max = np.max(in_np)
    in_np_exp = np.exp(in_np - in_max)
    in_np_exp_sum = np.sum(in_np_exp)
    
    return in_np_exp / in_np_exp_sum

#[검증용] -------------------------------------------------------------------------------------------------
'''
if True:
    a = np.array([1, 2, 3])
    b = calc_np_softmax(a)
    print(b)
    

if False:
    #https://gaussian37.github.io/vision-segmentation-miou/
    """
    x = [[0, 0, 0, 0],
         [0, 1, 1, 4],
         [5, 5, 2, 4],
         [5, 3, 3, 4]]

    y = [[0, 0, 0, 0],
         [0, 0, 1, 1],
         [5, 5, 2, 4],
         [5, 3, 3, 4]]
    """
    
    """
    #0: white(background)
    #1: red
    #2: green
    #3: blue
    #4: yellow
    
    x = [[1, 1, 0, 3, 0],
         [0, 0, 0, 3, 0],
         [0, 0, 0, 0, 0],
         [2, 2, 0, 4, 0],
         [2, 2, 4, 4, 4]]

    y = [[1, 4, 0, 3, 0],
         [0, 0, 0, 3, 0],
         [0, 0, 0, 0, 0],
         [2, 0, 0, 1, 0],
         [2, 2, 4, 4, 4]]
    """
    # 교집합 / 합집합 / IoU
    #0: 5 / 8 / 0.625
    #1: 3 / 7 / 0.429
    #2: 4 / 6 / 0.667
    #3: 1 / 2 / 0.5
    #4: 2 / 2 / 1
    
    x = [[0, 0, 0, 2, 2],
         [0, 1, 1, 2, 2],
         [1, 1, 1, 0, 3],
         [4, 4, 0, 0, 3]]

    y = [[0, 0, 1, 2, 2],
         [0, 1, 2, 2, 2],
         [0, 1, 1, 1, 2],
         [4, 4, 0, 0, 3]]
    
    
    #int_void_label = -1 : void 없음
    z = calc_miou(pil_gray_answer = Image.fromarray(np.array(x)),
                  pil_gray_predict = Image.fromarray(np.array(y)),
                  int_void_label = 1,
                  int_total_labels = 5)
    
    print("calc_miou",z)
'''
#---
'''
#pytorch -> ignite
cm = ConfusionMatrix(num_classes=3)
metric = mIoU(cm, ignore_index=0)
metric.attach(default_evaluator, 'miou')
y_true = torch.Tensor([0, 1, 0, 1, 2]).long()
y_pred = torch.Tensor([
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
])
state = default_evaluator.run([[y_pred, y_true]])
print(state.metrics['miou'])
'''


print("EoF: calc_func.py")