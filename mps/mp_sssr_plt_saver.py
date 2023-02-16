# mp_pils_saver.py -> mp.sssr_plt_saver

# mp 활용 함수들은 따로 py 파일 분리해야됨
# 아직 py 파일 프로세스 수만큼 실행되는 원인 못찾음
import os
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image
#----------------------------------------------------------------------------------------------------------------------

def MOD_pils_show_save(**kargs):
    name_func = "MOD_pils_show_save"
    '''
    pils_show_save(in_pil_1 = Image.open(path_x[0])
                  ,in_pil_2 = dict_out_degraded_image[path_x]
                  ,in_pil_3 = dict_out_restored_image[path_x]
                  ,in_pil_4 = label_2_RGB(in_pil_y_raw
                                          ,HP_COLOR_MAP
                                          )
                  ,in_pil_5 = out_pil_x_degrad_ss_hypo
                  ,in_pil_6 = out_pil_x_sr_hypo_ss_hypo
                  
                  ,in_title_sub_1 = "Origianl Image"
                  ,in_title_sub_2 = "Degraded Image"
                  ,in_title_sub_3 = title_sub_3
                  ,in_title_sub_4 = "Original Label"
                  ,in_title_sub_5 = "Predicted with Degraded Image"
                  ,in_title_sub_6 = title_sub_6
                  #(bool) 각 이미지 크기정보를 title_sub에 덧붙일것인가?
                  ,is_add_size_info = True
                  
                  ,show = tmp_show
                  ,save = True
                  ,path = in_path_out + "images/"
                  ,name = path_x[0].split("/")[-1].split(".")[0] + ".png"
                  ,title = record_plt_title
                  ,figsize = (21, 14)
                  )

    '''
    #PIL 이미지 n개 동시 출력 및 저장 ((1줄) 1,2,3, (2줄) 4,6장 입력 가능)
    #IN (2,3,4,6): (pil) 이미지
    #IN (*): 'show' : (bool) plt show 여부 
    #        'save' : (bool) 이미지 저장기능여부
    #        'path' : (str)  저장경로
    #        'name' : (str)  이미지 파일 이름
    #        'title': (str)  전체 타이틀 내용
    #        'in_title_sub_1': (str) 이미지 서브 타이틀 1
    #        'in_title_sub_2': (str) 이미지 서브 타이틀 2
    #        'in_title_sub_3': (str) 이미지 서브 타이틀 3
    #        'in_title_sub_4': (str) 이미지 서브 타이틀 4

    in_pil_count = 0
    #---(이미지, title_sub 입력)
    try:
        in_pil_1 = kargs['in_pil_1']
        in_pil_count += 1
        in_title_sub_1 = kargs['in_title_sub_1']
    except:
        in_title_sub_1 = "False"
    
    try:
        in_pil_2 = kargs['in_pil_2']
        in_pil_count += 1
        in_title_sub_2 = kargs['in_title_sub_2']
    except:
        in_title_sub_2 = "False"
    
    try:
        in_pil_3 = kargs['in_pil_3']
        in_pil_count += 1
        in_title_sub_3 = kargs['in_title_sub_3']
    except:
        in_title_sub_3 = "False"
    
    try:
        in_pil_4 = kargs['in_pil_4']
        in_pil_count += 1
        in_title_sub_4 = kargs['in_title_sub_4']
    except:
        in_title_sub_4 = "False"
    
    #pil 6장 지원
    try:
        in_pil_5 = kargs['in_pil_5']
        in_pil_6 = kargs['in_pil_6']
        in_pil_count += 2
        in_title_sub_5 = kargs['in_title_sub_5']
        in_title_sub_6 = kargs['in_title_sub_6']
    except:
        in_title_sub_5 = "False"
        in_title_sub_6 = "False"
    
    #---(옵션 설정)
    try: #plt 창 출력여부
        in_switch_show = kargs['show']
    except:
        in_switch_show = False
    
    try: #plt 이미지 저장경로
        in_path = kargs['path']
        if in_path[-1] != "/":
            in_path += "/"
    except:
        in_path = "False"
    
    try: #이미지 파일 이름
        in_name = kargs['name']
    except:
        in_name = "False"
    
    try: #전체 타이틀 내용
        in_title = kargs['title']
    except:
        in_title = "False"
        
    try: #저장기능 사용여부
        in_switch_save = kargs['save']
        if in_path == "False" or in_name == "False":
            in_switch_save = False
    except:
        in_switch_save = False
    
    
    #----
    #print(in_switch_show, in_switch_save)
    
    if 2 <= in_pil_count and in_pil_count <= 3:
        #입력 이미지 개수가 
        try:
            tuple_figsize = kargs['figsize']
            fig = plt.figure(figsize = tuple_figsize)
        except:
            fig = plt.figure(figsize = (in_pil_count * 7, 6))
        rows = 1
        cols = in_pil_count
    
    elif in_pil_count == 4:
        try:
            tuple_figsize = kargs['figsize']
            fig = plt.figure(figsize = tuple_figsize)
        except:
            fig = plt.figure(figsize = (16, 14))
        rows = 2 #가로 줄 수
        cols = 2 #세로 줄 수
    
    else: # 6장
        try:
            tuple_figsize = kargs['figsize']
            fig = plt.figure(figsize = tuple_figsize)
        except:
            fig = plt.figure(figsize = (16, 14))
        rows = 2 #가로 줄 수
        cols = 3 #세로 줄 수
    
    is_add_size_info = kargs['is_add_size_info']
    
    #pil 이미지 -> str "(w000,h000)"으로 변환
    def _pil_2_info(in_pil, is_work):
        if is_work:
            in_w, in_h = in_pil.size
            return " (w" + str(in_w) + ", h" + str(in_h) + ")"
        else:
            return ""
    
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(np.array(in_pil_1))
    if in_title_sub_1 != "False":
        ax1.set_title(in_title_sub_1 + _pil_2_info(in_pil_1, is_add_size_info))
    #ax3.axis("off")
    
    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(np.array(in_pil_2))
    if in_title_sub_2 != "False":
        ax2.set_title(in_title_sub_2 + _pil_2_info(in_pil_2, is_add_size_info))
    #ax3.axis("off")
    
    if in_pil_count >= 3:
        ax3 = fig.add_subplot(rows, cols, 3)
        ax3.imshow(np.array(in_pil_3))
        if in_title_sub_3 != "False":
            ax3.set_title(in_title_sub_3 + _pil_2_info(in_pil_3, is_add_size_info))
        #ax3.axis("off")
    
    if in_pil_count >= 4:
        ax4 = fig.add_subplot(rows, cols, 4)
        ax4.imshow(np.array(in_pil_4))
        if in_title_sub_4 != "False":
            ax4.set_title(in_title_sub_4 + _pil_2_info(in_pil_4, is_add_size_info))
        #ax4.axis("off")
    
    if in_pil_count == 6:
        ax5 = fig.add_subplot(rows, cols, 5)
        ax5.imshow(np.array(in_pil_5))
        if in_title_sub_5 != "False":
            ax5.set_title(in_title_sub_5 + _pil_2_info(in_pil_5, is_add_size_info))
        #ax5.axis("off")
        ax6 = fig.add_subplot(rows, cols, 6)
        ax6.imshow(np.array(in_pil_6))
        if in_title_sub_6 != "False":
            ax6.set_title(in_title_sub_6 + _pil_2_info(in_pil_6, is_add_size_info))
        #ax6.axis("off")
    
    
    if in_title != "False":
        fig.suptitle(in_title)
    
    if in_switch_save:
        #print("save_path:", in_path)
        #print("file_name:", in_name)
        
        in_path_name = in_path + in_name
        
        try:
            if not os.path.exists(in_path):
                os.makedirs(in_path)
            
            try:
                plt.savefig(in_path_name, dpi = 100)
                #print("fig saved:", in_path_name)
            except:
                print("(exc)", name_func, "save FAIL:", in_path_name)
                
        except OSError:
            print("(exc)", name_func, "makedirs FAIL:", in_path)
    
    if in_switch_show:
        plt.show()
    
    
    plt.close(fig)
    

#=== End of MOD_pils_show_save

def ts_minmax(in_ts):
        #min-max scaler (min:0, max:1)
        return (in_ts - torch.min(in_ts)) / (torch.max(in_ts) - torch.min(in_ts))

def MOD_cross_ft_2_plt(**kargs): # cross features to single plt
    name_func = "MOD_cross_ft_2_plt"
    # modified verison of cross_ft_2_plt_v2
    
    # only 1 part of batch should be input
    # tensor should be a input with .clone().detach().cpu()
    ts_pred_lab = kargs['ts_pred_lab']             # (ts) 3ch tensor (c, h, w) predicted label
    ts_pred_img = kargs['ts_pred_img']             # (ts) 3ch tensor (c, h, w) predicted image
    ts_ans_lab = kargs['ts_ans_lab']               # (ts) 3ch tensor (c, h, w) answer label
    ts_ans_img = kargs['ts_ans_img']               # (ts) 3ch tensor (c, h, w) answer image
    
    try:
        plt_title = kargs['plt_title']                                  # (str)  plt main title
    except:
        plt_title = "Cross Feature Visualize"
    
    try:
        is_show = kargs['is_show']                                         # (bool) do you want to show result?
    except:
        is_show = False
    
    try:
        is_save = kargs['is_save']                                         # (bool) do you want to save result?
    except:
        is_save = False
    
    
    try:
        in_path = kargs['path']                                     # (str) save file path
        if in_path[-1] != "/":
            in_path += "/"
    except:
        in_path = "False"
    
    try:
        in_name = kargs['name']                                     # (str) save file name
        if in_name[0] == '/':
            in_name = in_name[1:]
    except:
        in_name = "False"
    
    try:
        fig_size = kargs['fig_size']                                #(tuple) fig size (w,h)
        flag_fig_size_set = True
    except:
        flag_fig_size_set = False
    
    
    # all input H,W are same
    c_1, h_1, w_1 = ts_pred_lab.shape       # c = classes
    c_2, h_2, w_2 = ts_pred_img.shape       # c = 3 RGB
    c_3, h_3, w_3 = ts_ans_lab.shape        # c = classes
    c_4, h_4, w_4 = ts_ans_img.shape        # c = 3 RGB
    
    list_pil_pred_lab_ans_img = []
    list_pil_ans_lab_pred_img = []
    
    
    #--- generate cross-feature (1: pred_lab & ans_img, 2: ans_lab & pred_img)
    # transform image to GRAY: https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cv2.cvtColor
    # Gray = 0.299*R + 0.587*G + 0.114*B
    # Tensor C Channel order: R G B
    ts_pred_img_g = ts_minmax(ts_pred_img[0]*0.299 + ts_pred_img[1]*0.587 + ts_pred_img[2]*0.114)       # [h_2, w_2]
    ts_ans_img_g  = ts_minmax(ts_ans_img[0]*0.299  + ts_ans_img[1]*0.587  + ts_ans_img[2]*0.114)        # [h_4, w_4]
    
    
    if flag_fig_size_set:
        plt.figure(figsize = fig_size)
    else:
        plt.figure(figsize = (5*c_1, 10))
    
    list_class_name = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
                      ,"10", "11", "12", "13", "14", "15", "16", "17", "18", "19"
                      ,"20", "21", "22", "23", "24", "25", "26", "27", "28", "29"]
    
    if c_1 == 11:   #CamVid 12
        list_class_name = ["Sky", "Building", "Column pole", "Road", "Sidewalk"
                          ,"Tree", "Sign Symbol", "Fence", "Car", "Pedestrian", "Bicyclist"
                          ]
        
    
    
    for i_channel in range(c_1):
        plt.subplot(2, c_1, i_channel + 1)
        plt.imshow(to_pil_image(ts_minmax(ts_pred_lab[i_channel]) * ts_ans_img_g))
        plt.title('Pred_Lab & Ans_Img\nClass: ' + list_class_name[i_channel])
        
        plt.subplot(2, c_1, i_channel + 1 + c_1)
        plt.imshow(to_pil_image(ts_minmax(ts_ans_lab[i_channel]) * ts_pred_img_g))
        plt.title('Ans_Lab & Pred_Img\nClass: ' + list_class_name[i_channel])
    
    
    plt.suptitle(plt_title)
    
    if is_show:
        plt.show()
    
    if is_save:
        in_path_name = in_path + in_name
        
        try:
            if not os.path.exists(in_path):
                os.makedirs(in_path)
            
            try:
                plt.savefig(in_path_name, dpi = 100)
                #print("fig saved:", in_path_name)
            except:
                print("(exc)", name_func, "save FAIL:", in_path_name)
                
        except OSError:
            print("(exc)", name_func, "makedirs FAIL:", in_path)
        
    
    plt.close()
    



#=== End of cross_ft_2_plt_v2

#----------------------------------------------------------------------------------------------------------------------

def worker_plt(in_total):
    name_func = "worker_plt"
    model_type = in_total[0]
    
    # trainer dsrl v7.6 ~
    if model_type == "SSSR_A" or model_type == "SSSR_B" or model_type == "SSSR_C" or model_type == "SSSR_D":
        '''
        #-- plt a: pils_show_save
        # pils
        a_pil_1 = in_total[1]
        a_pil_2 = in_total[2]
        a_pil_3 = in_total[3]
        a_pil_4 = in_total[4]
        a_pil_5 = in_total[5]
        a_pil_6 = in_total[6]
        
        # sub-title for pils
        a_t_sub_1 = in_total[7]
        a_t_sub_2 = in_total[8]
        a_t_sub_3 = in_total[9]
        a_t_sub_4 = in_total[10]
        a_t_sub_5 = in_total[11]
        a_t_sub_6 = in_total[12]
        
        # save path
        a_file_path = in_total[13]
        
        
        #-- plt b: cross_ft_2_plt
        # tensor (predict lab & img, answer lab & img)
        b_ts_p_lab = in_total[14]
        b_ts_p_img = in_total[15]
        b_ts_a_lab = in_total[16]
        b_ts_a_img = in_total[17]
        
        # save path
        b_file_path = in_total[18]      # CF
        c_file_path = in_total[19]      # SR image 
        
        #-- shared
        # main title for plt
        s_t_main = in_total[20]
        # file name
        s_file_name = in_total[21]
        '''
        
        # 알 수 없는 이유로 첫 폴더 생성이 실패하는 경우가 발생함
        # 그러한 현상 방치 목적으로 폴더 검사를 한번씩 미리 시행
        
        try:
            if not os.path.exists(in_total[13]):
                os.makedirs(in_total[13])
        except OSError:
            #print("(exc)", name_func, "folder gen FAIL:", in_total[13])
            pass
        
        try:
            if not os.path.exists(in_total[18]):
                os.makedirs(in_total[18])
        except OSError:
            #print("(exc)", name_func, "folder gen FAIL:", in_total[18])
            pass
        
        try:
            if not os.path.exists(in_total[19]):
                os.makedirs(in_total[19])
        except OSError:
            #print("(exc)", name_func, "folder gen FAIL:", in_total[19])
            pass
        
        MOD_pils_show_save  (in_pil_1 = in_total[1]
                            ,in_pil_2 = in_total[2]
                            ,in_pil_3 = in_total[3]
                            ,in_pil_4 = in_total[4]
                            ,in_pil_5 = in_total[5]
                            ,in_pil_6 = in_total[6]
                            
                            # (bool) 각 이미지 크기정보를 title_sub에 덧붙일것인가?
                            ,is_add_size_info = True
                            
                            ,in_title_sub_1 = in_total[7]
                            ,in_title_sub_2 = in_total[8]
                            ,in_title_sub_3 = in_total[9]
                            ,in_title_sub_4 = in_total[10]
                            ,in_title_sub_5 = in_total[11]
                            ,in_title_sub_6 = in_total[12]
                            
                            ,show = False
                            ,save = True
                            ,path = in_total[13]
                            ,name = in_total[21]
                            
                            ,title = in_total[20]
                            ,figsize = (21, 16)
                            )
        
        if model_type == "SSSR_A" or model_type == "SSSR_B" or model_type == "SSSR_C":
            MOD_cross_ft_2_plt  (ts_pred_lab = in_total[14]
                                ,ts_pred_img = in_total[15]
                                ,ts_ans_lab = in_total[16]
                                ,ts_ans_img = in_total[17]
                                
                                ,is_save = True
                                ,path = in_total[18]
                                ,name = in_total[21]
                                
                                ,plt_title = in_total[20]
                                )
        
        
        
        #print('*', end='')
        print("\rSaved", in_total[21], end='')



def worker_plt_pil(in_total):
    name_func = "worker_plt"
    model_type = in_total[0]
    
    # trainer dsrl v7.6 ~
    if model_type == "SSSR_A" or model_type == "SSSR_B" or model_type == "SSSR_C" or model_type == "SSSR_D":
        '''
        #-- plt a: pils_show_save
        # pils
        a_pil_1 = in_total[1]
        a_pil_2 = in_total[2]
        a_pil_3 = in_total[3]
        a_pil_4 = in_total[4]
        a_pil_5 = in_total[5]
        a_pil_6 = in_total[6]
        
        # sub-title for pils
        a_t_sub_1 = in_total[7]
        a_t_sub_2 = in_total[8]
        a_t_sub_3 = in_total[9]
        a_t_sub_4 = in_total[10]
        a_t_sub_5 = in_total[11]
        a_t_sub_6 = in_total[12]
        
        # save path
        a_file_path = in_total[13]
        
        
        #-- plt b: cross_ft_2_plt
        # tensor (predict lab & img, answer lab & img)
        b_ts_p_lab = in_total[14]
        b_ts_p_img = in_total[15]
        b_ts_a_lab = in_total[16]
        b_ts_a_img = in_total[17]
        
        # save path
        b_file_path = in_total[18]      # CF
        c_file_path = in_total[19]      # SR image 
        
        #-- shared
        # main title for plt
        s_t_main = in_total[20]
        # file name
        s_file_name = in_total[21]
        '''
        
        # 알 수 없는 이유로 첫 폴더 생성이 실패하는 경우가 발생함
        # 그러한 현상 방치 목적으로 폴더 검사를 한번씩 미리 시행
        
        try:
            if not os.path.exists(in_total[13]):
                os.makedirs(in_total[13])
        except OSError:
            #print("(exc)", name_func, "folder gen FAIL:", in_total[13])
            pass
        
        try:
            if not os.path.exists(in_total[18]):
                os.makedirs(in_total[18])
        except OSError:
            #print("(exc)", name_func, "folder gen FAIL:", in_total[18])
            pass
        
        try:
            if not os.path.exists(in_total[19]):
                os.makedirs(in_total[19])
        except OSError:
            #print("(exc)", name_func, "folder gen FAIL:", in_total[19])
            pass
        
        MOD_pils_show_save  (in_pil_1 = in_total[1]
                            ,in_pil_2 = in_total[2]
                            ,in_pil_3 = in_total[3]
                            ,in_pil_4 = in_total[4]
                            ,in_pil_5 = in_total[5]
                            ,in_pil_6 = in_total[6]
                            
                            # (bool) 각 이미지 크기정보를 title_sub에 덧붙일것인가?
                            ,is_add_size_info = True
                            
                            ,in_title_sub_1 = in_total[7]
                            ,in_title_sub_2 = in_total[8]
                            ,in_title_sub_3 = in_total[9]
                            ,in_title_sub_4 = in_total[10]
                            ,in_title_sub_5 = in_total[11]
                            ,in_title_sub_6 = in_total[12]
                            
                            ,show = False
                            ,save = True
                            ,path = in_total[13]
                            ,name = in_total[21]
                            
                            ,title = in_total[20]
                            ,figsize = (21, 16)
                            )
        
        if model_type == "SSSR_A" or model_type == "SSSR_B" or model_type == "SSSR_C":
            MOD_cross_ft_2_plt  (ts_pred_lab = in_total[14]
                                ,ts_pred_img = in_total[15]
                                ,ts_ans_lab = in_total[16]
                                ,ts_ans_img = in_total[17]
                                
                                ,is_save = True
                                ,path = in_total[18]
                                ,name = in_total[21]
                                
                                ,plt_title = in_total[20]
                                )
        
        
        #SR image & SRSS label save
        if in_total[19][-1] == '/':
            tmp_path_name     = in_total[19] + in_total[21]
            tmp_path_name_lab = in_total[19] + "label_" + in_total[21]
        else:
            tmp_path_name     = in_total[19] + '/' + in_total[21]
            tmp_path_name_lab = in_total[19] + '/label_' + in_total[21]
        
        
        try:
            if not os.path.exists(in_total[19]):
                os.makedirs(in_total[19])
            try:
                in_total[6].save(tmp_path_name)
            except:
                print("(exc) PIL save FAIL:", tmp_path_name)
            
            try:
                in_total[3].save(tmp_path_name_lab)
            except:
                print("(exc) PIL save FAIL:", tmp_path_name_lab)
            
        except OSError:
            print("(exc)", name_func, "folder gen FAIL:", in_total[19])
        
        
        #print('*', end='')
        print("\rSaved", in_total[21], end='')



#----------------------------------------------------------------------------------------------------------------------



def plts_saver(list_data, **kargs):
    name_func = "plts_saver"
    print("")
    
    # list_data     : list with tuple (data 1, data 2, data 3, ...)
    # is_best       : (bool) 이번 epoch의 best 여부                        default: False
    # num_worker    : number of sub_processes
    # no_employ     : (bool) do not generate sub-process                default: False
    
    try:
        num_worker = int(kargs['num_worker'])
    except:
        try:
            total_worker = mp.cpu_count()
            num_worker = total_worker // 2
        except:
            num_worker = 1
    
    try:
        is_best = kargs['is_best']
    except:
        is_best = False
    
    try:
        no_employ = kargs['no_employ']
        if no_employ:
            num_worker = 1
    except:
        pass
    
    if num_worker <= 1:
        if num_worker < 1:
            num_worker = 1
            print("(caution)", name_func, "num_worker was < 1, so changed to 1")
        
        print(name_func, "started with no sub-process")
        
        for i_elem in list_data:
            if is_best:
                # plt & pil save
                worker_plt_pil(i_elem)
            else:
                # only plt save
                worker_plt(i_elem)
        
    else:
        # num_worker > 1
        if num_worker > 4:
            num_worker = 4      # CPU 과부하 방지
        
        print(name_func, "started with", num_worker, "workers")
        
        pool = mp.Pool(num_worker)
        
        if is_best:
            # plt & pil save
            pool.map(worker_plt_pil, list_data)
        else:
            # only plt save
            pool.map(worker_plt, list_data)
        
        pool.close()
        pool.join()
    
    print("")
    print(name_func, "finished\n")

#=== End of pils_saver


if __name__ == '__main__':
    print("End of mp_pils_saver.py")