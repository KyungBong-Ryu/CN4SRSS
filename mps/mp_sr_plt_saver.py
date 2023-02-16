# mp_pils_saver.py -> mp_sr_plt_saver

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


#----------------------------------------------------------------------------------------------------------------------

def worker_plt(in_total):
    name_func = "worker_plt"
    model_type = in_total[0]
    
    if model_type == "SR_A" or model_type == "SR_B" or model_type == "SR_C":
        '''
        #-- plt a: pils_show_save
        # pils
        a_pil_1 = in_total[1]
        a_pil_2 = in_total[2]
        a_pil_3 = in_total[3]
        
        # sub-title for pils
        a_t_sub_1 = in_total[4]
        a_t_sub_2 = in_total[5]
        a_t_sub_3 = in_total[6]
        
        # save path
        a_file_path = in_total[7]       # path plt
        b_file_path = in_total[8]       # path pil
        
        # main title for plt
        a_t_main = in_total[9]
        
        #-- shared
        
        # file name
        s_file_name = in_total[10]
        
        '''
        
        # 알 수 없는 이유로 첫 폴더 생성이 실패하는 경우가 발생함
        # 그러한 현상 방치 목적으로 폴더 검사를 한번씩 미리 시행
        
        try:
            if not os.path.exists(in_total[7]):
                os.makedirs(in_total[7])
        except OSError:
            print("(exc)", name_func, "folder gen FAIL:", in_total[7])
        
        
        MOD_pils_show_save  (in_pil_1 = in_total[1]
                            ,in_pil_2 = in_total[2]
                            ,in_pil_3 = in_total[3]
                            
                            # (bool) 각 이미지 크기정보를 title_sub에 덧붙일것인가?
                            ,is_add_size_info = True
                            
                            ,in_title_sub_1 = in_total[4]
                            ,in_title_sub_2 = in_total[5]
                            ,in_title_sub_3 = in_total[6]
                            
                            ,show = False
                            ,save = True
                            ,path = in_total[7]
                            ,name = in_total[10]
                            
                            ,title = in_total[9]
                            ,figsize = (18, 9)
                            )
        
        
        #print('*', end='')
        print("\rSaved", in_total[10], end='')



def worker_plt_pil(in_total):
    name_func = "worker_plt"
    model_type = in_total[0]
    
    if model_type == "SR_A" or model_type == "SR_B" or model_type == "SR_C":
        '''
        #-- plt a: pils_show_save
        # pils
        a_pil_1 = in_total[1]
        a_pil_2 = in_total[2]
        a_pil_3 = in_total[3]
        
        # sub-title for pils
        a_t_sub_1 = in_total[4]
        a_t_sub_2 = in_total[5]
        a_t_sub_3 = in_total[6]
        
        # save path
        a_file_path = in_total[7]       # path plt
        b_file_path = in_total[8]       # path pil
        
        # main title for plt
        a_t_main = in_total[9]
        
        #-- shared
        
        # file name
        s_file_name = in_total[10]
        
        '''
        
        # 알 수 없는 이유로 첫 폴더 생성이 실패하는 경우가 발생함
        # 그러한 현상 방치 목적으로 폴더 검사를 한번씩 미리 시행
        
        try:
            if not os.path.exists(in_total[7]):
                os.makedirs(in_total[7])
        except OSError:
            #print("(exc)", name_func, "folder gen FAIL:", in_total[7])
            pass
        
        try:
            if not os.path.exists(in_total[8]):
                os.makedirs(in_total[8])
        except OSError:
            #print("(exc)", name_func, "folder gen FAIL:", in_total[8])
            pass
        
        MOD_pils_show_save  (in_pil_1 = in_total[1]
                            ,in_pil_2 = in_total[2]
                            ,in_pil_3 = in_total[3]
                            
                            # (bool) 각 이미지 크기정보를 title_sub에 덧붙일것인가?
                            ,is_add_size_info = True
                            
                            ,in_title_sub_1 = in_total[4]
                            ,in_title_sub_2 = in_total[5]
                            ,in_title_sub_3 = in_total[6]
                            
                            ,show = False
                            ,save = True
                            ,path = in_total[7]
                            ,name = in_total[10]
                            
                            ,title = in_total[9]
                            ,figsize = (18, 9)
                            )
        
        #SR image save
        if in_total[8][-1] == '/':
            tmp_path_name = in_total[8] + in_total[10]
        else:
            tmp_path_name = in_total[8] + '/' + in_total[10]
        
        
        try:
            if not os.path.exists(in_total[8]):
                os.makedirs(in_total[8])
            try:
                in_total[3].save(tmp_path_name)
            except:
                print("(exc) PIL save FAIL:", tmp_path_name)
        except OSError:
            print("(exc)", name_func, "folder gen FAIL:", in_total[8])
        
        
        
        
        #print('*', end='')
        print("\rSaved", in_total[10], end='')




#----------------------------------------------------------------------------------------------------------------------

def plts_saver(list_data, **kargs):
    name_func = "plts_saver"
    print("")
    
    # list_data     : list with tuple (data 1, data 2, data 3, ...)
    # is_best       : (bool) 이번 epoch의 best 여부                        default: False
    # num_worker    : (int) number of sub_processes
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
    print("End of mp_sr_plt_saver.py")