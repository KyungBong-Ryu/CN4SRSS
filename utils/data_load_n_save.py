# data_load_n_save.py
# v 1.06 -> v 1.09 : Custom_Dataset_V3에 세부설명 주석만 추가됨

import os
import io
from PIL import Image
import numpy as np
import sys
import copy
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import cv2
import csv

import math
import torchvision.transforms as transforms
import random
from torchvision.transforms.functional import to_pil_image

import warnings

from utils.data_tool import ColorJitter_Double, pil_augm_v3, pil_2_patch_v6, pil_marginer_v3, pil_augm_lite, imshow_pil, imshow_ts, imshow_np

#----------------------
#verified in spyder
class RecordBox():
    #-------------------------
    # 1. init
    #   box_loss = RecordBox(name = "loss_name")
    #
    # 2. add item in batch
    #   box_loss.add_item(some_float_value)
    #
    # 3. at the end of batch
    #   box_loss.update_batch()     #-> str 형으로 log return 기능 추가히기 (is_return_str)
    #
    # 4. at the end of epoch
    #   box_loss.update_epoch() #this draws log graph
    #-------------------------
    
    def __init__(self, name='name', is_print=True, print_interval=1, will_update_graph=True, **kargs):
        # Parameter name
        self.name = name
        self.print_head = "(RecordBox) " + self.name + " -> "
        
        #print(self.print_head, "init")
        
        self.is_print = is_print
        
        if print_interval < 1:
            self.print_interval = 1
        else:
            self.print_interval = int(print_interval)
        
        
        #print(self.print_head, "Generated RecordBox name:", self.name)
        if '/' in self.name:
            print(self.print_head, "(exc) name should not include /")
            sys.exit(9)
        
        
        if self.is_print:
            print(self.print_head, "Update Batch log will be printed once per every", str(self.print_interval), "updates")
        
        
        self.count_single_item = 0                  # (float)   single item number in 1 batch
        self.single_item_sum = 0                    # (float)   single item sum per batch
        
        
        self.count_batch = 0                        # (int)     batch number in 1 epoch
        self.batch_item = 0                         # (float)   avg per batch (single_item_sum / count_single_item)
        self.batch_item_sum = 0                     # (float)   batch_item sum per epoch
        self.record_batch_item = []                 # (list)    all batch_item in 1 epoch
        self.record_batch_item_prev = []            # (list)    prev epoch's record_batch_item
        
        self.count_epoch = 0                        # (int)     epoch number in total RUN
        self.epoch_item = 0                         # (float)   avg per epoch (batch_item_sum / count_batch)
        self.record_epoch_item = []                 # (list)    all epoch_items in total RUN
        
        self.total_min = (0,0)                      # (tuple)   (epoch number, min epoch_item)
        self.total_max = (0,0)                      # (tuple)   (epoch number, MAX epoch_item)
        
        self.count_fig_save = 0                     # (int)     log fig save count
        
        self.is_best_max = False                    # (bool)    is lastly updated value is best (max)
        self.is_best_min = False                    # (bool)    is lastly updated value is best (min)
        
        self.will_update_graph = will_update_graph  # (bool)    will update(save) log graph
        
        if self.will_update_graph:
            print(self.print_head, "This will update graph every epoch.")
        else:
            print(self.print_head, "This will NOT update graph every epoch!")
        
        #--- for colab (RUN_WHERE == -1)
        '''
        try:
            self.RUN_WHERE = kargs['RUN_WHERE']
            self.PATH_COLAB_OUT_LOG = kargs['PATH_COLAB_OUT_LOG']
            if self.PATH_COLAB_OUT_LOG[-1] != '/':
                self.PATH_COLAB_OUT_LOG += '/'
        except:
            self.RUN_WHERE = 1
            self.PATH_COLAB_OUT_LOG = "False"
        '''
    
    #--- functions used outside
    
    # add new item (in batch)
    def add_item(self, item):
        self.count_single_item += 1         # (int)     update to current number of items
        
        self.single_item_sum += item        # (float)   Sum items in batch
    
    
    # update when batch end
    def update_batch(self, is_return = False):
        self.count_batch += 1               # (int)     update to current number of batches
        
        try:
            self.batch_item = self.single_item_sum / self.count_single_item
            self.single_item_sum = 0
            self.count_single_item = 0
        except:
            print(self.print_head, "(exc) self.count_single_item is Zero")
            sys.exit(9)
        
        self.batch_item_sum += self.batch_item
        
        if self.is_print and (self.count_batch - 1) % self.print_interval == 0:
            print(self.print_head, "update batch <", str(self.count_epoch + 1), "-" , str(self.count_batch), ">", str(round(self.batch_item, 4)))
        
        self.record_batch_item.append(self.batch_item)
        
        if is_return:
            # returns avg value of items in batch
            return self.batch_item
    
    
    # update when epoch end
    def update_epoch(self, is_return = False
                    ,is_show = False, is_save = True, path = "./"
                    ,is_print_sub = False
                    ,is_update_graph = None
                    ):
        self.is_best_max = False            # reset flag
        self.is_best_min = False            # reset flag
        self.count_epoch += 1               # update to current number of epoches
        
        if is_update_graph is None:
            _update_graph = self.will_update_graph
        else:
            _update_graph = is_update_graph
            if _update_graph:
                print(self.print_head, "graph updated.")
            else:
                print(self.print_head, "graph update SKIPPED!")
        
        try:
            self.epoch_item = self.batch_item_sum / self.count_batch
            self.batch_item_sum = 0
            self.count_batch = 0
        except:
            print(self.print_head, "(exc) self.count_batch is Zero")
            sys.exit(9)
        
        if self.is_print or is_print_sub:
            if _update_graph:
                print(self.print_head, "update epoch <", str(self.count_epoch), ">", str(round(self.epoch_item, 4)))
        
        self.record_epoch_item.append(self.epoch_item)
        
        if self.count_epoch == 1:
            self.total_min = (self.count_epoch, self.epoch_item)
            self.total_max = (self.count_epoch, self.epoch_item)
            self.is_best_max = True
            self.is_best_min = True
        else:
            if self.total_min[-1] >= self.epoch_item:
                self.total_min = (self.count_epoch, self.epoch_item)
                self.is_best_min = True
            
            if self.total_max[-1] <= self.epoch_item:
                self.total_max = (self.count_epoch, self.epoch_item)
                self.is_best_max = True
        
        self.record_batch_item_prev = copy.deepcopy(self.record_batch_item)
        self.record_batch_item = []
        
        #--- for update_graph()
        self.is_show = is_show
        self.is_save = is_save
        self.path = path
        
        if _update_graph:
            self.update_graph(is_show = self.is_show, is_save = self.is_save, path = self.path)
        
        if is_return:
            # returns avg value of items in epoch
            return self.epoch_item
    
    # return record list
    def get_record(self, select):
        if select == 'batch':
            # get last updated epoch's batch item list (per epoch)
            return self.record_batch_item_prev
        elif select == 'epoch':
            # get total's epoch item list
            return self.record_epoch_item
    
    #--- functions used inside class
    
    # save graph
    def update_graph(self, is_show = False, is_save = True, path = "./"):
        self.count_fig_save += 1
        
        list_x_labels = []
        for i in range(len(self.record_epoch_item)):
            list_x_labels.append(i+1)
        
        plt.figure(figsize=(10,8))
        # main graph
        plt.plot(list_x_labels, self.record_epoch_item)
        # MAX point
        plt.scatter([int(self.total_max[0])], [self.total_max[-1]], c = 'red', s = 100)
        # min point
        plt.scatter([int(self.total_min[0])], [self.total_min[-1]], c = 'lawngreen', s = 100)
        
        plt.xlabel("\nEpoch")
        
        tmp_title = ("[log] " + self.name + " in epoch " + str(int(self.count_epoch))
                    +"\nMAX in epoch " + str(int(self.total_max[0])) + ", " + str(round(self.total_max[-1], 4))
                    +"\nmin in epoch " + str(int(self.total_min[0])) + ", " + str(round(self.total_min[-1], 4))
                    +"\n"
                    )
        plt.title(tmp_title)
        if is_show:
            plt.show()
        
        if is_save:
            if path[-1] != '/':
                path += '/'
            tmp_path = path + "log_graph/" + self.name + "/"
            tmp_path_2 = path + "log_graph/"
            try:
                if not os.path.exists(tmp_path):
                    os.makedirs(tmp_path)
                plt.savefig(tmp_path + self.name + "_" + str(int(1+((self.count_fig_save - 1) % 5))) + ".png", dpi = 200)
                plt.savefig(tmp_path_2 + self.name + ".png", dpi = 200)
            except:
                print(self.print_head, "(exc) log graph save FAIL")
            
            # for COLAB
            '''
            if self.RUN_WHERE == -1:
                if self.PATH_COLAB_OUT_LOG == "False":
                    print(self.print_head, "Warning: save PATH not specified")
                    sys.exit(9)
                
                tmp_path_colab = self.PATH_COLAB_OUT_LOG + "log_graph/"
                try:
                    if not os.path.exists(tmp_path_colab):
                        os.makedirs(tmp_path_colab)
                    plt.savefig(tmp_path_colab + self.name + ".png", dpi = 200)
                except:
                    print(self.print_head, "(exc) log graph save at COLAB FAIL")
            '''
            
        plt.close()


#=== End of RecordBox

#verified on spyder
class RecordBox4IoUs():
    #-------------------------
    # 1. init
    #   box_ious = RecordBox4IoU(name = "iou_name")
    #
    # 2. add item in batch
    #   box_ious.add_item(dict_type_ious)
    #
    # 3. at the end of batch
    #   box_ious.update_batch()
    #
    # 4. at the end of epoch
    #   box_ious.update_epoch() #this draws log graph, can return (str) 'mIoU','IoU-1','IoU-2'...,'IoU-Last'
    #-------------------------
    
    def __init__(self, name='name', labels = 11, is_print = True, print_interval = 1, will_update_graph=True, **kargs):
        # Parameter name
        self.name = name
        self.print_head = "(RecordBox4IoUs) " + self.name + " ->"
        
        #print(self.print_head, "init")
        
        # self.labels = labels      # not used yet
        
        self.is_print = is_print
        
        if print_interval < 1:
            self.print_interval = 1
            print(self.print_head, "print_interval should be >= 1, so it changed to 1")
        else:
            self.print_interval = int(print_interval)
        
        
        #print(self.print_head, "Generated RecordBox4IoU name:", self.name)
        if '/' in self.name:
            print(self.print_head, "(exc) name should not include /")
            sys.exit(9)
        
        if self.is_print:
            print(self.print_head, "Update Batch log will be printed once per every", str(self.print_interval), "updates")
        
        #all ious dict constisted like this: {label_num: (valid_label_count, avg_or_sum_iou_value), ...}
        #item means mIoU
        
        
        self.count_single_item = 0                  # single item number in 1 batch
        self.single_ious_sum = {}                   # single ious sum per batch
        
        
        self.count_batch = 0                        # batch number in 1 epoch
        self.batch_ious_sum = {}                    # (dict with tuples) iou accumulates in 1 batch, (label_count, IoU_sum)
        
        
        self.count_epoch = 0                        # epoch number in total RUN
        self.epoch_ious_sum = {}                    # (dict with tuples) iou accumulates in 1 epoch, (label_count, IoU_sum)
        self.epoch_ious_avg = {}                    # (dict with floats) iou averages in 1 epoch,    (avg_value)
                                                    #    key "mioU" -> mIoU value
        
        
        self.record_epoch_ious = []                 # (list with dicts) all epoch_ious_avg in total RUN
        
        self.count_fig_save = 0                     # (int)     log fig save count
        
        self.total_min = (0,0)                      # (tuple)   (epoch number, min epoch_item)
        self.total_max = (0,0)                      # (tuple)   (epoch number, MAX epoch_item)
        
        self.is_best_max = False                    # (bool)    is lastly updated value is best (max)
        self.is_best_min = False                    # (bool)    is lastly updated value is best (min)
        
        self.will_update_graph = will_update_graph  # (bool)    will update(save) log graph
        if self.will_update_graph:
            print(self.print_head, "This will update graph every epoch.")
        else:
            print(self.print_head, "This will NOT update graph every epoch!")
        
        #--- for colab (RUN_WHERE == -1)
        '''
        try:
            self.RUN_WHERE = kargs['RUN_WHERE']
            self.PATH_COLAB_OUT_LOG = kargs['PATH_COLAB_OUT_LOG']
            if self.PATH_COLAB_OUT_LOG[-1] != '/':
                self.PATH_COLAB_OUT_LOG += '/'
        except:
            self.RUN_WHERE = 1
            self.PATH_COLAB_OUT_LOG = "False"
        '''
        
    #--- functions used outside
    
    # add new item (in batch)
    def add_item(self, dict_ious):
        self.count_single_item += 1         # update to current number of items
        
        for i_key in dict_ious:
            # convert str -> float
            try:
                tmp_float = float(dict_ious[i_key])
            except:
                tmp_float = float("NaN")
            
            # NaN check
            if math.isnan(tmp_float): # this is NaN
                if not i_key in self.batch_ious_sum:
                    # i_key is not in dict
                    self.batch_ious_sum[i_key] = (0, 0.0)       # init (label_count, IoU_sum)
                    
            else: # not a NaN
                if i_key in self.batch_ious_sum:
                    # i_key is in dict
                    prev_data = self.batch_ious_sum[i_key]      # (tuple) (label_count, IoU_sum)
                else:
                    # i_key is NOT in dict
                    prev_data = (0, 0.0)
                
                # update batch_ious_sum
                self.batch_ious_sum[i_key] = (prev_data[0] + 1, prev_data[-1] + tmp_float)
                
    
    
    # update when batch end
    def update_batch(self):
        self.count_batch += 1               # update to current number of batches
        
        if self.count_single_item == 0:
            print(self.print_head, "(exc) self.count_single_item is Zero")
            sys.exit(9)
        
        tmp_str = ""
        tmp_miou_sum = 0.0
        tmp_iou_count = 0
        
        for i_key in self.batch_ious_sum:
            input_data = self.batch_ious_sum[i_key]         # (tuple) (label_count, IoU_sum)
            
            if i_key in self.epoch_ious_sum:
                # i_key is in dict
                prev_data = self.epoch_ious_sum[i_key]      # (tuple) (label_count, IoU_sum)
            else:
                # i_key is NOT in dict
                prev_data = (0, 0.0)
            
            # update epoch_ious_sum
            self.epoch_ious_sum[i_key] = (prev_data[0] + input_data[0], prev_data[-1] + input_data[-1])
            
            # update str for print
            if input_data[0] == 0:
                # all batch_data was NaN -> this label was not in batch_data
                tmp_str += " " + str(i_key) + " : " + "NaN"
            else:
                # input_data[0] != 0
                tmp_iou = round(input_data[-1]/input_data[0], 4)
                tmp_str += " " + str(i_key) + " : " + str(tmp_iou)
                tmp_miou_sum += tmp_iou
                tmp_iou_count += 1
        
        if self.is_print and (self.count_batch - 1) % self.print_interval == 0:
            # verified calcuration
            print(self.print_head, "update batch <", str(self.count_epoch + 1), "-" , str(self.count_batch), ">", tmp_str)
            if tmp_iou_count == 0:
                print("    mIoU : NaN")
            else:
                print("    mIoU :", tmp_miou_sum / tmp_iou_count)
        
        
        #--- reset used para
        self.count_single_item = 0
        self.batch_ious_sum = {}
        
    
    
    # update when epoch end
    def update_epoch(self, is_return = False
                    ,is_show = False, is_save = True, path = "./"
                    ,is_print_sub = False
                    ,is_update_graph = None
                    ):
        self.is_best_max = False            # reset flag
        self.is_best_min = False            # reset flag
        self.count_epoch += 1               # update to current number of epoches
        
        if is_update_graph is None:
            _update_graph = self.will_update_graph
        else:
            _update_graph = is_update_graph
            if _update_graph:
                print(self.print_head, "graph updated.")
            else:
                print(self.print_head, "graph update SKIPPED!")
        
        if self.count_batch == 0:
            print(self.print_head, "(exc) self.count_batch is Zero")
            sys.exit(9)
        
        return_str_ious = ""
        return_str_miou = ""
        
        tmp_str = ""
        tmp_miou_sum = 0.0
        tmp_iou_count = 0
        
        self.epoch_ious_avg = {}
        
        for i_key in self.epoch_ious_sum:
            input_data = self.epoch_ious_sum[i_key]         # (tuple) (label_count, IoU_sum)
            
            if input_data[0] == 0:
                # all epoch_data was NaN -> it means this label was not in epoch_data
                tmp_iou = "NaN"
                tmp_str += " " + str(i_key) + " : " + "NaN"
                return_str_ious += ",NaN"
            else:
                # input_data[0] != 0
                tmp_iou = input_data[-1]/input_data[0]
                tmp_miou_sum += tmp_iou
                tmp_iou_count += 1
                tmp_str += " " + str(i_key) + " : " + str(round(tmp_iou, 4))
                return_str_ious += "," + str(round(tmp_iou, 4))
            
            self.epoch_ious_avg[i_key] = tmp_iou
            
        
        # mIoU per epoch update
        if tmp_iou_count == 0:
            self.epoch_ious_avg["miou"] = -9        # it means calc FAIL
        else:
            tmp_miou = tmp_miou_sum / tmp_iou_count
            self.epoch_ious_avg["miou"] = tmp_miou
            return_str_miou = str(round(tmp_miou, 4))
            # update min & max
            if self.count_epoch == 1:
                self.total_min = (self.count_epoch, tmp_miou)
                self.total_max = (self.count_epoch, tmp_miou)
                self.is_best_max = True
                self.is_best_min = True
            else:
                if self.total_min[-1] >= tmp_miou:
                    self.total_min = (self.count_epoch, tmp_miou)
                    self.is_best_min = True
                
                if self.total_max[-1] <= tmp_miou:
                    self.total_max = (self.count_epoch, tmp_miou)
                    self.is_best_max = True
        
        
        if self.is_print or is_print_sub:
            if _update_graph:
                # verified calcuration
                print(self.print_head, "update epoch <", str(self.count_epoch), ">", tmp_str)
                print("    mIoU :", self.epoch_ious_avg["miou"])
        
        
        
        # verified saving data
        self.record_epoch_ious.append(self.epoch_ious_avg)
        
        # reset used para
        self.count_batch = 0
        self.epoch_ious_sum = {}
        
        
        #--- for update_graph()
        self.is_show = is_show
        self.is_save = is_save
        self.path = path
        
        if _update_graph:
            self.update_graph(is_show = self.is_show, is_save = self.is_save, path = self.path)
        
        if is_return:
            # (str) 'mIoU','IoU-1','IoU-2'...,'IoU-Last'
            return return_str_miou + return_str_ious
    
    
    #--- functions used inside class
    
    # save graph
    def update_graph(self, is_show = False, is_save = True, path = "./"):
        self.count_fig_save += 1
        
        list_x_labels = []
        list_y_miou = []
        for i in range(len(self.record_epoch_ious)):
            list_x_labels.append(i+1)
            list_y_miou.append(self.record_epoch_ious[i]['miou'])
        
        plt.figure(figsize=(10,8))
        # main graph
        plt.plot(list_x_labels, list_y_miou)
        # MAX point
        plt.scatter([int(self.total_max[0])], [self.total_max[-1]], c = 'red', s = 100)
        # min point
        plt.scatter([int(self.total_min[0])], [self.total_min[-1]], c = 'lawngreen', s = 100)
        
        plt.xlabel("\nEpoch")
        
        tmp_title = ("[log] " + self.name + " in epoch " + str(int(self.count_epoch))
                    +"\nMAX in epoch " + str(int(self.total_max[0])) + ", " + str(round(self.total_max[-1], 4))
                    +"\nmin in epoch " + str(int(self.total_min[0])) + ", " + str(round(self.total_min[-1], 4))
                    +"\n"
                    )
        plt.title(tmp_title)
        
        if is_show:
            plt.show()
        
        #-------------------------------------------
        
        if is_save:
            if path[-1] != '/':
                path += '/'
            
            tmp_path = path + "log_graph/" + self.name + "/"
            tmp_path_2 = path + "log_graph/"
            
            try:
                if not os.path.exists(tmp_path):
                    os.makedirs(tmp_path)
                plt.savefig(tmp_path + self.name + "_" + str(int(1+((self.count_fig_save - 1) % 5))) + ".png", dpi = 200)
                plt.savefig(tmp_path_2 + self.name + ".png", dpi = 200)
            except:
                print(self.print_head, "(exc) log graph save FAIL")
            
            # for COLAB
            '''
            if self.RUN_WHERE == -1:
                if self.PATH_COLAB_OUT_LOG == "False":
                    print(self.print_head, "Warning: save PATH not specified")
                    sys.exit(9)
                
                tmp_path_colab = self.PATH_COLAB_OUT_LOG + "log_graph/"
                try:
                    if not os.path.exists(tmp_path_colab):
                        os.makedirs(tmp_path_colab)
                    plt.savefig(tmp_path_colab + self.name + ".png", dpi = 200)
                except:
                    print(self.print_head, "(exc) log graph save at COLAB FAIL")
            '''
            
        plt.close()


#=== End of RecordBox4IoUs




class Custom_Dataset_V6(data.Dataset):
    #=== Used externel functions ===
    # 1. pil_augm_lite
    # 2. pil_2_patch_v6
    # 3. label_2_tensor -> not used
    # 4. csv_2_dict
    # 5. pil_augm_v3
    # 6. ColorJitter_Double
    '''
    dataset_train = Custom_Dataset_V6(# Return: dict key order -> 'file_name'
                                      #                         , 'ts_img_hr_raw', 'ts_img_hr', 'info_augm'
                                      #                         , 'ts_lab_hr_raw', 'ts_lab_hr'
                                      #                         , 'ts_img_lr_raw', 'ts_img_lr', 'info_deg'
                                      # if not generated, output will be None
                                      
                                      name_memo                     = 'train'
                                     ,in_path_dataset               = PATH_BASE_IN
                                     ,in_category                   = NAME_FOLDER_TRAIN
                                     ,in_name_folder_image          = NAME_FOLDER_IMAGES
                                     
                                      #--- options for train 
                                     ,is_train                      = True
                                      # below options can be skipped when above option is False
                                     ,opt_augm_lite                 = False
                                     ,opt_augm_crop_init_range      = HP_AUGM_RANGE_CROP_INIT
                                     ,opt_augm_rotate_max_degree    = HP_AUGM_ROTATION_MAX
                                     ,opt_augm_prob_flip            = HP_AUGM_PROB_FLIP
                                     ,opt_augm_prob_crop            = HP_AUGM_PROB_CROP
                                     ,opt_augm_prob_rotate          = HP_AUGM_PROB_ROTATE
                                     ,opt_augm_cj_brigntess         = HP_CJ_BRIGHTNESS
                                     ,opt_augm_cj_contrast          = HP_CJ_CONTRAST
                                     ,opt_augm_cj_saturation        = HP_CJ_SATURATION
                                     ,opt_augm_cj_hue               = HP_CJ_HUE
                                     ,opt_augm_random_scaler        = [1.0, 1.5, 2.0]           # is_force_fix 의 경우에 적용가능
                                     
                                      #--- options for HR image
                                     ,in_path_alter_hr_image        = "some_path"               # 입력 안하면 초기입력경로의 이미지 사용됨 (default = None)
                                     
                                      #--- options for HR label
                                     ,is_return_label               = True
                                      # below options can be skipped when above option is False
                                     ,in_name_folder_label          = NAME_FOLDER_LABELS
                                     ,label_number_total            = HP_LABEL_TOTAL
                                     ,label_number_void             = HP_LABEL_VOID
                                     ,is_label_dilated              = True
                                     
                                     ,is_label_onehot_encode        = 
                                     
                                     ,is_label_verify               = 
                                      #(선택) if is_label_verify is True
                                     ,label_verify_try_ceiling      = 
                                     ,label_verify_class_min        =
                                     ,label_verify_ratio_max        = 
                                     
                                      #--- options for LR image
                                     ,is_return_image_lr            = True
                                      # below options can be skipped when above option is False
                                     ,scalefactor                   = HP_DG_SCALE_FACTOR
                                     ,in_path_dlc                   = PATH_BASE_IN_SUB
                                     ,in_name_dlc_csv               = HP_DG_CSV_NAME
                                     
                                      #--- increase dataset length
                                     ,in_dataset_loop               = HP_DATASET_LOOP                               #@@@ check required
                                     
                                      #--- options for generate patch or force margin
                                     ,is_patch                      = # (bool) is_force_fix와 동시 사용 불가
                                      # below options can be skipped when above option is False
                                     ,patch_stride                  = HP_???_STRIDES                                #@@@ check required
                                     ,patch_crop_init_range         = HP_???_RANGE_CROP_INIT_COOR                   #@@@ check required
                                     ,model_input_patch_size        = (HP_MODEL_IMG_W, HP_MODEL_IMG_H)              #(Patch 생성에 쓰임)
                                     
                                     ,is_force_fix                  = # (bool) is_patch와 동시 사용 불가, default = False
                                     ,force_fix_size_hr             = # (tuple with int) (w, h)
                                     
                                      #--- optionas for generate tensor
                                     ,transform_img                 =                                               #@@@ check required
                                     )
    '''
    
    def __init__(self, **kargs):
        try:
            self.name_memo = kargs['name_memo']                                     # (str) dataset name memo
        except:
            self.name_memo = ""
        
        self.name_func = "(Custom_Dataset_V6) " + self.name_memo + "-> "
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("")
        print(self.name_func, "init")
        
        self.is_train = kargs['is_train']                                           # (bool) Does this dataset used for train?
        
        #                                                                               this controls data augmentation & patch options
        #                                                                               patch for valid is center cropped
        self.opt_augm_lite                  = False                                 # 편의상 default 값 미리 선언
        if self.is_train:
            try:
                self.opt_augm_lite          = kargs['opt_augm_lite']                # SR 계열에서 사용되는 약한 수준의 data augmentation 시행여부
            except:                                                                 # 아래의 Flip Crop Rotate & color jitter와 동시적용 불가
                self.opt_augm_lite          = False                                 # Label 이미지 사용 시엔 적용 불가능
            
            # Flip Crop Rotate options
            self.opt_augm_crop_init_range   = kargs['opt_augm_crop_init_range']     # (tuple with int) (min, max) range for single axis coor
            #                                                                               same range aplied for W & H axis
            self.opt_augm_rotate_max_degree = kargs['opt_augm_rotate_max_degree']   # (int) max degree for rotation
            self.opt_augm_prob_flip         = kargs['opt_augm_prob_flip']           # (int) probability of apply flip
            self.opt_augm_prob_crop         = kargs['opt_augm_prob_crop']           # (int) probability of apply crop
            self.opt_augm_prob_rotate       = kargs['opt_augm_prob_rotate']         # (int) probability of apply roration
            
            # color jitter options
            self.opt_augm_cj_brigntess      = kargs['opt_augm_cj_brigntess']        # (tuple with float) (min, max) value
            self.opt_augm_cj_contrast       = kargs['opt_augm_cj_contrast']         # (tuple with float) (min, max) value
            self.opt_augm_cj_saturation     = kargs['opt_augm_cj_saturation']       # (tuple with float) (min, max) value
            self.opt_augm_cj_hue            = kargs['opt_augm_cj_hue']              # (tuple with float) (min, max) value
            
            self.transform_cj = ColorJitter_Double(brightness = self.opt_augm_cj_brigntess
                                                  ,contrast   = self.opt_augm_cj_contrast
                                                  ,saturation = self.opt_augm_cj_saturation
                                                  ,hue        = self.opt_augm_cj_hue
                                                  )
            
            try:
                self.opt_augm_random_scaler = kargs['opt_augm_random_scaler']       # (list with float) 사용할 scale 배율 목록
            except:
                self.opt_augm_random_scaler = [1.0]
            
        
        self.is_return_label =    kargs['is_return_label']                          # (bool) Does Dataset returns High Resolution label?
        
        self.is_return_image_lr = kargs['is_return_image_lr']                       # (bool) Does Dataset returns Low Resolution image?
        
        if self.opt_augm_lite and self.is_return_label:
            _str = self.name_func + "라벨 데이터가 사용되는 경우, opt_augm_lite 옵션 사용 불가능합니다."
            sys.exit(_str)
        
        
        if self.is_return_image_lr:
            self.scalefactor = int(kargs['scalefactor'])                            # (int) Scale Factor
        else:                                                                       #       only available when LR imaage is returned
            self.scalefactor = 1
            print(self.name_func, "scalefactor set to", str(self.scalefactor))
        
        self.in_path_dataset          = kargs['in_path_dataset']                    # (str) "D:/...Camvid" ┐
        #                                                                                                  │
        self.in_category              = kargs['in_category']                        # (str)                └─"train"┐ or "val" or "test"
        #                                                                                                           │
        self.in_name_folder_image     = kargs['in_name_folder_image']               # (str)                         └─"images"
        #                                                                                                                   img_1.png
        try:                                                                        #
            self.in_path_alter_hr_image = kargs['in_path_alter_hr_image']           # (str) HR Image 를 대체할 폴더의 전체 경로를 입력하세요
            if self.in_path_alter_hr_image is not None:                             #       해당 폴더 내 이미지의 이름은 원본과 동일해야 하며
                if self.in_path_alter_hr_image[-1] != '/':                          #       train / val / test 구분은
                    self.in_path_alter_hr_image += '/'                              #       원본 폴더의 구성을 따라감에 주의하세요
                warnings.warn("HR Image 교체됨")
                print(self.name_func, "HR Image from:", self.in_path_alter_hr_image)
            else:
                warnings.warn("HR Image 원본 사용됨")
        except:
            self.in_path_alter_hr_image = None
            warnings.warn("HR Image 원본 사용됨")
            
        
        #                                                                                   "D:/...Camvid" ┐
        #                                                                                                  │
        #                                                                                                  └─"train"┐ or "val" or "test"
        if self.is_return_label:                                                    #                               │       
            self.in_name_folder_label = kargs['in_name_folder_label']               # (str)                         └─"labels"
            #                                                                                                               lab_1.png
            # label_2_tensor options
            self.label_number_total = kargs['label_number_total']                   # (int) total label numbers (include void)
            self.label_number_void  = kargs['label_number_void']                    # (int) void label number
            self.is_label_dilated   = kargs['is_label_dilated']                     # (bool) does label smoothing applied ?
            
            self.is_label_onehot_encode = kargs['is_label_onehot_encode']           # (bool) label one-hot encoding 시행여부
            
            if self.is_label_dilated and not self.is_label_onehot_encode:
                _str = self.name_func + "If label is dilated, it must be in one-hot form."
                sys.exit(_str)
            
            try:
                self.is_label_verify = kargs['is_label_verify']                     # (bool) label verification (pil_marginer_v3)
            except:
                self.is_label_verify = False
            
            if self.is_label_verify:
                self.label_verify_try_ceiling = kargs['label_verify_try_ceiling']   # (int) Crop Retry 최대 횟수
                self.label_verify_class_min   = kargs['label_verify_class_min']     # (int) void 포함 최소 class 종류
                self.label_verify_ratio_max   = kargs['label_verify_ratio_max']     # (float) class 비율값의 상한값 (0 ~ 1.0)
                _str = self.name_func + "Label 검증 시행됨 (void " + str(self.label_number_void) + "번 포함 최소 class 종류: " + str(self.label_verify_class_min) + ", class 비율값 상한: " + str(self.label_verify_ratio_max) + ")"
                warnings.warn(_str)
            else:
                self.label_verify_try_ceiling = None
                self.label_verify_class_min   = None
                self.label_verify_ratio_max   = None
                _str = self.name_func + "Label 검증 시행 안함"
                warnings.warn(_str)
            
        
        if self.is_return_image_lr:
            self.in_path_dlc = kargs['in_path_dlc']                                 # (str) "D:/...Camvid_DLC/option"┐
            #                                                                                                        │
            # use self.in_name_folder_image again                                                                    └─"images"
            #                                                                                                               img_1.png
            try:
                self.in_name_dlc_csv = kargs['in_name_dlc_csv']                     # (str)                             deg_opt.csv
                #                                                                           Degradation info csv file
            except:
                self.in_name_dlc_csv = None
        
        try:
            self.in_dataset_loop = kargs['in_dataset_loop']                         # (int) dataset length multiple
        except:
            self.in_dataset_loop = 1
        
        # pil_2_patch options
        self.is_patch                  = kargs['is_patch']                          # (bool) Does model input is patch?
        #                                                                               -> Patch기 본래 이미지보다 작은 경우에만 사용 가능
        #                                                                               -> 그 이외에는 is_force_fix 옵션으로 margin 붙이기
        if self.is_patch:
            if self.is_train:
                self.patch_stride          = kargs['patch_stride']                  # (tuple with int) (w, h) stride for each axis
                self.patch_crop_init_range = kargs['patch_crop_init_range']         # (tuple with int) (min, max) range for single axis coor
            else:
                #below options will not used
                self.patch_stride          = (1,1)
                self.patch_crop_init_range = (0,1)
            
            # 변수 명 수정됨: model_input_size -> model_input_patch_size
            tmp_w, tmp_h = kargs['model_input_patch_size']                          # (tuple) model input patch size (Width, Heights)
            self.model_input_patch_size = (int(tmp_w), int(tmp_h))                  #   -> if patch used, it will be input patch size
            
            self.is_force_fix = False
        else:
            try:
                self.is_force_fix = kargs['is_force_fix']                           # (bool) (train, val) 최종 이미지의 강제 margin 추가여부
            except:
                self.is_force_fix = False
            
            if self.is_force_fix:
                _w, _h = kargs['force_fix_size_hr']                                 # (tuple with int) Margin 포함 HR 이미지 or 라벨 크기
                self.force_fix_size_hr = (int(_w), int(_h))
                
                if self.is_return_image_lr:
                    self.force_fix_size_lr = (int(_w//self.scalefactor), int(_h)//self.scalefactor)     # Margin 포함 LR 이미지 크기
                
        
        # transform to tensor funtion
        self.transform_img = kargs['transform_img']                             # (transforms.Compose) transforms to tensor
        #                                                                                   used for HR & LR image
        self.transform_raw = transforms.Compose([transforms.ToTensor()])        # (transforms.Compose) with no Norm
        
        #--- path input fix -> make file list
        
        if self.in_path_dataset[-1] != '/':
            self.in_path_dataset += '/'
        
        
        if self.in_category[0] == '/':
            self.in_category = self.in_category[1:]
        
        if self.in_category[-1] != '/':
            self.in_category += '/'
        
        
        # HR image
        
        if self.in_name_folder_image[0] == '/':
            self.in_name_folder_image = self.in_name_folder_image[1:]
        
        if self.in_name_folder_image[-1] == '/':
            self.in_name_folder_image = self.in_name_folder_image[:-1]
        
        self.list_file_img = os.listdir(self.in_path_dataset + self.in_category + self.in_name_folder_image)
        
        # HR label
        if self.is_return_label:
            if self.in_name_folder_label[0] == '/':
                self.in_name_folder_label = self.in_name_folder_label[1:]
            
            if self.in_name_folder_label[-1] == '/':
                self.in_name_folder_label = self.in_name_folder_label[:-1]
            
            self.list_file_lab = os.listdir(self.in_path_dataset + self.in_category + self.in_name_folder_label) # (list with str) file names
            
            if len(self.list_file_img) != len(self.list_file_lab):
                print(self.name_func, "파일 수가 일치하지 않습니다:", len(self.list_file_img), len(self.list_file_lab))
                sys.exit(9)
        
        # LR image
        if self.is_return_image_lr:
            self.in_path_dlc = kargs['in_path_dlc']
            if self.in_path_dlc[-1] != '/':
                self.in_path_dlc += '/'
            
            self.list_file_img_lr = os.listdir(self.in_path_dlc + self.in_name_folder_image)
        
        
        #--- make dict with pil
        
        self.dict_pil_raw_img_hr = {}                                               # (dict with pil) raw HR image
        
        if self.is_return_label:
            self.dict_pil_raw_lab_hr = {}                                           # (dict with pil) raw HR label
        
        if self.is_return_image_lr:
            self.dict_pil_raw_img_lr = {}                                           # (dict with pil) raw LR image
            self.dict_info_degrade = {}                                             # (dict with str) LR image degradation info
        
            try:
                dict_dlc_img_info = csv_2_dict(path_csv = self.in_path_dlc + self.in_name_dlc_csv)  # key = (str) img file name
                #                                                                                     item = (list with str) infos
                self.is_dlc_img_info_loaded = True
            except:
                self.is_dlc_img_info_loaded = False
        
        tmp_len = len(self.list_file_img)
        for i_file in range(len(self.list_file_img)):
            tmp_str = self.name_func + "Loading Main ... " + str(i_file + 1) + " / " + str(tmp_len)
            print("", end = '\r')
            print(tmp_str, end = '')
            
            # HR image
            if self.in_path_alter_hr_image is None:
                # 원본 폴더 내용물 사용
                path_img_hr = self.in_path_dataset + self.in_category + self.in_name_folder_image + '/' + self.list_file_img[i_file]
            else:
                # 대체 폴더 내용물 사용
                path_img_hr = self.in_path_alter_hr_image + self.list_file_img[i_file]
            
            pil_img_hr_raw = Image.open(path_img_hr)                                # (pil) HR image
            
            self.dict_pil_raw_img_hr[self.list_file_img[i_file]] = pil_img_hr_raw
            
            # HR label
            if self.is_return_label:
                path_lab_hr = self.in_path_dataset + self.in_category + self.in_name_folder_label + '/' + self.list_file_lab[i_file]
                pil_lab_hr_raw = Image.open(path_lab_hr)                            # (pil) HR label
                self.dict_pil_raw_lab_hr[self.list_file_lab[i_file]] = pil_lab_hr_raw
            
        print("")
        
        #LR image
        if self.is_return_image_lr:
            tmp_len = len(self.list_file_img_lr)
            _count_loaded_image = 0     # 불러온 이미지 수 확인용
            
            for i_file in range(len(self.list_file_img_lr)):
                tmp_str = self.name_func + "Loading DLC ... " + str(i_file + 1) + " / " + str(tmp_len)
                print("", end = '\r')
                print(tmp_str, end = '')
                
                # 해당 data 묶음에 매칭되는 HR 이미지가 존재하는 경우에만 파일 불러오기
                if not self.list_file_img_lr[i_file] in self.list_file_img:
                    continue
                
                _count_loaded_image += 1
                
                path_img_lr = self.in_path_dlc + self.in_name_folder_image + '/' + self.list_file_img_lr[i_file]
                pil_img_lr_raw = Image.open(path_img_lr)                            # (pil) LR image
                self.dict_pil_raw_img_lr[self.list_file_img_lr[i_file]] = pil_img_lr_raw
                if self.is_dlc_img_info_loaded:
                    if self.in_name_dlc_csv in ["degradation_2.csv", "degradation_MiniCity.csv"]:
                        list_dg_csv = dict_dlc_img_info[self.list_file_img_lr[i_file]]
                        str_dg_option  = "Blur = Gaussian, Downscale(" + self.in_path_dlc.split('/')[-2] + ")"
                        str_dg_option += ", Noise = (Gaussian, " + list_dg_csv[0] + ", mu = 0, Sigma = " + list_dg_csv[1] + ")"
                    else:
                        str_dg_option = "Info File not Supported"
                    self.dict_info_degrade[self.list_file_img_lr[i_file]] = str_dg_option
                else:
                    self.dict_info_degrade[self.list_file_img_lr[i_file]] = "Info not loaded"
            
            _str = " -> Loaded images: " + str(_count_loaded_image)
            print(_str)
        
        self.list_file_img = self.list_file_img * int(self.in_dataset_loop)
        
        if self.is_return_label:
            self.list_file_lab = self.list_file_lab * int(self.in_dataset_loop)
    
    
    
    def __len__(self):
        return len(self.list_file_img)    # self.list_file_name : (list with str) name of files in "images" folder
    
    
    
    def __getitem__(self, idx):
        
        # HR image
        pil_img_hr_raw = self.dict_pil_raw_img_hr[self.list_file_img[idx]]
        
        # HR label
        if self.is_return_label:
            pil_lab_hr_raw = self.dict_pil_raw_lab_hr[self.list_file_lab[idx]]
        
        # LR image
        if self.is_return_image_lr:
            pil_img_lr_raw = self.dict_pil_raw_img_lr[self.list_file_img[idx]]
            str_dg_option = "(Pre-Degraded) " + self.dict_info_degrade[self.list_file_img[idx]]
        
        #--- data augm for train
        if self.is_train:
            # (1/2) Flip Crop Rotate
            if self.is_return_image_lr and self.is_return_label:
                # HR image, LR image, HR Label
                pil_img_hr_raw, pil_img_lr_raw, pil_lab_hr_raw, str_augm_option = pil_augm_v3(in_pil_x             = pil_img_hr_raw
                                                                                             ,in_pil_x_lr          = pil_img_lr_raw
                                                                                             ,in_option_resize_x   = Image.LANCZOS
                                                                                             ,in_option_rotate_x   = Image.BICUBIC
                                                                                             ,in_pil_y             = pil_lab_hr_raw
                                                                                             ,in_option_resize_y   = Image.NEAREST
                                                                                             ,in_option_rotate_y   = Image.NEAREST
                                                                                             ,in_crop_wh_min       = self.opt_augm_crop_init_range[0]
                                                                                             ,in_crop_wh_max       = self.opt_augm_crop_init_range[-1]
                                                                                             ,in_rotate_degree_max = self.opt_augm_rotate_max_degree
                                                                                             ,in_percent_flip      = self.opt_augm_prob_flip
                                                                                             ,in_percent_crop      = self.opt_augm_prob_crop
                                                                                             ,in_percent_rotate    = self.opt_augm_prob_rotate
                                                                                             ,is_return_options    = True
                                                                                             )
            
            elif self.is_return_image_lr:
                # HR image, LR image
                if not self.opt_augm_lite:
                    pil_img_hr_raw, pil_img_lr_raw, str_augm_option = pil_augm_v3(in_pil_x             = pil_img_hr_raw
                                                                                 ,in_pil_x_lr          = pil_img_lr_raw
                                                                                 ,in_option_resize_x   = Image.LANCZOS
                                                                                 ,in_option_rotate_x   = Image.BICUBIC
                                                                                 ,in_crop_wh_min       = self.opt_augm_crop_init_range[0]
                                                                                 ,in_crop_wh_max       = self.opt_augm_crop_init_range[-1]
                                                                                 ,in_rotate_degree_max = self.opt_augm_rotate_max_degree
                                                                                 ,in_percent_flip      = self.opt_augm_prob_flip
                                                                                 ,in_percent_crop      = self.opt_augm_prob_crop
                                                                                 ,in_percent_rotate    = self.opt_augm_prob_rotate
                                                                                 ,is_return_options    = True
                                                                                 )
            
            elif self.is_return_label:
                # HR image, HR Label
                pil_img_hr_raw, pil_lab_hr_raw, str_augm_option = pil_augm_v3(in_pil_x             = pil_img_hr_raw
                                                                             ,in_option_resize_x   = Image.LANCZOS
                                                                             ,in_option_rotate_x   = Image.BICUBIC
                                                                             ,in_pil_y             = pil_lab_hr_raw
                                                                             ,in_option_resize_y   = Image.NEAREST
                                                                             ,in_option_rotate_y   = Image.NEAREST
                                                                             ,in_crop_wh_min       = self.opt_augm_crop_init_range[0]
                                                                             ,in_crop_wh_max       = self.opt_augm_crop_init_range[-1]
                                                                             ,in_rotate_degree_max = self.opt_augm_rotate_max_degree
                                                                             ,in_percent_flip      = self.opt_augm_prob_flip
                                                                             ,in_percent_crop      = self.opt_augm_prob_crop
                                                                             ,in_percent_rotate    = self.opt_augm_prob_rotate
                                                                             ,is_return_options    = True
                                                                             )
            '''
            else:
                # HR image -> 쓸데 없음
                pil_img_hr_raw, str_augm_option = pil_augm_v3(in_pil_x             = pil_img_hr_raw
                                                             ,in_option_resize_x   = Image.LANCZOS
                                                             ,in_option_rotate_x   = Image.BICUBIC
                                                             ,in_crop_wh_min       = self.opt_augm_crop_init_range[0]
                                                             ,in_crop_wh_max       = self.opt_augm_crop_init_range[-1]
                                                             ,in_rotate_degree_max = self.opt_augm_rotate_max_degree
                                                             ,in_percent_flip      = self.opt_augm_prob_flip
                                                             ,in_percent_crop      = self.opt_augm_prob_crop
                                                             ,in_percent_rotate    = self.opt_augm_prob_rotate
                                                             ,is_return_options    = True
                                                             )
            '''
            
            # (2/2) color jitter
            if self.is_return_image_lr:
                pil_img_hr_raw, pil_img_lr_raw = self.transform_cj(pil_img_hr_raw, pil_img_lr_raw)
            else:
                pil_img_hr_raw = self.transform_cj(pil_img_hr_raw)
            
            
        
        #--- Gen Patch for model input
        if self.is_patch:
            crop_init_coor_w = int(random.uniform(self.patch_crop_init_range[0], self.patch_crop_init_range[-1]))
            crop_init_coor_h = int(random.uniform(self.patch_crop_init_range[0], self.patch_crop_init_range[-1]))
            if self.is_return_label and self.is_return_image_lr:
                # when HR image and both HR label & LR image patch needed 
                # output dict key = (int) 0 ~ (number of patches -1)
                #             item = pil image
                d_p_hr_img, d_p_hr_lab, d_p_lr_img = pil_2_patch_v6(in_pil_hr       = pil_img_hr_raw
                                                                   ,in_pil_hr_label = pil_lab_hr_raw
                                                                   ,in_pil_lr       = pil_img_lr_raw
                                                                    
                                                                   ,in_scale_factor = self.scalefactor
                                                                   
                                                                   ,is_val          = not self.is_train
                                                                    # val 모드의 경우, 일부 설정 무시 후 center crop 1장만 생성
                                                                   ,batch_size      = 1
                                                                   ,strides         = self.patch_stride
                                                                   ,patch_size      = self.model_input_patch_size
                                                                   ,crop_init_coor  = (crop_init_coor_w, crop_init_coor_h)
                                                                   )
                
            elif self.is_return_label:
                # when HR image and HR label patch needed 
                d_p_hr_img, d_p_hr_lab = pil_2_patch_v6(in_pil_hr       = pil_img_hr_raw
                                                       ,in_pil_hr_label = pil_lab_hr_raw
                                                        
                                                       ,in_scale_factor = 1                 # 어차피 내부적으로 1로 처리됨
                                                       
                                                       ,is_val          = not self.is_train
                                                        # val 모드의 경우, 일부 설정 무시 후 center crop 1장만 생성
                                                       ,batch_size      = 1
                                                       ,strides         = self.patch_stride
                                                       ,patch_size      = self.model_input_patch_size
                                                       ,crop_init_coor  = (crop_init_coor_w, crop_init_coor_h)
                                                       )
            elif self.is_return_image_lr:
                # when HR image and LR image patch needed 
                d_p_hr_img, d_p_lr_img = pil_2_patch_v6(in_pil_hr       = pil_img_hr_raw
                                                       ,in_pil_lr       = pil_img_lr_raw
                                                        
                                                       ,in_scale_factor = self.scalefactor
                                                       
                                                       ,is_val          = not self.is_train
                                                        # val 모드의 경우, 일부 설정 무시 후 center crop 1장만 생성
                                                       ,batch_size      = 1
                                                       ,strides         = self.patch_stride
                                                       ,patch_size      = self.model_input_patch_size
                                                       ,crop_init_coor  = (crop_init_coor_w, crop_init_coor_h)
                                                       )
            
            for i_key in d_p_hr_img: # i_key is 0 ~ 0 (single value)
                
                #self.list_pil_img_hr.append(d_p_hr_img[i_key]) 
                out_pil_img_hr = d_p_hr_img[i_key]                                              # (pil) HR images
                
                if self.is_return_label:
                    #self.list_pil_lab_hr.append(d_p_hr_lab[i_key])              
                    out_pil_lab_hr = d_p_hr_lab[i_key]                                          # (pil) HR labels
                
                if self.is_return_image_lr:
                    #self.list_pil_img_lr.append(d_p_lr_img[i_key])              
                    out_pil_img_lr = d_p_lr_img[i_key]                                          # (pil) LR images
                    
                break #use only single patch
        
        else:
            # not a patch
            if self.is_force_fix:
                # 강제 margin 추가 -> 세부 옵션은 CamVid 11라벨 기준
                
                if self.is_train:
                    r_s_f = float(np.random.choice(self.opt_augm_random_scaler, 1)[0])  # random scale factor
                else:
                    r_s_f = float(1.0)
                
                if self.is_return_label and self.is_return_image_lr:
                    out_pil_img_hr, out_pil_lab_hr, out_pil_img_lr = pil_marginer_v3(in_pil_hr          = pil_img_hr_raw
                                                                                    ,target_size_hr     = self.force_fix_size_hr
                                                                                    ,img_background     = (0, 0, 0)
                                                                                    # (선택) 세부옵션 (각각 default 값 있음)
                                                                                    ,scaler             = r_s_f
                                                                                    ,is_random          = self.is_train
                                                                                    ,itp_opt_img        = Image.LANCZOS
                                                                                    ,itp_opt_lab        = Image.NEAREST
                                                                                    
                                                                                     # 선택 (HR Label 관련)
                                                                                    ,in_pil_hr_label    = pil_lab_hr_raw
                                                                                    ,lab_total          = self.label_number_total
                                                                                    ,lab_background     = self.label_number_void
                                                                                    ,is_lab_verify      = self.is_label_verify
                                                                                    # 선택 - 선택 (Label 검증 관련, is_lab_verify=True에만 적용)
                                                                                    ,lab_try_ceiling    = self.label_verify_try_ceiling
                                                                                    ,lab_class_min      = self.label_verify_class_min
                                                                                    ,lab_ratio_max      = self.label_verify_ratio_max
                                                                                    
                                                                                     # 선택 (LR Image 관련)
                                                                                    ,in_pil_lr          = pil_img_lr_raw
                                                                                    ,in_scale_factor    = self.scalefactor
                                                                                    ,target_size_lr     = self.force_fix_size_lr
                                                                                    )
                    
                elif self.is_return_label:
                    out_pil_img_hr, out_pil_lab_hr = pil_marginer_v3(in_pil_hr          = pil_img_hr_raw
                                                                    ,target_size_hr     = self.force_fix_size_hr
                                                                    ,img_background     = (0, 0, 0)
                                                                    # (선택) 세부옵션 (각각 default 값 있음)
                                                                    ,scaler             = r_s_f
                                                                    ,is_random          = self.is_train
                                                                    ,itp_opt_img        = Image.LANCZOS
                                                                    ,itp_opt_lab        = Image.NEAREST
                                                                     # 선택 (HR Label 관련)
                                                                    ,in_pil_hr_label    = pil_lab_hr_raw
                                                                    ,lab_total          = self.label_number_total
                                                                    ,lab_background     = self.label_number_void
                                                                    ,is_lab_verify      = self.is_label_verify
                                                                    # 선택 - 선택 (Label 검증 관련, is_lab_verify=True에만 적용)
                                                                    ,lab_try_ceiling    = self.label_verify_try_ceiling
                                                                    ,lab_class_min      = self.label_verify_class_min
                                                                    ,lab_ratio_max      = self.label_verify_ratio_max
                                                                    )
                    
                elif self.is_return_image_lr:
                    out_pil_img_hr, out_pil_img_lr = pil_marginer_v3(in_pil_hr          = pil_img_hr_raw
                                                                    ,target_size_hr     = self.force_fix_size_hr
                                                                    ,img_background     = (0, 0, 0)
                                                                    # (선택) 세부옵션 (각각 default 값 있음)
                                                                    ,scaler             = r_s_f
                                                                    ,is_random          = self.is_train
                                                                    ,itp_opt_img        = Image.LANCZOS
                                                                    ,itp_opt_lab        = Image.NEAREST
                                                                     # 선택 (LR Image 관련)
                                                                    ,in_pil_lr          = pil_img_lr_raw
                                                                    ,in_scale_factor    = self.scalefactor
                                                                    ,target_size_lr     = self.force_fix_size_lr
                                                                    )
                
            else:
                # 이미지 그대로 return
                out_pil_img_hr = pil_img_hr_raw
                
                if self.is_return_label:
                    out_pil_lab_hr = pil_lab_hr_raw
                
                if self.is_return_image_lr:
                    out_pil_img_lr = pil_img_lr_raw
        
        
        if self.is_train and self.opt_augm_lite:
            out_pil_img_hr, out_pil_img_lr, str_augm_option = pil_augm_lite(out_pil_img_hr, out_pil_img_lr, get_info=True)
        
        #--- ready for return
        
        dict_box = {}
        
        dict_box['file_name']     = self.list_file_img[idx]                                 # (str) file name
        dict_box['ts_img_hr_raw'] = self.transform_raw(out_pil_img_hr)                      # (pil) HR Image - Full or Patch for model
        dict_box['ts_img_hr']     = self.transform_img(out_pil_img_hr)                      # (ts) HR image - Full or Patch for model
        
        if self.is_train:
            dict_box['info_augm'] = "(Augmentation) " + str_augm_option + " / RSF: x" + str(r_s_f)      # (str) Augmentation option info
            
        else:
            dict_box['info_augm'] = "(Augmentation not applied)"
        
        if self.is_return_label:
            dict_box['ts_lab_hr_raw'] = self.transform_raw(out_pil_lab_hr)                  # (pil) HR Label - Full or Patch for model
            #<<< from label_2_tensor
            flag_init_label_gen = 0
            
            label_total = self.label_number_total
            label_void  = self.label_number_void
            is_dilated  = self.is_label_dilated
            is_onehot   = self.is_label_onehot_encode
            
            out_tensor = None
            
            if is_onehot:
                # one-hot encode
                in_np = np.array(out_pil_lab_hr)
                if is_dilated:
                    cv_kernel_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                    for i_label in range(label_total):
                        if i_label == label_void:
                            #void 라벨 번호는 텐서변환 생략
                            continue
                        np_label_single = np.where(in_np == i_label, 1, 0).astype(np.uint8)
                        #첫 라벨 생성
                        if out_tensor is None:
                            pil_dilated = Image.fromarray(cv2.dilate(np_label_single, cv_kernel_dilation), mode="L")
                            out_tensor = self.transform_raw(pil_dilated)
                        #2번쨰 라벨부터 ~
                        else:
                            pil_dilated = Image.fromarray(cv2.dilate(np_label_single, cv_kernel_dilation), mode="L")
                            out_tensor = torch.cat([out_tensor, self.transform_raw(pil_dilated)], dim = 0)
                            
                else:
                    # not dilated
                    for i_label in range(label_total):
                        if i_label == label_void:
                            #void 라벨 번호는 텐서변환 생략
                            continue
                        np_label_single = np.where(in_np == i_label, 1, 0).astype(np.uint8)
                        if out_tensor is None:
                            out_tensor = self.transform_raw(np_label_single)
                        else:
                            out_tensor = torch.cat([out_tensor, self.transform_raw(np_label_single)], dim = 0)
                    
                    
                    '''
                    _tensor = torch.Tensor(in_np)
                    _tensor = _tensor.to(self.device).type(torch.int64)
                    _tensor_onehot_raw = torch.nn.functional.one_hot(_tensor, num_classes=self.label_number_total).permute(2,0,1)
                    _tensor_onehot_raw = _tensor_onehot_raw.cpu()
                    if self.label_number_void == self.label_number_total - 1:
                        out_tensor = _tensor_onehot_raw[:-1,:,:].to(torch.float32)
                    elif self.label_number_void == 0:
                        out_tensor = _tensor_onehot_raw[1:,:,:].to(torch.float32)
                    else:
                        _str = "양 끝 번호가 아닌 void는 지원하지 않습니다."
                        sys.exit(_str)
                    '''
                
                
                dict_box['ts_lab_hr'] = out_tensor                                          # (ts) HR labels - may Resized for model
                
                np_label_single = np.where(in_np == label_void, 1, 0)  # void 영역을 0으로, 나머지를 1로
                pil_onehot = Image.fromarray(np_label_single, mode="L")
                _ts_onehot = self.transform_raw(pil_onehot)
                out_tensor = None
                
                for i_label in range(label_total):
                    if i_label == label_void:
                        continue
                    #첫 라벨 생성
                    if out_tensor is None:
                        out_tensor = _ts_onehot
                    #2번쨰 라벨부터 ~
                    else:
                        out_tensor = torch.cat([out_tensor, _ts_onehot], dim = 0)
                
                dict_box['ts_lab_hr_void'] = out_tensor                                     # (ts) HR labels - void channels for subtract
            else:
                # no one-hot encode
                dict_box['ts_lab_hr']      = False
                dict_box['ts_lab_hr_void'] = False
            #>>> from label_2_tensor
        else:
            dict_box['ts_lab_hr_raw']  = False
            dict_box['ts_lab_hr']      = False
            dict_box['ts_lab_hr_void'] = False
        
        
        if self.is_return_image_lr:
            dict_box['ts_img_lr_raw'] = self.transform_raw(out_pil_img_lr)              # (pil) LR Image - Full or Patch for model (Raw)
            dict_box['ts_img_lr']     = self.transform_img(out_pil_img_lr)              # (ts)  LR image - Full or Patch for model (normalized)
            dict_box['info_deg']      = str_dg_option                                   # (str) Degradation option info
        else:
            dict_box['ts_img_lr_raw'] = False
            dict_box['ts_img_lr']     = False
            dict_box['info_deg']      = False
        
        
        
        # 사용되지 않은 값들은 None으로 배정됨
        return dict_box['file_name'] \
             , dict_box['ts_img_hr_raw'], dict_box['ts_img_hr'], dict_box['info_augm'] \
             , dict_box['ts_lab_hr_raw'], dict_box['ts_lab_hr'], dict_box['ts_lab_hr_void'] \
             , dict_box['ts_img_lr_raw'], dict_box['ts_img_lr'], dict_box['info_deg']


#=== End of Custom_Dataset_V6


"""
#        in_path_dataset: (str)  dataset 폴더 경로
#        in_category:     (str)  train / val / test 중 한가지 (데이터셋 하위폴더명)
# (선택)  in_dataset_loop: (int)  데이터셋 반복 횟수 (입력 배율로 list 길이 늘려줌, default = 1)
# (선택)  is_name_same:    (bool) 파일이름 동일여부 (default = False)
#        batch_size:      (int)  dataloader batch 크기
#        shuffle:         (bool) dataloader shuffle

#한번에 dataloader를 생성
def custom_dataloader(**kargs):
    #<<< @@@
    
    name_func = "[custom_dataloader] ->"
    
    #--- (1/3) func: make_list_img_n_lab
    
    #(str) 데이터셋 상위 경로 (PATH_BASE_IN: ./"name_dataset")
    in_path_dataset = kargs['in_path_dataset']
    
    if not in_path_dataset[-1] == "/":
        in_path_dataset += "/"
    
    #(str) 데이터 종류 ("train" or "val" or "test")
    in_category = kargs['in_category']
    if not in_category[-1] == "/":
        in_category += "/"
    
    try:
        #(int) 데이터셋 반복 횟수
        in_dataset_loop = int(kargs['in_dataset_loop'])
    except:
        in_dataset_loop = 1
    
    try:
        #(bool) 파일명 동일 여부
        is_name_same = kargs['is_name_same']
    except:
        is_name_same = False
    #---
    #image, label 이미지 파일의 폴더 이름
    in_name_folder_image = kargs['in_name_folder_image']
    in_name_folder_label = kargs['in_name_folder_label']
    
    list_file_img = os.listdir(in_path_dataset + in_category + in_name_folder_image)
    list_file_lab = os.listdir(in_path_dataset + in_category + in_name_folder_label)
    
    print("in list img: ", list_file_img[0], len(list_file_img))
    print("in list lab: ", list_file_lab[0], len(list_file_lab))
    
    if len(list_file_img) != len(list_file_lab):
        print(name_func, "파일 수가 일치하지 않습니다:", len(list_file_img), len(list_file_lab))
        sys.exit(1)
    
    out_list_path_file_img = []
    out_list_path_file_lab = []
    
    for i_file in range(len(list_file_img)):
        if is_name_same:
            if not np.array_equal(np.array(list_file_img), np.array(list_file_lab)):
                print(name_func, "입력과 라벨이 일치하지 않습니다.")
                sys.exit(1)
        
        out_list_path_file_img.append(in_path_dataset + in_category + in_name_folder_image + "/" + list_file_img[i_file])
        out_list_path_file_lab.append(in_path_dataset + in_category + in_name_folder_label + "/" + list_file_lab[i_file])
    
    if in_dataset_loop == 1:
        return_list_img = sorted(out_list_path_file_img)
        return_list_lab = sorted(out_list_path_file_lab)

    else:
        sorted_list_img = sorted(out_list_path_file_img)
        sorted_list_lab = sorted(out_list_path_file_lab)
        
        return_list_img = []
        return_list_lab = []
        
        for i_loop in range(in_dataset_loop):
            return_list_img += sorted_list_img
            return_list_lab += sorted_list_lab
    
    print("returned list img: ", return_list_img[0], len(return_list_img))
    print("returned list lab: ", return_list_lab[0], len(return_list_lab))
    
    #return return_list_img, return_list_lab
    
    #--- (2/3) class: custom_dataset_img_n_lab
    
    #커스텀 데이터셋 클래스 (이름 변경됨: custom_dataset -> custom_dataset_img_n_lab)
    #in(2): (list) img & lab 파일경로 / out(1): 데이터셋 변수 (x= img 경로, y= lab 경로)
    class custom_dataset_img_n_lab(torch.utils.data.Dataset):
        def __init__(self, image_paths, label_paths):
            self.image_paths = image_paths
            self.label_paths = label_paths
            
        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            filepath_image = self.image_paths[idx]
            filepath_label = self.label_paths[idx]

            return filepath_image, filepath_label
    
    return_dataset = custom_dataset_img_n_lab(return_list_img, return_list_lab)
    
    #--- (3/3) import: torch.utils.data.DataLoader
    
    #(int) dataloader batch size
    dataloader_batch_size = kargs['batch_size']
    #(bool) dataloader shuffle 여부
    dataloader_shuffle = kargs['shuffle']
    
    return torch.utils.data.DataLoader(return_dataset
                                      ,batch_size = dataloader_batch_size
                                      ,shuffle = dataloader_shuffle
                                      )
    #>>> @@@

"""
#=== End of custom_dataloader

def load_pils_2_dict(**kargs):
    '''
    dict_loaded_pils = load_pils_2_dict(#경로 내 pil 이미지를 전부 불러와서 dict 형으로 묶어버림
                                        #(str) 파일 경로
                                        in_path = 
                                        #(선택, str) 파일 경로 - 하위폴더명
                                       ,in_path_sub = 
                                       )
    '''
    in_path_tmp = kargs['in_path']
    try:
        in_path_sub = kargs['in_path_sub']
        if not in_path_tmp[-1] == "/":
            in_path = in_path_tmp + "/" + in_path_sub
        else:
            in_path = in_path_tmp + in_path_sub
    except:
        in_path = in_path_tmp
    
    if not in_path[-1] == "/":
        in_path += "/"
    
    list_name_files_tmp = os.listdir(in_path)
    list_name_files = sorted(list_name_files_tmp)
    
    print("\nin path:", in_path)
    print("in list images:", len(list_name_files))
    print("start with...", list_name_files[0])
    print("end with...", list_name_files[-1])
    
    dict_loaded_pils = {}
    for i_name in list_name_files:
        dict_loaded_pils[i_name] = Image.open(in_path + i_name)
    
    return dict_loaded_pils

#=== End of load_pils_2_dict


#PIL 이미지 3개 동시 출력 및 저장
#IN (3): (pil) 입력 원본, (pil) 정답, (pil) 예측
#IN (*): 'show' : (bool) plt show 여부 
#        'save' : (bool) 이미지 저장기능여부
#        'path' : (str)  저장경로
#        'name' : (str)  이미지 파일 이름
#        'title': (str)  전체 타이틀 내용
def pil_3_show_save(in_pil_x, in_pil_y, in_pil_h, **kargs):
    #print("(func) imshow_pil_3_v2")
    #print(kargs)
    try: #plt 창 출력여부
        in_switch_show = kargs['show']
    except:
        in_switch_show = False
    
    try: #plt 이미지 저장경로
        in_path = kargs['path']
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
    
    fig = plt.figure(figsize = (21,6))
    rows = 1
    cols = 3
    
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(np.array(in_pil_x))
    ax1.set_title("Input")
    #ax1.axis("off")
    
    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(np.array(in_pil_y))
    ax2.set_title("Ground truth")
    #ax2.axis("off")
    
    ax3 = fig.add_subplot(rows, cols, 3)
    ax3.imshow(np.array(in_pil_h))
    ax3.set_title("Predict")
    #ax3.axis("off")
    
    if in_title != "False":
        fig.suptitle(in_title)
    
    if in_switch_save:
        #print("save_path:", in_path)
        #print("file_name:", in_name)
        in_path_name = in_path + "/" + in_name
        
        try:
            if not os.path.exists(in_path):
                os.makedirs(in_path)
            
            try:
                plt.savefig(in_path_name, dpi = 100)
                #print("fig saved:", in_path_name)
            except:
                print("(except) save FAIL:", in_path_name)
                
        except OSError:
            print("(except) makedirs", in_path)
    
    if in_switch_show:
        plt.show()
    
    plt.close(fig)

#===

def pils_show_save(**kargs):
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
                  
                  ,return_pil_n_info = False
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
    
    
    try: # (pil, 파일이름) return 여부
        return_pil_n_info = kargs['return_pil_n_info']
    except:
        return_pil_n_info = False
    
    
    
    
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
                print("(except) save FAIL:", in_path_name)
                
        except OSError:
            print("(except) makedirs", in_path)
    
    if in_switch_show:
        plt.show()
    
    if return_pil_n_info:
        #return_pil = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())      # not works...
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, dpi = 100, format='png')
        return_pil = Image.open(img_buffer)
        return_path = in_path
        return_name = in_name
        
    plt.close(fig)
    
    
    if return_pil_n_info:
        return return_pil, return_path, return_name
    
#>>> @@@

#=== End of pils_show_save

'''

def cross_ft_2_plt_v1(**kargs): # cross features to single plt
    
    # only 1 part of batch should be input
    
    ts_pred_lab = kargs['ts_pred_lab'].clone().detach().cpu()             # (ts) 3ch tensor (c, h, w) predicted label
    ts_pred_img = kargs['ts_pred_img'].clone().detach().cpu()             # (ts) 3ch tensor (c, h, w) predicted image
    ts_ans_lab = kargs['ts_ans_lab'].clone().detach().cpu()               # (ts) 3ch tensor (c, h, w) answer label
    ts_ans_img = kargs['ts_ans_img'].clone().detach().cpu()               # (ts) 3ch tensor (c, h, w) answer image
    
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
    
    if is_save:
        in_path = kargs['path']                                     # (str) save file path
        in_name = kargs['name']                                     # (str) save file name
    
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
    ts_pred_img_g = ts_pred_img[0]*0.299 + ts_pred_img[1]*0.587 + ts_pred_img[2]*0.114       # [h_2, w_2]
    ts_ans_img_g  = ts_ans_img[0]*0.299  + ts_ans_img[1]*0.587  + ts_ans_img[2]*0.114        # [h_4, w_4]
    
    #print("ts_pred_lab", torch.min(ts_pred_lab), torch.max(ts_pred_lab))
    #print("ts_ans_lab", torch.min(ts_ans_lab), torch.max(ts_ans_lab))
    #print("ts_pred_img_g", torch.min(ts_pred_img_g), torch.max(ts_pred_img_g))
    #print("ts_ans_img_g", torch.min(ts_ans_img_g), torch.max(ts_ans_img_g))
    
    
    # cross-feature 1 & 2 ([c_1, h_1, w_1])
    #-> element-wise multiply (Hadamard product) with every single label-feature per class & gray answer image
    cf_1 = ts_pred_lab * ts_ans_img_g
    cf_2 = ts_ans_lab * ts_pred_img_g
    
    print("cf_1", torch.min(cf_1), torch.max(cf_1))
    print("cf_2", torch.min(cf_2), torch.max(cf_2))
    
    if flag_fig_size_set:
        plt.figure(figsize = fig_size)
    else:
        plt.figure(figsize = (4*c_1, 7))
    
    for i_channel in range(c_1):
        plt.subplot(2, c_1, i_channel + 1)
        plt.imshow(to_pil_image(cf_1[i_channel]))
        plt.title('PredLab & AnsImg C' + str(i_channel))
        
        plt.subplot(2, c_1, i_channel + 1 + c_1)
        plt.imshow(to_pil_image(cf_2[i_channel]))
        plt.title('AnsLab & PredImg C ' + str(i_channel))
    
    
    plt.suptitle(plt_title)
    
    if is_show:
        plt.show()
    
    if is_save:
        if in_path[-1] != "/":
            in_path += "/"
        
        if in_name[0] == '/':
            in_name = in_name[1:]
        
        in_path_name = in_path + in_name
        
        try:
            if not os.path.exists(in_path):
                os.makedirs(in_path)
            
            try:
                plt.savefig(in_path_name, dpi = 100)
                #print("fig saved:", in_path_name)
            except:
                print("(except) save FAIL:", in_path_name)
                
        except OSError:
            print("(except) makedirs", in_path)
        
    plt.close()

'''

#=== End of cross_ft_2_plt_v1

def cross_ft_2_plt_v2(**kargs): # cross features to single plt
    
    def ts_minmax(in_ts):
        #min-max scaler (min:0, max:1)
        return (in_ts - torch.min(in_ts)) / (torch.max(in_ts) - torch.min(in_ts))
    
    # only 1 part of batch should be input
    
    ts_pred_lab = kargs['ts_pred_lab'].clone().detach().cpu()             # (ts) 3ch tensor (c, h, w) predicted label
    ts_pred_img = kargs['ts_pred_img'].clone().detach().cpu()             # (ts) 3ch tensor (c, h, w) predicted image
    ts_ans_lab = kargs['ts_ans_lab'].clone().detach().cpu()               # (ts) 3ch tensor (c, h, w) answer label
    ts_ans_img = kargs['ts_ans_img'].clone().detach().cpu()               # (ts) 3ch tensor (c, h, w) answer image
    
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
    
    try:
        return_pil_n_info = kargs['return_pil_n_info']
    except:
        return_pil_n_info = False
    
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
        plt.figure(figsize = (4*c_1, 7))
    
    
    
    for i_channel in range(c_1):
        plt.subplot(2, c_1, i_channel + 1)
        plt.imshow(to_pil_image(ts_minmax(ts_pred_lab[i_channel]) * ts_ans_img_g))
        plt.title('PredLab & AnsImg C ' + str(i_channel))
        
        plt.subplot(2, c_1, i_channel + 1 + c_1)
        plt.imshow(to_pil_image(ts_minmax(ts_ans_lab[i_channel]) * ts_pred_img_g))
        plt.title('AnsLab & PredImg C ' + str(i_channel))
    
    
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
                print("(except) save FAIL:", in_path_name)
                
        except OSError:
            print("(except) makedirs", in_path)
        
    
    if return_pil_n_info:
        #return_pil = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())      # not works...
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, dpi = 100, format='png')
        return_pil = Image.open(img_buffer)
        return_path = in_path
        return_name = in_name
    
    plt.close()
    
    if return_pil_n_info:
        return return_pil, return_path, return_name


#=== End of cross_ft_2_plt_v2

def save_pil(**kargs):
    '''
    save_pil(#경로대로 폴더 생성 후 pil 이미지 저장
             #(pil) 이미지 
             pil = 
             #(str) 저장경로: ".asd/fghj"
            ,path = 
             #(str) 파일이름 + 확장자: "name.png"
            ,name = 
            )
    '''
    try:
        #저장할 pil 이미지
        in_pil = kargs['pil']
        
        #저장할 경로
        in_path = kargs['path']
        if not in_path[-1] == "/":
            in_path += "/"
        
        #파일 이름
        in_name = kargs['name']
        
        try:
            if not os.path.exists(in_path):
                os.makedirs(in_path)
                
            try:
                #이미지 저장
                in_path_name = in_path + in_name
                in_file_type = in_name.split(".")[-1]
                
                in_pil.save(in_path_name, in_file_type)
                print("PIL Image Saved: ", in_path_name)
                
            except:
                print("(except) in save_pil: save FAIL\n", in_path_name, in_file_type)
        except:
            print("(except) in save_pil: makedirs FAIL\n", in_path)
        
    except:
        print("(except) in save_pil: input error")

#=== End 0f save_pil

#IN (*):
#        line: (str)  출력할 문장들 (개수 자유)
#IN (**2):
#         path: (str)  폴더 경로
#         name: (str)  파일 이름
#IN (**):
#         reset:(bool) 파일 초기화 여부 (초기값: false)
#         jump: (bool) 첫 문장 "\n" 추가 여부 (초기값: false)
def update_txts(*lines, **kargs):
    try:
        in_path = kargs['path']
        in_name = kargs['name']
    except:
        print("(exc) in update_txt: input FAIL")
    
    try:
        is_reset = kargs['reset']
    except:
        is_reset = False
    
    try:
        is_jump = kargs['jump']
    except:
        is_jump = False
    
    #폴더 경로 체크 후 파일 체크
    try:
        if not os.path.exists(in_path):
            os.makedirs(in_path)
        #파일 존재 여부에 따라 방식 선택
        try:
            if is_reset:
                #파일 초기화, 기존 파일 내용을 제거한다
                file_txt = open(in_path + "/" + in_name, "w") 
                print("\n[TxT] File Reset Complete")
            else:
                #파일이 없는 경우, 생성 후 쓰기모드로 엶
                file_txt = open(in_path + "/" + in_name, "x") 
        except:
            #파일이 있는 경우, 이어쓰기모드로 엶
            file_txt = open(in_path + "/" + in_name, "a") 
        
        if is_jump:
            file_txt.write("\n")
        
        #문장 출력
        for line in lines:
            try:
                file_txt.write(line)
                file_txt.write("\n")
                print("[TxT]", line)
            except:
                file_txt.write("(exc) in update_txt: file write FAIL")
                file_txt.write("\n")
                print("(exc) in update_txt: file write FAIL")
        
        file_txt.close()
        

    except:
        print("(exc) in update_txt: file open FAIL")

#=== End of update_txts

#dict를 txt 파일로 저장
#IN (**3)
#       in_file_path: 저장경로 지정
#       in_file_name: 파일 이름 + (txt or csv)
#       in_dict: 딕셔너리 변수
#def dict_2_txt(**kargs):
def dict_2_txt(**kargs):
    #파일 경로
    in_file_path = kargs['in_file_path']

    #파일 이름
    in_file_name = kargs['in_file_name']
    #딕셔너리 변수
    in_dict = kargs['in_dict']

    in_keys = list(in_dict.keys())

    if in_file_path[-1] == "/":
        in_file_name = in_file_path + in_file_name
    else:
        in_file_name = in_file_path + "/" + in_file_name
    
    if not os.path.exists(in_file_path):
        os.makedirs(in_file_path)

    try:
        #기존 파일 덮어쓰기
        file_txt = open(in_file_name, 'w')
        try:
            list_keys = list(in_dict.keys())
            list_values = list(in_dict.values())
            for i_list in range(len(list_keys)):
                file_txt.write(str(list_keys[i_list]) + "," + str(list_values[i_list]) + "\n")
                
        except:
            print("(exc) dict access FAIL")
            sys.exit(1)
        
        file_txt.close()
        print("dict -> txt finished:", in_file_name)
    except:
        print("(exc) file open FAIL")

#=== End of dict_2_txt

#dict를 txt 파일로 저장
#IN (**3 or 4)
#       in_file_path: 저장경로 지정
#       in_file_name: 파일 이름 + (txt or csv)
#       <case 1>
#       in_dict: 딕셔너리 변수
#       <case 2>
#       in_dict_dict : (dict) in_dict 후보로 구성된 dict
#       in_dict_key  : (str)  이번에 저장할 dict의 key
def dict_2_txt_v2(**kargs):
    #파일 경로
    in_file_path = kargs['in_file_path']
    #파일 이름
    in_file_name = kargs['in_file_name']
    
    try:
        #딕셔너리 변수 바로 선택
        in_dict = kargs['in_dict']
    except:
        #후보 dict의 dict
        in_dict_dict = kargs['in_dict_dict']
        #이번에 update 할 dict의 key (str)
        in_dict_key = kargs['in_dict_key']
        #dict 결정
        in_dict = in_dict_dict[in_dict_key]
    
    in_keys = list(in_dict.keys())
    
    if in_file_path[-1] == "/":
        in_file_name = in_file_path + in_file_name
    else:
        in_file_name = in_file_path + "/" + in_file_name
    
    if not os.path.exists(in_file_path):
        os.makedirs(in_file_path)
    
    try:
        #기존 파일 덮어쓰기
        file_txt = open(in_file_name, 'w')
        try:
            list_keys = list(in_dict.keys())
            list_values = list(in_dict.values())
            for i_list in range(len(list_keys)):
                file_txt.write(str(list_keys[i_list]) + "," + str(list_values[i_list]) + "\n")
                
        except:
            print("(exc) dict access FAIL")
            sys.exit(1)
        
        file_txt.close()
        print("dict -> txt finished:", in_file_name)
    except:
        print("(exc) file open FAIL")

#=== End of dict_2_txt_v2

def csv_2_dict(**kargs):
    '''
    dict_from_csv = csv_2_dict(path_csv = "./aaa/bbb.csv")
    '''
    
    #첫 열을 key로, 나머지 열을 value로서 list 형 묶음으로 저장한 dict 변수 생성
    #csv 파일 경로
    path_csv = kargs['path_csv']

    file_csv = open(path_csv, 'r', encoding = 'utf-8')
    lines_csv = csv.reader(file_csv)
    
    dict_csv = {}
    
    for line_csv in lines_csv:
        item_num = 0
        items_tmp = []
        for item_csv in line_csv:
            #print(item_csv)
            if item_num != 0:
                items_tmp.append(item_csv)
            item_num += 1
        dict_csv[line_csv[0]] = items_tmp
        
    file_csv.close()
    
    return dict_csv

#=== End of csv_2_dict

if __name__ == '__main__':
    print("EoF: data_load_n_save.py")