import torch
import numpy as np
import torch.nn.functional as F
from scipy.special import softmax
from copy import deepcopy

# Prioritized Path Board
class PrioritizedBoard():
    def __init__(self, cfg, CHOICE_NUM=6, sta_num=(4, 4, 4, 4, 4), acc_gap=5):
        self.cfg = cfg
        self.prioritized_board = []
        self.choice_num = CHOICE_NUM
        self.sta_num = sta_num
        self.acc_gap = acc_gap


    # select teacher from prioritized board
    def select_teacher(self, model, random_cand):
        if self.cfg.SUPERNET.PICK_METHOD == 'top1':
            meta_value, teacher_cand = 0.5, sorted(
                self.prioritized_board, reverse=True)[0][3]
        elif self.cfg.SUPERNET.PICK_METHOD == 'meta':
            meta_value, cand_idx, teacher_cand = -1000000000, -1, None
            for now_idx, item in enumerate(self.prioritized_board):
                inputx = item[4]
                output = F.softmax(model(inputx, random_cand), dim=1)
                weight = model.module.forward_meta(output - item[5])
                if weight > meta_value:
                    meta_value = weight
                    cand_idx = now_idx
                    teacher_cand = self.prioritized_board[cand_idx][3]
            assert teacher_cand is not None
            meta_value = torch.sigmoid(-weight)
        else:
            raise ValueError('Method Not supported')

        return meta_value, teacher_cand


    def board_size(self):
        return len(self.prioritized_board)


    # get prob from config file
    def get_prob(self):
        if self.cfg.SUPERNET.HOW_TO_PROB == 'even' or (
                self.cfg.SUPERNET.HOW_TO_PROB == 'teacher' and len(self.prioritized_board) == 0):
            return None
        elif self.cfg.SUPERNET.HOW_TO_PROB == 'pre_prob':
            return self.cfg.SUPERNET.PRE_PROB
        elif self.cfg.SUPERNET.HOW_TO_PROB == 'teacher':
            op_dict = {}
            for i in range(self.choice_num):
                op_dict[i] = 0
            for item in self.prioritized_board:
                cand = item[3]
                for block in cand:
                    for op in block:
                        op_dict[op] += 1
            sum_op = 0
            for i in range(self.choice_num):
                sum_op = sum_op + op_dict[i]
            prob = []
            for i in range(self.choice_num):
                prob.append(float(op_dict[i]) / sum_op)
            del op_dict, sum_op
            return prob


    # sample random architecture
    def get_cand_with_prob(self, ignore_stages=[], prob=None):
        if prob is None:
            get_random_cand = [
                np.random.choice(
                    self.choice_num,
                    item).tolist() for item in self.sta_num]
        else:
            get_random_cand = [
                np.random.choice(
                    self.choice_num,
                    item,
                    prob).tolist() for item in self.sta_num]

        for stage in ignore_stages:
            for i, block in enumerate(get_random_cand[stage]):
                get_random_cand[stage][i] = 0

        return get_random_cand

    def sample_according_to_thetas(self, ignore_stages, thetas):
        current_theta = 0
        rng = np.linspace(0, self.choice_num - 1, self.choice_num, dtype=int)
        get_random_cand = [
                np.random.choice(
                    self.choice_num,
                    item).tolist() for item in self.sta_num]

        for stage in range(ignore_stages[-1]):
            for i, block in enumerate(get_random_cand[stage]):
                if stage in ignore_stages:
                    get_random_cand[stage][i] = 0
                else:
                    distribution = softmax(thetas[current_theta]().detach().cpu().numpy())
                    get_random_cand[stage][i] = np.random.choice(rng, p=distribution)
            if stage not in ignore_stages:
                current_theta += 1
                
        for stage in ignore_stages:
            for i, block in enumerate(get_random_cand[stage]):
                get_random_cand[stage][i] = 0
        
        return get_random_cand

    def parse_attentive_candidate(self, candidate, ignore_stages, block_to_choice_map):
        get_random_cand = [
            np.random.choice(
                self.choice_num,
                item).tolist() for item in self.sta_num]
            
        for stage in ignore_stages:
            for i, block in enumerate(get_random_cand[stage]):
                get_random_cand[stage][i] = 0

        if block_to_choice_map is not None:
            current_csp_block_idx = 0
            for stage in range(len(get_random_cand)):
                if stage not in ignore_stages:
                    for i, block in enumerate(get_random_cand[stage]):
                        n_bottlenecks = candidate['n_bottlenecks'][current_csp_block_idx]
                        gamma = candidate['gamma'][current_csp_block_idx]
                        current_choice = next((i for i, item in enumerate(block_to_choice_map) if (item['n_bottlenecks'] == n_bottlenecks and item['gamma'] == gamma)), None)# print('choice', get_random_cand[stage][i])
                        get_random_cand[stage][i] = current_choice

                    current_csp_block_idx += 1
    
        return get_random_cand


    def isUpdate(self, current_epoch, prec1, flops):
        if current_epoch <= self.cfg.SUPERNET.META_STA_EPOCH:
            return False

        if len(self.prioritized_board) < self.cfg.SUPERNET.POOL_SIZE:
            return True

        if prec1 > self.prioritized_board[-1][1] + self.acc_gap:
            return True

        if prec1 > self.prioritized_board[-1][1] and flops < self.prioritized_board[-1][2]:
            return True

        return False

    def update_prioritized_board(self, inputs, current_epoch, prec1, flops, cand, candidate_config):
        if self.isUpdate(current_epoch, prec1, flops):
            val_prec1 = prec1
            training_data = deepcopy(inputs[:self.cfg.SUPERNET.SLICE].detach())
            self.prioritized_board.append(
                (val_prec1,
                 prec1,
                 flops,
                 cand,
                 training_data,
                 candidate_config))
            self.prioritized_board = sorted(self.prioritized_board, reverse=True)

        if len(self.prioritized_board) > self.cfg.SUPERNET.POOL_SIZE:
            self.prioritized_board = sorted(self.prioritized_board, reverse=True)
            del self.prioritized_board[-1]

    def get_best_candidate_configuration(self):
        if (len(self.prioritized_board) > 0):
            return self.prioritized_board[0][6]

    def get_best_candidate_map(self):
        if (len(self.prioritized_board) > 0):
            return self.prioritized_board[0][0]