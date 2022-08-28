from tqdm import tqdm
import numpy as np
import json

def collect_samples(n_samples, model, prioritized_board, flops_estimator):
    with open('candidate_samples_8_6_4_2_025_05_075.txt', 'w') as samples_file:
        for i in tqdm(range(n_samples)):
            prob = prioritized_board.get_prob()
            random_cand = prioritized_board.get_cand_with_prob(prob=prob, ignore_stages=model.ignore_stages)
            random_cand = random_cand[1:]
            random_cand.insert(0, [0])
            cand_flops = flops_estimator.get_flops(random_cand)
            mapped_candidate = map_arch_to_blocks(random_cand, model)
            mapped_candidate['flops'] = cand_flops
            mapped_candidate['resolution'] = 416
            samples_file.write(f'{mapped_candidate}\n')

# def parse_candidate_samples(path_to_samples, model):
#     samples = []
#     with open(path_to_samples, 'r') as samples_file:
#         for candidate in samples_file:
#             candidate_info = candidate.split(':')
#             arch_definition = json.loads(candidate_info[0])
#             flops = float(candidate_info[1])
#             mapped_candidate = map_arch_to_blocks(arch_definition, model)
#             mapped_candidate['flops'] = flops
#             samples.append(mapped_candidate)
            

def map_arch_to_blocks(arch_definition, model):
    num_csp = []
    gamma = []
    for layer, layer_arch in zip(model.blocks, arch_definition):
        for blocks, arch in zip(layer, layer_arch):
            if arch == -1:
                continue

            # if (arch <= 2):
            #     chosen_block = blocks[0]
            # elif (arch > 2):
            #     chosen_block = blocks[1]
            #     arch -= len(model.hetero_choices)

            if (arch <= 3):
                chosen_block = blocks[0]
            elif (arch > 3 and arch <= 7):
                chosen_block = blocks[1]
                arch -= len(model.hetero_choices)
            elif (arch > 7):
                chosen_block = blocks[2]
                arch -= len(model.hetero_choices) * 2

            if 'bottlecsp' in chosen_block.block_name:
                current_num_csp, current_gamma = parse_block_name(chosen_block.block_name)
                num_csp.append(model.hetero_choices[arch])
                gamma.append(current_gamma)
    
    mapped_candidate = {'n_bottlenecks': num_csp, 'gamma': gamma}
    return mapped_candidate

        


def parse_block_name(block_name):
    num_csp = int(block_name.split('_')[1][-1])
    gamma = float(block_name.split('_')[-1].split('gamma')[1])
    return num_csp, gamma

        