# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

# This dictionary is generated from calculating each operation of each layer to quickly search for layers.
# flops_op_dict[which_stage][which_operation] =
# (flops_of_operation_with_stride1, flops_of_operation_with_stride2)

flops_op_dict = {}
for i in range(27):
    flops_op_dict[i] = {}
flops_op_dict[0][0] = (21.828704, 18.820752)
flops_op_dict[0][1] = (32.669328, 28.16048)
flops_op_dict[0][2] = (25.039968, 23.637648)
flops_op_dict[0][3] = (37.486224, 35.385824)
flops_op_dict[0][4] = (29.856864, 30.862992)
flops_op_dict[0][5] = (44.711568, 46.22384)
flops_op_dict[1][0] = (11.808656, 11.86712)
flops_op_dict[1][1] = (17.68624, 17.780848)
flops_op_dict[1][2] = (13.01288, 13.87416)
flops_op_dict[1][3] = (19.492576, 20.791408)
flops_op_dict[1][4] = (14.819216, 16.88472)
flops_op_dict[1][5] = (22.20208, 25.307248)
flops_op_dict[2][0] = (8.198, 10.99632)
flops_op_dict[2][1] = (12.292848, 16.5172)
flops_op_dict[2][2] = (8.69976, 11.99984)
flops_op_dict[2][3] = (13.045488, 18.02248)
flops_op_dict[2][4] = (9.4524, 13.50512)
flops_op_dict[2][5] = (14.174448, 20.2804)
flops_op_dict[3][0] = (12.006112, 15.61632)
flops_op_dict[3][1] = (18.028752, 23.46096)
flops_op_dict[3][2] = (13.009632, 16.820544)
flops_op_dict[3][3] = (19.534032, 25.267296)
flops_op_dict[3][4] = (14.514912, 18.62688)
flops_op_dict[3][5] = (21.791952, 27.9768)
flops_op_dict[4][0] = (11.307456, 15.292416)
flops_op_dict[4][1] = (17.007072, 23.1504)
flops_op_dict[4][2] = (11.608512, 15.894528)
flops_op_dict[4][3] = (17.458656, 24.053568)
flops_op_dict[4][4] = (12.060096, 16.797696)
flops_op_dict[4][5] = (18.136032, 25.40832)
flops_op_dict[5][0] = (11.307456, 15.292416)
flops_op_dict[5][1] = (17.007072, 23.1504)
flops_op_dict[5][2] = (11.608512, 15.894528)
flops_op_dict[5][3] = (17.458656, 24.053568)
flops_op_dict[5][4] = (12.060096, 16.797696)
flops_op_dict[5][5] = (18.136032, 25.40832)
flops_op_dict[6][1] = (17.007072, 23.1504)
flops_op_dict[6][2] = (11.608512, 15.894528)
flops_op_dict[6][3] = (17.458656, 24.053568)
flops_op_dict[6][4] = (12.060096, 16.797696)
flops_op_dict[6][5] = (18.136032, 25.40832)
flops_op_dict[7][1] = (17.007072, 23.1504)
flops_op_dict[7][2] = (11.608512, 15.894528)
flops_op_dict[7][3] = (17.458656, 24.053568)
flops_op_dict[7][4] = (12.060096, 16.797696)
flops_op_dict[7][5] = (18.136032, 25.40832)
flops_op_dict[8][1] = (17.007072, 23.1504)
flops_op_dict[8][2] = (11.608512, 15.894528)
flops_op_dict[8][3] = (17.458656, 24.053568)
flops_op_dict[8][4] = (12.060096, 16.797696)
flops_op_dict[8][5] = (18.136032, 25.40832)
flops_op_dict[9][1] = (17.007072, 23.1504)
flops_op_dict[9][2] = (11.608512, 15.894528)
flops_op_dict[9][3] = (17.458656, 24.053568)
flops_op_dict[9][4] = (12.060096, 16.797696)
flops_op_dict[9][5] = (18.136032, 25.40832)
flops_op_dict[10][1] = (17.007072, 23.1504)
flops_op_dict[10][2] = (11.608512, 15.894528)
flops_op_dict[10][3] = (17.458656, 24.053568)
flops_op_dict[10][4] = (12.060096, 16.797696)
flops_op_dict[10][5] = (18.136032, 25.40832)
flops_op_dict[11][1] = (17.007072, 23.1504)
flops_op_dict[11][2] = (11.608512, 15.894528)
flops_op_dict[11][3] = (17.458656, 24.053568)
flops_op_dict[11][4] = (12.060096, 16.797696)
flops_op_dict[11][5] = (18.136032, 25.40832)
flops_op_dict[12][1] = (17.007072, 23.1504)
flops_op_dict[12][2] = (11.608512, 15.894528)
flops_op_dict[12][3] = (17.458656, 24.053568)
flops_op_dict[12][4] = (12.060096, 16.797696)
flops_op_dict[12][5] = (18.136032, 25.40832)
flops_op_dict[13][1] = (17.007072, 23.1504)
flops_op_dict[13][2] = (11.608512, 15.894528)
flops_op_dict[13][3] = (17.458656, 24.053568)
flops_op_dict[13][4] = (12.060096, 16.797696)
flops_op_dict[13][5] = (18.136032, 25.40832)
flops_op_dict[14][1] = (17.007072, 23.1504)
flops_op_dict[14][2] = (11.608512, 15.894528)
flops_op_dict[14][3] = (17.458656, 24.053568)
flops_op_dict[14][4] = (12.060096, 16.797696)
flops_op_dict[14][5] = (18.136032, 25.40832)
flops_op_dict[15][1] = (17.007072, 23.1504)
flops_op_dict[15][2] = (11.608512, 15.894528)
flops_op_dict[15][3] = (17.458656, 24.053568)
flops_op_dict[15][4] = (12.060096, 16.797696)
flops_op_dict[15][5] = (18.136032, 25.40832)
flops_op_dict[16][1] = (17.007072, 23.1504)
flops_op_dict[16][2] = (11.608512, 15.894528)
flops_op_dict[16][3] = (17.458656, 24.053568)
flops_op_dict[16][4] = (12.060096, 16.797696)
flops_op_dict[16][5] = (18.136032, 25.40832)
flops_op_dict[17][1] = (17.007072, 23.1504)
flops_op_dict[17][2] = (11.608512, 15.894528)
flops_op_dict[17][3] = (17.458656, 24.053568)
flops_op_dict[17][4] = (12.060096, 16.797696)
flops_op_dict[17][5] = (18.136032, 25.40832)
flops_op_dict[18][1] = (17.007072, 23.1504)
flops_op_dict[18][2] = (11.608512, 15.894528)
flops_op_dict[18][3] = (17.458656, 24.053568)
flops_op_dict[18][4] = (12.060096, 16.797696)
flops_op_dict[18][5] = (18.136032, 25.40832)
flops_op_dict[19][1] = (17.007072, 23.1504)
flops_op_dict[19][2] = (11.608512, 15.894528)
flops_op_dict[19][3] = (17.458656, 24.053568)
flops_op_dict[19][4] = (12.060096, 16.797696)
flops_op_dict[19][5] = (18.136032, 25.40832)
flops_op_dict[20][1] = (17.007072, 23.1504)
flops_op_dict[20][2] = (11.608512, 15.894528)
flops_op_dict[20][3] = (17.458656, 24.053568)
flops_op_dict[20][4] = (12.060096, 16.797696)
flops_op_dict[20][5] = (18.136032, 25.40832)
flops_op_dict[21][1] = (17.007072, 23.1504)
flops_op_dict[21][2] = (11.608512, 15.894528)
flops_op_dict[21][3] = (17.458656, 24.053568)
flops_op_dict[21][4] = (12.060096, 16.797696)
flops_op_dict[21][5] = (18.136032, 25.40832)
flops_op_dict[22][1] = (17.007072, 23.1504)
flops_op_dict[22][2] = (11.608512, 15.894528)
flops_op_dict[22][3] = (17.458656, 24.053568)
flops_op_dict[22][4] = (12.060096, 16.797696)
flops_op_dict[22][5] = (18.136032, 25.40832)
flops_op_dict[23][1] = (17.007072, 23.1504)
flops_op_dict[23][2] = (11.608512, 15.894528)
flops_op_dict[23][3] = (17.458656, 24.053568)
flops_op_dict[23][4] = (12.060096, 16.797696)
flops_op_dict[23][5] = (18.136032, 25.40832)
flops_op_dict[24][1] = (17.007072, 23.1504)
flops_op_dict[24][2] = (11.608512, 15.894528)
flops_op_dict[24][3] = (17.458656, 24.053568)
flops_op_dict[24][4] = (12.060096, 16.797696)
flops_op_dict[24][5] = (18.136032, 25.40832)
flops_op_dict[25][1] = (17.007072, 23.1504)
flops_op_dict[25][2] = (11.608512, 15.894528)
flops_op_dict[25][3] = (17.458656, 24.053568)
flops_op_dict[25][4] = (12.060096, 16.797696)
flops_op_dict[25][5] = (18.136032, 25.40832)
flops_op_dict[26][1] = (17.007072, 23.1504)
flops_op_dict[26][2] = (11.608512, 15.894528)
flops_op_dict[26][3] = (17.458656, 24.053568)
flops_op_dict[26][4] = (12.060096, 16.797696)
flops_op_dict[26][5] = (18.136032, 25.40832)
