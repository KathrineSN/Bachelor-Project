# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 11:16:54 2021

@author: kathr
"""
#Importing dependencies
import os
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)
from mat_functions import *
import matplotlib.pyplot as plt
#from ccorr import ccorr
#from avg_matrices import load_avg_matrix


#%% Preprocessing of long epochs
file = 'pair0010_20200205_1230.bdf'
epo1, epo2 = prepocess_long(file)
#%% Bad channel identification
bads1 = ['FT7']
bads2 = ['PO3','P1']
epo1 = bad_removal(epo1, bads1)
#epo2 = bad_removal(epo2, bads2)  
#%%
ica1 = ica_part(epo1)
ica2 = ica_part(epo2)
#%%  Bad componenet identification of a
exclude1 = [0,1]
save_name1 = 'epochs_a_long_10.fif'
epo1_c = ica_removal_long(epo1, ica1, exclude1, save_name1)
#%%Bad componenet identification of b
exclude2 = [1,3]
save_name2 = 'epochs_b_long_10.fif'
epo2_c = ica_removal_long(epo2, ica2, exclude2, save_name2)
#%%
ccorr(epo1_c, epo2_c, 'pair0010', 'long',drop_list = [])
#%%
plt.close('all')
load_avg_matrix('beta','Coupled','long', plot = 0, sep = 1, save = 0)

#%% Preprocessing of short epochs
file_s = 'pair0010_20200205_1230.bdf'
epo1_s, epo2_s = prepocess_short(file_s)
#%% Bad channel identification
bads1_s = ['FT7']
bads2_s = ['PO3','P1']
epo1_2 = bad_removal(epo1_s, bads1_s)
epo2_s = bad_removal(epo2_s, bads2_s) 
#%%
plt.close('all')
ica1_s = ica_part(epo1_s)
ica2_s = ica_part(epo2_s)
#%%  Bad componenet identification of a
plt.close('all')
exclude1_s = [0,2]
exclude2_s = [1,3]
save_name1_s = 'epochs_a_short_10.fif'
save_name2_s = 'epochs_b_short_10.fif'
epo1_c_s, epo2_c_s = ica_removal_short(epo1_s, epo2_s, ica1_s, ica2_s, exclude1_s, exclude2_s, save_name1_s, save_name2_s)
#%%
plt.close('all')
drop_list_4 = [37, 87, 98, 123, 186, 194, 195, 217, 267, 268, 321, 336, 360, 394, 413, 575, 582, 657, 658, 676, 741, 742, 767, 768, 773, 774, 778, 779, 792, 793, 809, 819, 828, 893, 943, 966, 991, 1047, 1048, 1068, 1069, 1070, 1071, 1072, 1073, 1096, 1097, 1109, 1177, 1205, 1206, 1225, 1252, 1256, 1257, 1282, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1347, 1356, 1382, 1384, 1385, 1386, 1387, 1406, 1407, 1408, 1411, 1431, 1432, 1433, 1438, 1443, 1444, 1487, 1489, 1492, 1493, 1494, 1507, 1513, 1514, 1515, 1517, 1518, 1519, 1521, 1568, 1578, 1591, 1595, 1613, 1616, 1659, 1660, 1661]
drop_list_5 = [1, 3, 4, 5, 29, 106, 107, 140, 147, 149, 150, 151, 159, 163, 263, 264, 265, 266, 267, 268, 294, 299, 314, 320, 321, 323, 324, 327, 328, 353, 403, 404, 420, 527, 528, 556, 557, 558, 559, 577, 580, 588, 589, 597, 630, 690, 731, 739, 743, 744, 745, 746, 767, 807, 810, 813, 817, 821, 865, 866, 914, 932, 933, 934, 971, 994, 1025, 1026, 1027, 1028, 1029, 1030, 1046, 1047, 1048, 1050, 1051, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1078, 1088, 1089, 1123, 1124, 1147, 1151, 1152, 1154, 1155, 1156, 1157, 1161, 1176, 1196, 1198, 1199, 1229, 1230, 1231, 1232, 1233, 1234, 1236, 1237, 1238, 1239, 1240, 1242, 1252, 1253, 1254, 1255, 1274, 1304, 1305, 1306, 1307, 1331, 1332, 1359, 1360, 1403, 1450, 1458, 1459, 1464, 1487, 1512, 1514, 1515, 1516, 1517, 1518, 1537, 1562, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1586, 1588, 1589, 1590, 1591, 1601, 1604, 1605, 1606, 1607, 1608, 1621, 1622, 1623, 1624, 1627, 1629, 1631, 1639, 1641, 1646, 1647, 1648, 1660, 1661, 1662]
drop_list_7 = [60, 115, 133, 134, 165, 181, 186, 190, 207, 210, 211, 216, 218, 238, 239, 240, 280, 287, 293, 294, 295, 320, 321, 327, 348, 360, 373, 376, 377, 379, 387, 393, 396, 397, 399, 400, 401, 427, 428, 449, 451, 452, 469, 472, 473, 474, 492, 497, 498, 502, 505, 523, 549, 550, 551, 555, 577, 578, 585, 591, 619, 620, 661, 662, 675, 677, 689, 690, 691, 692, 731, 735, 737, 738, 739, 741, 742, 749, 755, 757, 758, 760, 762, 782, 783, 784, 788, 807, 809, 816, 817, 865, 866, 873, 878, 884, 886, 891, 892, 893, 894, 907, 913, 915, 917, 927, 939, 943, 945, 946, 965, 979, 983, 984, 990, 999, 1017, 1018, 1020, 1023, 1024, 1045, 1047, 1048, 1049, 1050, 1051, 1052, 1055, 1069, 1070, 1071, 1079, 1082, 1094, 1095, 1097, 1106, 1112, 1113, 1114, 1121, 1125, 1127, 1134, 1136, 1137, 1144, 1146, 1151, 1152, 1158, 1169, 1171, 1172, 1174, 1176, 1180, 1181, 1182, 1183, 1204, 1205, 1206, 1216, 1217, 1218, 1219, 1223, 1226, 1227, 1229, 1231, 1232, 1233, 1242, 1251, 1260, 1264, 1266, 1272, 1273, 1274, 1275, 1276, 1277, 1279, 1280, 1283, 1286, 1287, 1291, 1300, 1301, 1307, 1308, 1312, 1313, 1314, 1315, 1316, 1322, 1323, 1324, 1325, 1328, 1329, 1331, 1332, 1335, 1336, 1341, 1346, 1347, 1348, 1351, 1353, 1355, 1358, 1360, 1361, 1362, 1366, 1367, 1368, 1373, 1375, 1384, 1386, 1387, 1402, 1405, 1406, 1407, 1408, 1409, 1410, 1414, 1434, 1436, 1437, 1438, 1439, 1440, 1441, 1446, 1449, 1460, 1461, 1465, 1466, 1467, 1474, 1477, 1480, 1481, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1492, 1495, 1496, 1498, 1500, 1501, 1503, 1504, 1505, 1507, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1520, 1521, 1522, 1523, 1524, 1528, 1529, 1531, 1532, 1534, 1536, 1537, 1538, 1541, 1542, 1543, 1544, 1545, 1561, 1565, 1570, 1571, 1575, 1576, 1581, 1582, 1583, 1584, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1616, 1617, 1619, 1624, 1630, 1632, 1634, 1635, 1637, 1639, 1642, 1643, 1644, 1648, 1652, 1654, 1655, 1660]
drop_list_9 = [2, 30, 55, 81, 82, 110, 135, 151, 152, 163, 210, 211, 239, 240, 242, 258, 259, 262, 266, 289, 308, 309, 321, 325, 346, 351, 352, 353, 373, 408, 420, 471, 495, 496, 499, 501, 508, 509, 510, 511, 523, 533, 536, 581, 600, 603, 605, 609, 612, 613, 614, 618, 620, 622, 629, 653, 679, 688, 690, 694, 702, 706, 710, 738, 740, 761, 767, 791, 792, 794, 795, 797, 804, 805, 814, 835, 840, 841, 846, 852, 853, 858, 887, 891, 892, 893, 894, 918, 939, 990, 997, 998, 999, 1000, 1018, 1024, 1044, 1048, 1049, 1050, 1069, 1072, 1075, 1097, 1124, 1131, 1135, 1137, 1144, 1158, 1178, 1207, 1208, 1209, 1210, 1224, 1230, 1249, 1252, 1259, 1277, 1287, 1305, 1307, 1308, 1309, 1329, 1357, 1359, 1381, 1390, 1408, 1413, 1414, 1415, 1416, 1417, 1425, 1434, 1435, 1436, 1437, 1459, 1462, 1463, 1464, 1468, 1469, 1470, 1484, 1512, 1514, 1515, 1535, 1540, 1541, 1542, 1543, 1544, 1562, 1567, 1573, 1574, 1588, 1592, 1594, 1595, 1597, 1601, 1606, 1609, 1611, 1613, 1621, 1632, 1642, 1647, 1648, 1652, 1654]
drop_list_10 = [316, 323, 325, 326, 364, 481, 508, 577, 598, 599, 600, 806, 962, 988, 1105, 1118, 1196, 1202, 1430, 1482, 1586, 1587]
ccorr(epo1_c_s, epo2_c_s, 'pair0010', 'short', drop_list_10)
#%%
plt.close('all')
load_avg_matrix('beta','Coupled','short', plot = 0, sep = 1, save = 0)

#%%average matrices
load_avg_matrix('beta','Control','short', plot = 0, sep = 0, save = 0)

#%%
#Short epochs
load_avg_matrix('alpha', 'Coupled', 'short', save = 1)
load_avg_matrix('beta', 'Coupled', 'short', save = 1)
load_avg_matrix('theta', 'Coupled', 'short', save = 1)
load_avg_matrix('alpha', 'Uncoupled', 'short', save = 1)
load_avg_matrix('beta', 'Uncoupled', 'short', save = 1)
load_avg_matrix('theta', 'Uncoupled', 'short', save = 1)
load_avg_matrix('alpha', 'Control', 'short', save = 1)
load_avg_matrix('beta', 'Control', 'short', save = 1)
load_avg_matrix('theta', 'Control', 'short', save = 1)
load_avg_matrix('alpha', 'Leader-Follower', 'short', save = 1)
load_avg_matrix('beta', 'Leader-Follower', 'short', save = 1)
load_avg_matrix('theta', 'Leader-Follower', 'short', save = 1)

#%%
#Long epochs
load_avg_matrix('alpha', 'Coupled', 'long', save = 1)
load_avg_matrix('beta', 'Coupled', 'long', save = 1)
load_avg_matrix('theta', 'Coupled', 'long', save = 1)
load_avg_matrix('alpha', 'Uncoupled', 'long', save = 1)
load_avg_matrix('beta', 'Uncoupled', 'long', save = 1)
load_avg_matrix('theta', 'Uncoupled', 'long', save = 1)
load_avg_matrix('alpha', 'Control', 'long', save = 1)
load_avg_matrix('beta', 'Control', 'long', save = 1)
load_avg_matrix('theta', 'Control', 'long', save = 1)
load_avg_matrix('alpha', 'Leader-Follower', 'long', save = 1)
load_avg_matrix('beta', 'Leader-Follower', 'long', save = 1)
load_avg_matrix('theta', 'Leader-Follower', 'long', save = 1)