# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:50:16 2021

@author: kathr
"""
import os
path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Bachelor-Project"
os.chdir(path)
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from hypyp import prep 
from hypyp import analyses
from hypyp import stats
from hypyp import viz
from collections import Counter
from collections import OrderedDict
from itertools import groupby
from con_functions import *

#%% Loading pair
# for each pair do
#epochs_a = mne.read_epochs('epochs_a_long_3.fif')
#epochs_b = mne.read_epochs('epochs_b_long_3.fif')

#epochs_a_3s = mne.read_epochs('epochs_a_3sec_3.fif')
#epochs_b_3s = mne.read_epochs('epochs_b_3sec_3.fif')

epochs_a_s = mne.read_epochs('epochs_a_short_18.fif')
epochs_b_s = mne.read_epochs('epochs_b_short_18.fif')

#%% defining drop lists
#drop_list_3 = [91, 108, 300, 301, 341, 351, 354, 355, 356, 381, 382, 383, 397, 398, 416, 442, 443, 473, 474, 476, 477, 497, 498, 502, 507, 508, 509, 510, 511, 512, 513, 514, 528, 530, 550, 551, 553, 554, 555, 556, 557, 559, 561, 578, 585, 586, 587, 588, 589, 603, 604, 622, 632, 654, 658, 660, 662, 663, 664, 669, 675, 677, 678, 683, 684, 686, 720, 721, 723, 724, 725, 727, 908, 967, 974, 975, 976, 980, 1011, 1027, 1031, 1033, 1034, 1036, 1041, 1042, 1072, 1073, 1125, 1126, 1145, 1146, 1147, 1148, 1260, 1271, 1279, 1292, 1303, 1314, 1315, 1329, 1338, 1339, 1340, 1341, 1347, 1349, 1351, 1358, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1378, 1395, 1396, 1408, 1439, 1463, 1486, 1514, 1515, 1517, 1519, 1520, 1521, 1522, 1523, 1628, 1630, 1631, 1632, 1633, 1643, 1655, 1656, 1657, 1660, 1661, 1662, 1663]
#drop_list_4 = [37, 87, 98, 123, 186, 194, 195, 217, 267, 268, 321, 336, 360, 394, 413, 575, 582, 657, 658, 676, 741, 742, 767, 768, 773, 774, 778, 779, 792, 793, 809, 819, 828, 893, 943, 966, 991, 1047, 1048, 1068, 1069, 1070, 1071, 1072, 1073, 1096, 1097, 1109, 1177, 1205, 1206, 1225, 1252, 1256, 1257, 1282, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1347, 1356, 1382, 1384, 1385, 1386, 1387, 1406, 1407, 1408, 1411, 1431, 1432, 1433, 1438, 1443, 1444, 1487, 1489, 1492, 1493, 1494, 1507, 1513, 1514, 1515, 1517, 1518, 1519, 1521, 1568, 1578, 1591, 1595, 1613, 1616, 1659, 1660, 1661]
#drop_list_5 = [1, 3, 4, 5, 29, 106, 107, 140, 147, 149, 150, 151, 159, 163, 263, 264, 265, 266, 267, 268, 294, 299, 314, 320, 321, 323, 324, 327, 328, 353, 403, 404, 420, 527, 528, 556, 557, 558, 559, 577, 580, 588, 589, 597, 630, 690, 731, 739, 743, 744, 745, 746, 767, 807, 810, 813, 817, 821, 865, 866, 914, 932, 933, 934, 971, 994, 1025, 1026, 1027, 1028, 1029, 1030, 1046, 1047, 1048, 1050, 1051, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1078, 1088, 1089, 1123, 1124, 1147, 1151, 1152, 1154, 1155, 1156, 1157, 1161, 1176, 1196, 1198, 1199, 1229, 1230, 1231, 1232, 1233, 1234, 1236, 1237, 1238, 1239, 1240, 1242, 1252, 1253, 1254, 1255, 1274, 1304, 1305, 1306, 1307, 1331, 1332, 1359, 1360, 1403, 1450, 1458, 1459, 1464, 1487, 1512, 1514, 1515, 1516, 1517, 1518, 1537, 1562, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1586, 1588, 1589, 1590, 1591, 1601, 1604, 1605, 1606, 1607, 1608, 1621, 1622, 1623, 1624, 1627, 1629, 1631, 1639, 1641, 1646, 1647, 1648, 1660, 1661, 1662]
#drop_list_7 = [60, 115, 133, 134, 165, 181, 186, 190, 207, 210, 211, 216, 218, 238, 239, 240, 280, 287, 293, 294, 295, 320, 321, 327, 348, 360, 373, 376, 377, 379, 387, 393, 396, 397, 399, 400, 401, 427, 428, 449, 451, 452, 469, 472, 473, 474, 492, 497, 498, 502, 505, 523, 549, 550, 551, 555, 577, 578, 585, 591, 619, 620, 661, 662, 675, 677, 689, 690, 691, 692, 731, 735, 737, 738, 739, 741, 742, 749, 755, 757, 758, 760, 762, 782, 783, 784, 788, 807, 809, 816, 817, 865, 866, 873, 878, 884, 886, 891, 892, 893, 894, 907, 913, 915, 917, 927, 939, 943, 945, 946, 965, 979, 983, 984, 990, 999, 1017, 1018, 1020, 1023, 1024, 1045, 1047, 1048, 1049, 1050, 1051, 1052, 1055, 1069, 1070, 1071, 1079, 1082, 1094, 1095, 1097, 1106, 1112, 1113, 1114, 1121, 1125, 1127, 1134, 1136, 1137, 1144, 1146, 1151, 1152, 1158, 1169, 1171, 1172, 1174, 1176, 1180, 1181, 1182, 1183, 1204, 1205, 1206, 1216, 1217, 1218, 1219, 1223, 1226, 1227, 1229, 1231, 1232, 1233, 1242, 1251, 1260, 1264, 1266, 1272, 1273, 1274, 1275, 1276, 1277, 1279, 1280, 1283, 1286, 1287, 1291, 1300, 1301, 1307, 1308, 1312, 1313, 1314, 1315, 1316, 1322, 1323, 1324, 1325, 1328, 1329, 1331, 1332, 1335, 1336, 1341, 1346, 1347, 1348, 1351, 1353, 1355, 1358, 1360, 1361, 1362, 1366, 1367, 1368, 1373, 1375, 1384, 1386, 1387, 1402, 1405, 1406, 1407, 1408, 1409, 1410, 1414, 1434, 1436, 1437, 1438, 1439, 1440, 1441, 1446, 1449, 1460, 1461, 1465, 1466, 1467, 1474, 1477, 1480, 1481, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1492, 1495, 1496, 1498, 1500, 1501, 1503, 1504, 1505, 1507, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1520, 1521, 1522, 1523, 1524, 1528, 1529, 1531, 1532, 1534, 1536, 1537, 1538, 1541, 1542, 1543, 1544, 1545, 1561, 1565, 1570, 1571, 1575, 1576, 1581, 1582, 1583, 1584, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1616, 1617, 1619, 1624, 1630, 1632, 1634, 1635, 1637, 1639, 1642, 1643, 1644, 1648, 1652, 1654, 1655, 1660]
#drop_list_9 = [2, 30, 55, 81, 82, 110, 135, 151, 152, 163, 210, 211, 239, 240, 242, 258, 259, 262, 266, 289, 308, 309, 321, 325, 346, 351, 352, 353, 373, 408, 420, 471, 495, 496, 499, 501, 508, 509, 510, 511, 523, 533, 536, 581, 600, 603, 605, 609, 612, 613, 614, 618, 620, 622, 629, 653, 679, 688, 690, 694, 702, 706, 710, 738, 740, 761, 767, 791, 792, 794, 795, 797, 804, 805, 814, 835, 840, 841, 846, 852, 853, 858, 887, 891, 892, 893, 894, 918, 939, 990, 997, 998, 999, 1000, 1018, 1024, 1044, 1048, 1049, 1050, 1069, 1072, 1075, 1097, 1124, 1131, 1135, 1137, 1144, 1158, 1178, 1207, 1208, 1209, 1210, 1224, 1230, 1249, 1252, 1259, 1277, 1287, 1305, 1307, 1308, 1309, 1329, 1357, 1359, 1381, 1390, 1408, 1413, 1414, 1415, 1416, 1417, 1425, 1434, 1435, 1436, 1437, 1459, 1462, 1463, 1464, 1468, 1469, 1470, 1484, 1512, 1514, 1515, 1535, 1540, 1541, 1542, 1543, 1544, 1562, 1567, 1573, 1574, 1588, 1592, 1594, 1595, 1597, 1601, 1606, 1609, 1611, 1613, 1621, 1632, 1642, 1647, 1648, 1652, 1654]
#drop_list_10 = [316, 323, 325, 326, 364, 481, 508, 577, 598, 599, 600, 806, 962, 988, 1105, 1118, 1196, 1202, 1430, 1482, 1586, 1587]

#drop_list_3 = [0, 117, 134, 192, 193, 326, 327, 348, 367, 377, 378, 380, 381, 382, 390, 392, 407, 408, 409, 423, 424, 442, 468, 469, 499, 500, 501, 502, 503, 523, 524, 528, 529, 533, 534, 535, 536, 537, 538, 539, 540, 541, 554, 556, 576, 577, 578, 579, 580, 581, 582, 585, 586, 587, 601, 604, 608, 610, 611, 612, 613, 614, 615, 629, 630, 631, 648, 657, 658, 680, 684, 685, 686, 688, 689, 690, 695, 701, 703, 704, 708, 709, 710, 712, 736, 737, 746, 747, 749, 750, 751, 752, 753, 774, 844, 917, 933, 934, 993, 1000, 1001, 1002, 1003, 1006, 1018, 1019, 1028, 1036, 1037, 1052, 1053, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1067, 1068, 1097, 1098, 1099, 1100, 1143, 1151, 1152, 1171, 1172, 1173, 1174, 1180, 1216, 1217, 1235, 1237, 1286, 1297, 1305, 1318, 1333, 1340, 1341, 1352, 1355, 1356, 1364, 1365, 1366, 1367, 1373, 1375, 1377, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1404, 1421, 1422, 1434, 1465, 1466, 1489, 1512, 1540, 1541, 1543, 1545, 1546, 1547, 1548, 1549, 1615, 1651, 1654, 1656, 1657, 1658, 1659, 1662, 1669, 1680, 1681, 1682, 1683, 1686, 1687, 1688, 1689, 1694]
#drop_list_4 = [63, 220, 243, 294, 362, 386, 503, 601, 608, 609, 683, 684, 702, 767, 794, 799, 800, 804, 818, 835, 854, 865, 992, 1017, 1019, 1073, 1074, 1094, 1095, 1096, 1097, 1098, 1099, 1122, 1123, 1203, 1231, 1251, 1282, 1283, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1373, 1382, 1410, 1411, 1413, 1433, 1457, 1458, 1459, 1469, 1470, 1515, 1518, 1519, 1520, 1539, 1540, 1543, 1544, 1545, 1604, 1639, 1642, 1685, 1686]
#drop_list_5 = [29, 55, 87, 162, 175, 176, 177, 191, 271, 289, 290, 293, 294, 384, 436, 471, 495, 521, 573, 582, 583, 584, 585, 588, 597, 599, 601, 603, 614, 615, 631, 658, 659, 751, 757, 768, 769, 770, 771, 772, 773, 775, 833, 837, 839, 840, 843, 844, 846, 847, 864, 894, 939, 940, 941, 947, 957, 958, 959, 960, 1020, 1021, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1072, 1073, 1074, 1077, 1095, 1097, 1099, 1104, 1114, 1121, 1129, 1149, 1150, 1151, 1171, 1172, 1173, 1174, 1175, 1177, 1178, 1180, 1181, 1182, 1183, 1187, 1200, 1222, 1223, 1224, 1225, 1226, 1252, 1253, 1255, 1256, 1257, 1258, 1259, 1260, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1278, 1279, 1280, 1281, 1300, 1330, 1331, 1332, 1333, 1334, 1335, 1357, 1358, 1359, 1385, 1386, 1410, 1513, 1540, 1541, 1542, 1543, 1544, 1563, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1602, 1607, 1612, 1614, 1616, 1618, 1619, 1627, 1630, 1631, 1632, 1633, 1634, 1635, 1639, 1642, 1647, 1648, 1649, 1667, 1673, 1686, 1688, 1693]
#drop_list_7 = [86, 156, 160, 212, 236, 237, 264, 265, 306, 320, 374, 402, 477, 478, 495, 498, 528, 549, 575, 576, 577, 603, 611, 645, 646, 715, 716, 717, 718, 765, 767, 788, 814, 843, 844, 891, 892, 912, 917, 918, 919, 920, 939, 941, 953, 971, 972, 991, 1005, 1009, 1010, 1016, 1043, 1050, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1095, 1096, 1097, 1108, 1121, 1139, 1151, 1160, 1162, 1163, 1170, 1184, 1197, 1200, 1202, 1206, 1207, 1231, 1242, 1243, 1244, 1245, 1253, 1255, 1257, 1269, 1277, 1286, 1290, 1292, 1298, 1299, 1300, 1302, 1303, 1313, 1326, 1333, 1334, 1340, 1341, 1342, 1354, 1355, 1362, 1377, 1381, 1384, 1386, 1387, 1392, 1399, 1412, 1428, 1431, 1433, 1434, 1435, 1440, 1462, 1463, 1464, 1472, 1477, 1486, 1500, 1503, 1506, 1511, 1512, 1513, 1518, 1521, 1522, 1526, 1529, 1530, 1533, 1539, 1543, 1544, 1546, 1547, 1548, 1554, 1557, 1560, 1562, 1563, 1565, 1566, 1569, 1587, 1597, 1602, 1607, 1609, 1610, 1617, 1618, 1619, 1620, 1624, 1626, 1656, 1658, 1660, 1661, 1663, 1668, 1680, 1681, 1686]
#drop_list_9 = [2, 28, 56, 81, 107, 136, 161, 189, 236, 237, 265, 288, 315, 351, 372, 378, 379, 399, 446, 497, 527, 534, 535, 536, 537, 549, 626, 631, 635, 655, 679, 705, 732, 764, 817, 840, 861, 901, 913, 918, 919, 939, 944, 965, 969, 1016, 1024, 1025, 1026, 1044, 1050, 1075, 1076, 1095, 1153, 1184, 1204, 1233, 1234, 1235, 1256, 1278, 1303, 1331, 1333, 1334, 1335, 1355, 1383, 1407, 1434, 1439, 1440, 1441, 1442, 1443, 1460, 1485, 1488, 1489, 1492, 1495, 1510, 1538, 1567, 1568, 1569, 1570, 1593, 1594, 1599, 1600, 1618, 1620, 1623, 1647, 1668, 1674, 1713, 1714, 1715]
#drop_list_10 = [342, 351, 352, 353, 534, 603, 624, 625, 626, 832, 988, 1014, 1131, 1144, 1196, 1222, 1228, 1456, 1612, 1613, 1614]


#%% Circular correlation
       
#theta, alpha, beta, angle, complex_signal, epo_a_cleaned, epo_a = ccorr(epochs_a, epochs_b, 'pair0014', 'long', drop_list = [])
theta, alpha, beta, angle, complex_signal, epo_a_cleaned, epo_a = ccorr(epochs_a_s, epochs_b_s, 'pair0018', 'short', drop_list = [])
#theta, alpha, beta, angle, complex_signal, epo_a_cleaned, epo_a = ccorr(epochs_a_3s, epochs_b_3s, 'pair003', '3sec', drop_list = [])


#%% Coherence

#theta, alpha, beta, amp, complex_signal, epo_a, epo_a_cleaned = coh(epochs_a, epochs_b, 'pair0014', 'long', drop_list = [])
theta, alpha, beta, amp, complex_signal_coh, epo_a_coh, epo_a_cleaned_coh = coh(epochs_a_3s, epochs_b_3s, 'pair003', '3sec', drop_list = [])

#%% Avg. matrices
#Short epochs
#ccorr
load_avg_matrix('ccorr','alpha', 'Coupled', 'short', save = 1)
load_avg_matrix('ccorr','beta', 'Coupled', 'short', save = 1)
load_avg_matrix('ccorr','theta', 'Coupled', 'short', save = 1)
load_avg_matrix('ccorr','alpha', 'Uncoupled', 'short', save = 1)
load_avg_matrix('ccorr','beta', 'Uncoupled', 'short', save = 1)
load_avg_matrix('ccorr','theta', 'Uncoupled', 'short', save = 1)
load_avg_matrix('ccorr','alpha', 'Control', 'short', save = 1)
load_avg_matrix('ccorr','beta', 'Control', 'short', save = 1)
load_avg_matrix('ccorr','theta', 'Control', 'short', save = 1)
load_avg_matrix('ccorr','alpha', 'Leader-Follower', 'short', save = 1)
load_avg_matrix('ccorr','beta', 'Leader-Follower', 'short', save = 1)
load_avg_matrix('ccorr','theta', 'Leader-Follower', 'short', save = 1)
#load_avg_matrix('ccorr','alpha', 'Resting', 'short', save = 1)
#load_avg_matrix('ccorr','beta', 'Resting', 'short', save = 1)
#load_avg_matrix('ccorr','theta', 'Resting', 'short', save = 1)
#%%
#coh
load_avg_matrix('coh','alpha', 'Coupled', 'short', save = 1)
load_avg_matrix('coh','beta', 'Coupled', 'short', save = 1)
load_avg_matrix('coh','theta', 'Coupled', 'short', save = 1)
load_avg_matrix('coh','alpha', 'Uncoupled', 'short', save = 1)
load_avg_matrix('coh','beta', 'Uncoupled', 'short', save = 1)
load_avg_matrix('coh','theta', 'Uncoupled', 'short', save = 1)
load_avg_matrix('coh','alpha', 'Control', 'short', save = 1)
load_avg_matrix('coh','beta', 'Control', 'short', save = 1)
load_avg_matrix('coh','theta', 'Control', 'short', save = 1)
load_avg_matrix('coh','alpha', 'Leader-Follower', 'short', save = 1)
load_avg_matrix('coh','beta', 'Leader-Follower', 'short', save = 1)
load_avg_matrix('coh','theta', 'Leader-Follower', 'short', save = 1)
#load_avg_matrix('coh','alpha', 'Resting', 'short', save = 1)
#load_avg_matrix('coh','beta', 'Resting', 'short', save = 1)
#load_avg_matrix('coh','theta', 'Resting', 'short', save = 1)
#%%
#3sec epochs
#ccorr
load_avg_matrix('ccorr','alpha', 'Coupled', '3sec', save = 1)
load_avg_matrix('ccorr','beta', 'Coupled', '3sec', save = 1)
load_avg_matrix('ccorr','theta', 'Coupled', '3sec', save = 1)
load_avg_matrix('ccorr','alpha', 'Uncoupled', '3sec', save = 1)
load_avg_matrix('ccorr','beta', 'Uncoupled', '3sec', save = 1)
load_avg_matrix('ccorr','theta', 'Uncoupled', '3sec', save = 1)
load_avg_matrix('ccorr','alpha', 'Control', '3sec', save = 1)
load_avg_matrix('ccorr','beta', 'Control', '3sec', save = 1)
load_avg_matrix('ccorr','theta', 'Control', '3sec', save = 1)
load_avg_matrix('ccorr','alpha', 'Leader-Follower', '3sec', save = 1)
load_avg_matrix('ccorr','beta', 'Leader-Follower', '3sec', save = 1)
load_avg_matrix('ccorr','theta', 'Leader-Follower', '3sec', save = 1)
#load_avg_matrix('ccorr','alpha', 'Resting', 'short', save = 1)
#load_avg_matrix('ccorr','beta', 'Resting', 'short', save = 1)
#load_avg_matrix('ccorr','theta', 'Resting', 'short', save = 1)
#%%
#coh
load_avg_matrix('coh','alpha', 'Coupled', '3sec', save = 1)
load_avg_matrix('coh','beta', 'Coupled', '3sec', save = 1)
load_avg_matrix('coh','theta', 'Coupled', '3sec', save = 1)
load_avg_matrix('coh','alpha', 'Uncoupled', '3sec', save = 1)
load_avg_matrix('coh','beta', 'Uncoupled', '3sec', save = 1)
load_avg_matrix('coh','theta', 'Uncoupled', '3sec', save = 1)
load_avg_matrix('coh','alpha', 'Control', '3sec', save = 1)
load_avg_matrix('coh','beta', 'Control', '3sec', save = 1)
load_avg_matrix('coh','theta', 'Control', '3sec', save = 1)
load_avg_matrix('coh','alpha', 'Leader-Follower', '3sec', save = 1)
load_avg_matrix('coh','beta', 'Leader-Follower', '3sec', save = 1)
load_avg_matrix('coh','theta', 'Leader-Follower', '3sec', save = 1)
#load_avg_matrix('coh','alpha', 'Resting', 'short', save = 1)
#load_avg_matrix('coh','beta', 'Resting', 'short', save = 1)
#load_avg_matrix('coh','theta', 'Resting', 'short', save = 1)
#%%
#Long epochs
#ccorr
plt.close('all')
load_avg_matrix('ccorr','alpha', 'Coupled', 'long', save = 1)
load_avg_matrix('ccorr','beta', 'Coupled', 'long', save = 1)
load_avg_matrix('ccorr','theta', 'Coupled', 'long', save = 1)
load_avg_matrix('ccorr','alpha', 'Uncoupled', 'long', save = 1)
load_avg_matrix('ccorr','beta', 'Uncoupled', 'long', save = 1)
load_avg_matrix('ccorr','theta', 'Uncoupled', 'long', save = 1)
load_avg_matrix('ccorr','alpha', 'Control', 'long', save = 1)
load_avg_matrix('ccorr','beta', 'Control', 'long', save = 1)
load_avg_matrix('ccorr','theta', 'Control', 'long', save = 1)
load_avg_matrix('ccorr','alpha', 'Leader-Follower', 'long', save = 1)
load_avg_matrix('ccorr','beta', 'Leader-Follower', 'long', save = 1)
load_avg_matrix('ccorr','theta', 'Leader-Follower', 'long', save = 1)
#load_avg_matrix('ccorr','alpha', 'Resting', 'long', save = 1)
#load_avg_matrix('ccorr','beta', 'Resting', 'long', save = 1)
#load_avg_matrix('ccorr','theta', 'Resting', 'long', save = 1)
#%%
#coh
plt.close('all')
load_avg_matrix('coh','alpha', 'Coupled', 'long', save = 1)
load_avg_matrix('coh','beta', 'Coupled', 'long', save = 0, sep = 1)
load_avg_matrix('coh','theta', 'Coupled', 'long', save = 1)
load_avg_matrix('coh','alpha', 'Uncoupled', 'long', save = 1)
load_avg_matrix('coh','beta', 'Uncoupled', 'long', save = 1)
load_avg_matrix('coh','theta', 'Uncoupled', 'long', save = 1)
load_avg_matrix('coh','alpha', 'Control', 'long', save = 1)
load_avg_matrix('coh','beta', 'Control', 'long', save = 1)
load_avg_matrix('coh','theta', 'Control', 'long', save = 1)
load_avg_matrix('coh','alpha', 'Leader-Follower', 'long', save = 1)
load_avg_matrix('coh','beta', 'Leader-Follower', 'long', save = 1)
load_avg_matrix('coh','theta', 'Leader-Follower', 'long', save = 1)
#load_avg_matrix('ccorr','alpha', 'Resting', 'long', save = 1)
#load_avg_matrix('ccorr','beta', 'Resting', 'long', save = 1)
#load_avg_matrix('ccorr','theta', 'Resting', 'long', save = 1)


#%% Difference between with and with resting
alpha_coupled_with = load_avg_matrix('ccorr','alpha', 'Coupled', 'long', save = 0)
plt.title('coupled alpha with resting')

beta_coupled_with = load_avg_matrix('ccorr','beta', 'Coupled', 'long', save = 0)
theta_coupled_with = load_avg_matrix('ccorr','theta', 'Coupled', 'long', save = 0)
alpha_uncoupled_with = load_avg_matrix('ccorr','alpha', 'Uncoupled', 'long', save = 0)
beta_uncoupled_with = load_avg_matrix('ccorr','beta', 'Uncoupled', 'long', save = 0)
theta_uncoupled_with = load_avg_matrix('ccorr','theta', 'Uncoupled', 'long', save = 0)
alpha_control_with = load_avg_matrix('ccorr','alpha', 'Control', 'long', save = 0)
beta_control_with = load_avg_matrix('ccorr','beta', 'Control', 'long', save = 0)
theta_control_with = load_avg_matrix('ccorr','theta', 'Control', 'long', save = 0)
alpha_LF_with = load_avg_matrix('ccorr','alpha', 'Leader-Follower', 'long', save = 0)
beta_LF_with = load_avg_matrix('ccorr','beta', 'Leader-Follower', 'long', save = 0)
theta_LF_with = load_avg_matrix('ccorr','theta', 'Leader-Follower', 'long', save = 0)

alpha_coupled_without = load_avg_matrix('ccorr','alpha', 'Coupled', 'long', save = 0, p = 1)
plt.title('coupled alpha without resting')
beta_coupled_without = load_avg_matrix('ccorr','beta', 'Coupled', 'long', save = 0, p = 1)
theta_coupled_without = load_avg_matrix('ccorr','theta', 'Coupled', 'long', save = 0, p = 1)
alpha_uncoupled_without = load_avg_matrix('ccorr','alpha', 'Uncoupled', 'long', save = 0, p = 1)
beta_uncoupled_without = load_avg_matrix('ccorr','beta', 'Uncoupled', 'long', save = 0, p = 1)
theta_uncoupled_without = load_avg_matrix('ccorr','theta', 'Uncoupled', 'long', save = 0, p = 1)
alpha_control_without = load_avg_matrix('ccorr','alpha', 'Control', 'long', save = 0, p = 1)
beta_control_without = load_avg_matrix('ccorr','beta', 'Control', 'long', save = 0, p = 1)
theta_control_without = load_avg_matrix('ccorr','theta', 'Control', 'long', save = 0, p = 1)
alpha_LF_without = load_avg_matrix('ccorr','alpha', 'Leader-Follower', 'long', save = 0, p = 1)
beta_LF_without = load_avg_matrix('ccorr','beta', 'Leader-Follower', 'long', save = 0, p = 1)
theta_LF_without = load_avg_matrix('ccorr','theta', 'Leader-Follower', 'long', save = 0, p = 1)

alpha_coupled = alpha_coupled_with - alpha_coupled_without
beta_coupled = beta_coupled_with - beta_coupled_without
theta_coupled = theta_coupled_with - theta_coupled_without

alpha_uncoupled = alpha_uncoupled_with - alpha_uncoupled_without
beta_uncoupled = beta_uncoupled_with - beta_uncoupled_without
theta_uncoupled = theta_uncoupled_with - theta_uncoupled_without

alpha_control = alpha_control_with - alpha_control_without
beta_control = beta_control_with - beta_control_without
theta_control = theta_control_with - theta_control_without

alpha_LF = alpha_LF_with - alpha_LF_without
beta_LF = beta_LF_with - beta_LF_without
theta_LF = theta_LF_with - theta_LF_without
#%%
fig = plt.figure()
name = 'LF theta'
plt.title(name)
plt.imshow(theta_LF,cmap=plt.cm.seismic)
plt.clim(-0.02,0.02)
plt.colorbar()
plt.show()
fig.savefig('diff. between with and without resting/' + name + '.png')

#%% Previous results
coupled = load_avg_matrix('ccorr', 'alpha', 'Coupled', 'long', save = 0, p = 1)
uncoupled = load_avg_matrix('ccorr', 'alpha', 'Uncoupled', 'long', save = 0, p = 1)
control = load_avg_matrix('ccorr', 'alpha', 'Control', 'long', save = 0, p = 1)

contrast2 = coupled-control
contrast3 = uncoupled-control

fig = plt.figure()
name = 'ccorr coupled vs. control (alpha long)'
plt.title(name)
plt.imshow(contrast2,cmap=plt.cm.seismic)
plt.clim(-0.05,0.05)
plt.colorbar()
plt.show()
fig.savefig('avg_matrices/contrast/' + name + '.png')

fig = plt.figure()
name = 'ccorr uncoupled vs. control (alpha long)'
plt.title(name)
plt.imshow(contrast3,cmap=plt.cm.seismic)
plt.clim(-0.05,0.05)
plt.colorbar()
plt.show()
fig.savefig('avg_matrices/contrast/' + name + '.png')

#%% Plotting contrasts for ccorr
control_alpha = load_avg_matrix('ccorr', 'alpha', 'Control', 'long', save = 1)
resting_alpha = load_avg_matrix('ccorr', 'alpha', 'Resting', 'long', save = 1)
control_alpha_s = load_avg_matrix('ccorr', 'alpha', 'Control', 'short', save = 1)
resting_alpha_s = load_avg_matrix('ccorr', 'alpha', 'Resting', 'short', save = 1)
control_beta_s = load_avg_matrix('ccorr', 'beta', 'Control', 'short', save = 1)
resting_beta_s = load_avg_matrix('ccorr', 'beta', 'Resting', 'short', save = 1)
control_theta = load_avg_matrix('ccorr', 'theta', 'Control', 'long', save = 1)
resting_theta = load_avg_matrix('ccorr', 'theta', 'Resting', 'long', save = 1)

contrast1 = resting_alpha-control_alpha
contrast2 = resting_alpha_s-control_alpha_s
contrast3 = resting_beta_s-control_beta_s
contrast4 = resting_theta-control_theta

fig = plt.figure()
name = 'ccorr resting vs. control contrast (alpha long)'
plt.title(name)
plt.imshow(contrast1,cmap=plt.cm.seismic)
plt.clim(-0.35,0.35)
plt.colorbar()
plt.show()
fig.savefig('avg_matrices/contrast/ccorr/' + name + '.png')

fig = plt.figure()
name = 'ccorr resting vs. control contrast (alpha short)'
plt.title(name)
plt.imshow(contrast2,cmap=plt.cm.seismic)
plt.clim(-0.35,0.35)
plt.colorbar()
plt.show()
fig.savefig('avg_matrices/contrast/ccorr/' + name + '.png')

fig = plt.figure()
name = 'ccorr resting vs. control contrast (beta short)'
plt.title(name)
plt.imshow(contrast3,cmap=plt.cm.seismic)
plt.clim(-0.35,0.35)
plt.colorbar()
plt.show()
fig.savefig('avg_matrices/contrast/ccorr/' + name + '.png')

fig = plt.figure()
name = 'ccorr resting vs. control contrast (theta long)'
plt.title(name)
plt.imshow(contrast4,cmap=plt.cm.seismic)
plt.clim(-0.35,0.35)
plt.colorbar()
plt.show()
fig.savefig('avg_matrices/contrast/ccorr/' + name + '.png')

#%% Plotting contrasts for coh
path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Bachelor-Project"
os.chdir(path)
coupled_beta = load_avg_matrix('coh', 'beta', 'Coupled', 'long', save = 1)
control_beta = load_avg_matrix('coh', 'beta', 'Control', 'long', save = 1)
uncoupled_beta = load_avg_matrix('coh', 'beta', 'Uncoupled', 'long', save = 1)
coupled_beta_3s = load_avg_matrix('coh', 'beta', 'Coupled', '3sec', save = 1)
LF_beta_3s = load_avg_matrix('coh', 'beta', 'Leader-Follower', '3sec', save = 1)
coupled_alpha_3s = load_avg_matrix('coh', 'alpha', 'Coupled', '3sec', save = 1)
LF_alpha_3s = load_avg_matrix('coh', 'alpha', 'Leader-Follower', '3sec', save = 1)

contrast1 = coupled_beta-control_beta
contrast2 = coupled_beta_3s-LF_beta_3s
contrast3 = coupled_beta-uncoupled_beta
contrast4 = coupled_alpha_3s-LF_alpha_3s

fig = plt.figure()
name = 'coherence coupled - control contrast (beta long)'
plt.title(name)
plt.imshow(contrast1,cmap=plt.cm.seismic)
plt.clim(-0.02,0.02)
plt.colorbar()
plt.show()
fig.savefig('Visualization of significant clusters (12 pairs)/contrasts/'+ name +'.png')

fig = plt.figure()
name = 'coherence coupled - LF contrast (beta 3 sec.)'
plt.title(name)
plt.imshow(contrast2,cmap=plt.cm.seismic)
plt.clim(-0.05,0.05)
plt.colorbar()
plt.show()
fig.savefig('Visualization of significant clusters (12 pairs)/contrasts/' + name +'.png')

fig = plt.figure()
name = 'coherence coupled - uncoupled contrast (beta long)'
plt.title(name)
plt.imshow(contrast3,cmap=plt.cm.seismic)
plt.clim(-0.02,0.02)
plt.colorbar()
plt.show()
fig.savefig('Visualization of significant clusters (12 pairs)/contrasts/' + name +'.png')

fig = plt.figure()
name = 'coherence coupled - LF contrast (alpha 3 sec.)'
plt.title(name)
plt.imshow(contrast4,cmap=plt.cm.seismic)
plt.clim(-0.05,0.05)
plt.colorbar()
plt.show()
fig.savefig('Visualization of significant clusters (12 pairs)/contrasts/' + name +'.png')


