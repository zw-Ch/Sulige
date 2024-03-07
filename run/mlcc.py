import copy
import os
import pickle
import numpy as np
import pandas as pd
import os.path as osp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import xlrd
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


def tran_locate(loc, range_len, dgo):
    if dgo[loc] != 0:
        return loc
    for i in range(range_len):
        if dgo[loc + i] != 0:
            return loc + i
        elif dgo[loc - i] != 0:
            return loc - i
    raise ValueError("请增大区间长度range_len!")


def get_class(xls_ad, name):
    new_cj_hw = pd.read_excel(osp.join(xls_ad, 'new_cj_su_hwBasicinfo.xls'), sheet_name='水平井')
    new_cj_vw = pd.read_excel(osp.join(xls_ad, 'new_cj_su_vwBasicinfo.xls'), sheet_name='常规井')
    hw = new_cj_hw.loc[:, '井名'].values.reshape(-1)
    vw = new_cj_vw.loc[:, '井名'].values.reshape(-1)

    if np.isin([name], hw)[0]:
        bi_class = '水平井'
    elif np.isin([name], vw)[0]:
        bi_class = '直井'
    else:
        bi_class = '既非水平井，也非直井'

    dyc = pd.read_csv(osp.join(xls_ad, '动态分类1.csv'))
    dyc_name = dyc.loc[:, '井号'].values.reshape(-1)
    dyc_cate = dyc.loc[:, '分类'].values.reshape(-1)
    dy_idx = np.argwhere(dyc_name == name).reshape(-1)
    if dy_idx.shape[0] == 0:
        dyc_class = '不存在于动态分类.csv'
    else:
        dy_idx = dy_idx[0]
        dyc_class = dyc_cate[dy_idx]

    return bi_class, dyc_class


def drop_zero():
    pass


def get_sheet(xls_ad, name):
    name_list = os.listdir(xls_ad)
    num_xls = len(name_list)
    df = None
    for i in range(num_xls):
        xls_name = name_list[i]
        xls = xlrd.open_workbook(osp.join(xls_ad, xls_name), on_demand=True)
        sheet_names = np.array(xls.sheet_names())
        if np.argwhere(sheet_names == name).reshape(-1).shape[0] != 0:
            df = pd.read_excel(osp.join(xls_ad, xls_name), sheet_name=name)
            break
    if df is None:
        raise ValueError('该油井不在已记录的动态数据中！')
    return df


class Fun(object):
    def __init__(self):
        super(Fun, self).__init__()

    def exp(self, t, q_i, D_i):
        q = q_i * np.exp(-D_i * t)
        return q

    def depletion(self, t, q_i, D_i):
        q = q_i / (1 + 0.5 * D_i * t) ** 2
        return q

"""
默认文件存储路径格式满足：
——dataset
————class_info
——————目前气井分类.xlsx
————product_data
——————120（1）.xls
——————120（2）.xls
——————......
——run
————mlcc.py （本文件所在路径）

增大nat_len/mea_len/aba_len，红色/紫色/黑色虚线将向左移动，减小将向右移动
若黑色虚线（经济废弃段开始时刻）不位于四条虚线的最右边，请适当增大aba_len；
若红色虚线（自然连续生产段结束时刻）位于紫色虚线右侧，请适当增大nat_len
若紫色虚线（措施连续生产段结束时刻）位于红色虚线右侧，请适当增大mea_len；若位于黑色虚线右侧，请适当减小mea_len

"""
data_ad = '../dataset'              # 文件所在路径（包含：井号、日期、生产时间(h)、套压(MPa)、油压(MPa)、日产气量(10⁴m³/d)）
name = '桃2-6-25'                    # 油井名称
# start_time = 90                       # 投产初期段结束时刻，default=90
nat_len = 10                    # 满足自然连续条件的长度，用于定位：自然连续生产段结束时刻，default=100
mea_len = 10                      # 满足措施连续条件的长度，用于定位：措施连续生产段结束时刻，default=70
aba_len = 50                     # 满足经济废弃条件的长度，用于定位：间歇生产段结束时刻/经济废弃段开始时刻，default=150
rrg = 45.65                         # 最低经济气量最大值，default=45.65
thre_dgo = 0.25                     # 认为日产气量小于该值的油井没有生产潜力，default=0.25
range_len = 200              # 用于最终调整定位位置的区间，选择其中日产气量非零的位置

rho_L_new = 1000                  # 若措施为”泡排“，修改后的液体密度(kg/m^3)，范围在500~1500，default=1000
r_new = 0.031                      # 若措施为”速度管柱“，修改后的油管内径(mm)，default=0.031

# name = '桃2-6-25'                # 示例：措施-速度管柱    （在原始参数设置下，具有较好效果）
# name = '苏14-12-50'              # 示例：措施-泡排       （在原始参数设置下，具有较好效果）
save_txt = True

"""
读取所有文件数据
"""
# with open(osp.join(data_ad, 'product_all.pkl'), 'rb') as f:
#     product_all = pickle.load(f)
# ops = product_all.ops                           # 所有油压
# dgos = product_all.dgos                         # 所有油井的日产气量
# cps = product_all.cps
# names = product_all.names                       # 所有油井的名称
# dates = product_all.dates                       # 所有油井的生产日期
# idx = np.argwhere(np.array(names) == name).reshape(-1)
# if idx.shape[0] == 0:
#     raise ValueError('该井不存在！')
# else:
#     idx = idx[0]                                # 该油井的索引位置
# op = np.array(ops[idx]).astype(float)           # 油压
# dgo = np.array(dgos[idx]).astype(float)         # 日产气量
# cp = np.array(cps[idx]).astype(float)
# date = np.array(dates[idx])                     # 生产日期

"""
读取数据
"""
df = get_sheet(osp.join(data_ad, 'product_data'), name)             # 动态数据，包括日期、产量、套压、油压
op = df.loc[:, '油压(MPa)'].values.reshape(-1)
cp = df.loc[:, '套压(MPa)'].values.reshape(-1)
dgo = df.loc[:, '日产气量(10⁴m³/d)'].values.reshape(-1)
date = df.loc[:, '日期'].values.reshape(-1)
dgo_ori = copy.deepcopy(dgo)                    # 拷贝原始的日产气量
if dgo.shape[0] < 365 * 3:
    raise ValueError('日产量时间少于3年，请选择具有更长开采时间的油井！')

bi_class, dyc_class = get_class(osp.join(data_ad, 'class_info'), name)
print("{}，{}".format(bi_class, dyc_class))

r = 0.062
A = np.pi * (r ** 2)
Z = 0.85
T = 383.15
rho_L = 1074

# 气体密度
rho_g = 3484.4 * 0.56 * op / 0.85 / 383.15

# 计算临界携液流量
u_g = 0.5 * np.power(((rho_L - rho_g) * 0.06 / (rho_g ** 2)), 1/4)
q_sc = (2.5 * 1e8 * (A * op * u_g) / (Z * T)) / 1e4
q_sc_other = pd.read_excel(osp.join(data_ad, 'static_data', '临界携液流量.xlsx'), sheet_name='临界携液流量')
print("\n{}".format(name))

"""
定位投产初期结束时刻          修改处
"""
cp_diff = np.diff(cp)
if (dyc_class == "Ⅰ类井") | (dyc_class == 'Ⅱ类井'):
    start_time = 120
elif dyc_class == "Ⅲ类井":
    start_time = 90
else:
    raise TypeError("Unknown type of 'dyc_class")
for i in range(start_time, cp_diff.shape[0]):
    if cp_diff[i] > 0.04:
        start_time = i
        break

"""
定位自然连续生产段结束时刻
"""
idx_cri = np.argwhere(dgo < q_sc).reshape(-1)
for i_cri in range(idx_cri.shape[0]):
    if idx_cri[i_cri] <= start_time:
        continue
    idx_cri_one = idx_cri[i_cri: (i_cri + nat_len + 1)]
    a = np.diff(idx_cri_one)
    if (a == np.array([1] * nat_len)).all():
        break
date_false = date[idx_cri[i_cri]]
dgo_false = dgo[idx_cri[i_cri]]
op_false = op[idx_cri[i_cri]]
print('\n生产日期: {}  临界日期：{}\n产量: {}  油压: {}'.format(date[0], date_false, dgo_false, op_false))

# 除去nan值
idx_nan = ~np.isnan(q_sc)
q_sc = q_sc[idx_nan]
dgo = dgo[idx_nan]
cp = cp[idx_nan]

# 使用函数拟合日产气量
n = 0.5
Fun = Fun()
t = np.arange(1, 1 + dgo.shape[0])
if n == 0:
    func = Fun.exp
    popt, pcov = curve_fit(func, t, dgo)
elif n == 0.5:
    func = Fun.depletion
    popt, pcov = curve_fit(func, t, dgo)
else:
    raise TypeError('!')
data_fit = func(t, *popt)

# 计算EUR
if n == 0:
    eur = popt[0] / popt[1]                     # 指数递减
elif n == 0.5:
    eur = 2 * popt[0] / popt[1]                 # 衰竭递减
else:
    raise TypeError('Unknown n')

dgo_sum_ = []
for i in range(dgo.shape[0]):
    dgo_sum_one = np.sum(dgo[:i])
    dgo_sum_.append(dgo_sum_one)
dgo_sum = np.array(dgo_sum_)[-1]
eur = eur - dgo_sum

# 寻找没有生产潜力的节点
dgo_behind_sum = []
eur_new = []
for i in range(dgo.shape[0]):
    dgo_behind, dgo_front = dgo[i:], dgo[:i]
    dgo_behind_sum, dgo_front_sum = np.sum(dgo_behind), np.sum(dgo_front)
    eur_new_one = eur - dgo_front_sum
    eur_new.append(eur_new_one)
eur_new = np.array(eur_new)

idx_aba = np.argwhere((eur_new < rrg) & (dgo < 0.25)).reshape(-1)
if idx_aba.shape[0] == 0:
    item_aba = dgo.shape[0] - 10
else:
    for i_aba in range(idx_aba.shape[0] - 1):           # 修改处
        if idx_aba[i_aba] <= start_time:
            continue
        idx_aba_one = idx_aba[i_aba: (i_aba + aba_len + 1)]
        a = np.diff(idx_aba_one)
        if a.shape[0] < aba_len:
            raise ValueError("aba_len过大，请适当减小")
        if (a == np.array([1] * aba_len)).all():
            break
    item_aba = idx_aba[i_aba]

meas_info = pd.read_excel(osp.join(data_ad, 'class_info', '目前气井分类.xlsx'), sheet_name='目前措施')
meas = meas_info.loc[:, '措施'].values.reshape(-1)
idx_mea = np.argwhere(meas_info.loc[:, '井号'].values.reshape(-1) == name).reshape(-1)
if idx_mea.shape[0] == 0:
    raise ValueError('没有在目前措施中找到该油井')
mea = meas[idx_mea[0]]
if mea == '泡排':
    u_g = 0.5 * np.power(((rho_L_new - rho_g) * 0.06 / (rho_g ** 2)), 1 / 4)
    q_sc = (2.5 * 1e8 * (A * op * u_g) / (Z * T)) / 1e4
elif mea == '柱塞':
    pass
elif mea == '速度管柱':
    A_new = np.pi * (r_new ** 2)
    q_sc = (2.5 * 1e8 * (A_new * op * u_g) / (Z * T)) / 1e4
elif mea == '压缩机气举':
    pass
elif mea == '脱硫':
    pass
elif mea == '密闭抽吸':
    pass
elif mea == '同步回转':
    pass
elif mea == '电动针阀':
    pass
elif mea == '气动阀':
    pass

# 除去nan值
idx_nan = ~np.isnan(q_sc)
q_sc = q_sc[idx_nan]
dgo = dgo[idx_nan]
cp = cp[idx_nan]

idx_mea = np.argwhere(dgo < q_sc).reshape(-1)
for i_mea in range(idx_mea.shape[0]):
    if i_mea <= start_time:
        continue
    idx_mea_one = idx_mea[i_mea: (i_mea + mea_len + 1)]
    a = np.diff(idx_mea_one)
    if a.shape[0] < mea_len:
        raise ValueError("mea_len过大，请适当减小")
    try:
        if (a == np.array([1] * mea_len)).all():
            break
    except:
        if (a == np.array([1] * a.shape[0])).all():
            break

# 绘制y = x 曲线
t = np.linspace(0, np.max(np.concatenate((q_sc, dgo), axis=0)), 100)
if name[0] == '苏':
    idx = name.index('苏')
    name_new = name[0: idx] + 'su' + name[idx + 1:]
elif name[0] == '桃':
    idx = name.index('桃')
    name_new = name[0: idx] + 'tao' + name[idx + 1:]
else:
    raise TypeError('Unknown type of name[0]')

start_date = start_time
criti_date = idx_cri[i_cri]
measu_date = idx_mea[i_mea]
aband_date = item_aba

start_date = tran_locate(start_date, range_len, dgo)
criti_date = tran_locate(criti_date, range_len, dgo)
measu_date = tran_locate(measu_date, range_len, dgo)
aband_date = tran_locate(aband_date, range_len, dgo)

# plt.figure(figsize=(18, 18))
# plt.plot(dgo, label="Gas production")
# y_min, y_max = 0, np.max(t) * 0.5
# plt.vlines(start_time, y_min, y_max, color='green', lw=8, label='Start date', ls='dashed')
# plt.vlines(idx_cri[i_cri], y_min, y_max, color='red', lw=8, label='Critical date', ls='dashed')
# plt.vlines(idx_mea[i_mea], y_min, y_max, color='purple', lw=8, label='Measure date', ls='dashed')
# plt.vlines(item_aba, y_min, y_max, color='black', lw=8, label='Abandon date', ls='dashed')
#
# plt.title(name_new, fontsize=45)
# plt.legend(fontsize=40)
# plt.xticks(fontsize=35)
# plt.yticks(fontsize=35)
#
# plt.figure(figsize=(18, 18))
# plt.scatter(q_sc, dgo)
# plt.plot(t, t, c='black')
# # plt.ylabel('气体实际产量', fontsize=45)
# plt.ylabel('Daily gas production (10⁴m³/d)', fontsize=45)
# plt.yticks(fontsize=35)
# plt.xlabel('Critical Liquid-carrying flow rate (10⁴m³/d)', fontsize=45)
# plt.xticks(fontsize=35)
# plt.title(name_new, fontsize=45)


"""
以下 by 田力
"""
plt.figure()
plt.plot(cp)

# 绘制套压油压对比图
fig, ax1 = plt.subplots(figsize=(18, 18))
plt.sca(ax1)
plt.plot(dgo, "g--", label="Gas production", alpha=0.5)
y_min, y_max = 0, np.max(t) * 0.5
plt.vlines(start_date, y_min, y_max, color='green', lw=8, label='Start date', ls='dashed')

text1_1 = f"持续时间: ${start_date}days$"
text1_2 = f"累积产气量: ${round(dgo_sum_[start_date], 5)}×10^4m^3$"
text1_3 = f"产气量: ${round(dgo[start_date], 5)}×10^4m^3$"
text1_4 = f"套压: ${round(cp[start_date])}MPa$"
# plt.text(start_date, 8, text1, size=20, family="Times new roman", color="black", style='italic', weight = "light")
plt.text(start_date-50, y_max+0.15, text1_3, size=15, color="black", style='italic', weight = "light")
plt.text(start_date-50, y_max+0.1, text1_4, size=15, color="black", style='italic', weight = "light")
plt.text(start_date-50, y_max+0.05, text1_2, size=15, color="black", style='italic', weight = "light")
plt.text(start_date-50, y_max, text1_1, size=15, color="black", style='italic', weight = "light")
# plt.text(start_date-50, y_max-0.5, text2, size=20, family="Times new roman", color="black", style='italic', weight = "light")

plt.vlines(criti_date, y_min, y_max, color='red', lw=8, label='Critical date', ls='dashed')
# text3 = f"daytime: {date[criti_date]}"
text2_1 = f"持续时间: ${criti_date-start_date}days$"
text2_2 = f"累积产气量: ${round(dgo_sum_[criti_date],2)}×10^4m^3$"
text2_3 = f"产气量: ${round(dgo[criti_date], 2)}×10^4m^3$"
text2_4 = f"套压: ${round(cp[criti_date])}MPa$"
# plt.text(criti_date, 8, text3, size=20, family="Times new roman", color="black", style='italic', weight = "light")
plt.text(criti_date, y_max+0.15, text2_3, size=15, color="black", style='italic', weight = "light")
plt.text(criti_date, y_max+0.1, text2_4, size=15, color="black", style='italic', weight = "light")
plt.text(criti_date, y_max+0.05, text2_2, size=15, color="black", style='italic', weight = "light")
plt.text(criti_date, y_max, text2_1, size=15, color="black", style='italic', weight = "light")


plt.vlines(measu_date, y_min, y_max, color='purple', lw=8, label='Measure date', ls='dashed')
# text5 = f"daytime: {date[measu_date]}"
text3_1 = f"持续时间: ${measu_date-item_aba}days$"
text3_2 = f"累积产气量: ${round(dgo_sum_[measu_date],2)}×10^4m^3$"
text3_3 = f"产气量: ${round(dgo[measu_date],2)}×10^4m^3$"
text3_4 = f"套压: ${round(cp[measu_date])}MPa$"
# plt.text(measu_date, 8, text5, size=20, family="Times new roman", color="black", style='italic', weight = "light")
plt.text(measu_date-30, y_max+0.15, text3_3, size=15, color="black", style='italic', weight = "light")
plt.text(measu_date-30, y_max+0.1, text3_4, size=15, color="black", style='italic', weight = "light")
plt.text(measu_date-30, y_max+0.05, text3_2, size=15, color="black", style='italic', weight = "light")
plt.text(measu_date-30, y_max, text3_1, size=15, color="black", style='italic', weight = "light")


plt.vlines(aband_date, y_min, y_max, color='black', lw=8, label='Abandon date', ls='dashed')
# text7 = f"daytime: {date[aband_date]}"
text4_1 = f"持续时间: ${aband_date-criti_date}days$"
text4_2 = f"累积产气量: ${round(dgo_sum_[aband_date], 4)}×10^4m^3$"
text4_3 = f"产气量: ${round(dgo[aband_date], 4)}×10^4m^3$"
text4_4 = f"套压: ${round(cp[aband_date])}MPa$"
# plt.text(aband_date, 8, text7, size=20, family="Times new roman", color="black", style='italic', weight = "light")
plt.text(aband_date-30, y_max+0.15, text4_3, size=15, color="black", style='italic', weight = "light")
plt.text(aband_date-30, y_max+0.1, text4_4, size=15, color="black", style='italic', weight = "light")
plt.text(aband_date-30, y_max+0.05, text4_2, size=15, color="black", style='italic', weight = "light")
plt.text(aband_date-30, y_max, text4_1, size=15,  color="black", style='italic', weight = "light")
plt.title(name_new, fontsize=45)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("日期", fontsize=20)
plt.ylabel("日产气量($10^4m^3/d$)", fontsize=20)
# 绘制子坐标
plt.sca(ax1.twinx())
cp = cp[idx_nan]
plt.plot(cp, "y--", label="Casing pressure")
plt.legend(fontsize=20,loc="upper left")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("套压(MPa)", fontsize=20)

plt.figure(figsize=(18, 18))
plt.scatter(q_sc, dgo)
plt.plot(t, t, c='black')
# plt.ylabel('气体实际产量', fontsize=45)
plt.ylabel('Daily gas production (10⁴m³/d)', fontsize=45)
plt.yticks(fontsize=35)
plt.xlabel('Critical Liquid-carrying flow rate (10⁴m³/d)', fontsize=45)
plt.xticks(fontsize=35)
plt.title(name_new, fontsize=45)


if save_txt:
    info_txt_address = osp.join('../result', "mlcc.txt")
    info_df_address = osp.join('../result', "mlcc.csv")
    f = open(info_txt_address, 'a')
    if osp.getsize(info_txt_address) == 0:
        f.write('油井名称 类1(水平井/直井) 类2(I/II/III) '
                '自然连续-开始 自然连续-累产 自然连续-产量 自然连续-套压 '
                '措施连续-开始 措施连续-累产 措施连续-产量 措施连续-套压 '
                '间歇生产-开始 间歇生产-累产 间歇生产-产量 间歇生产-套压 '
                '经济废弃-开始 经济废弃-累产 经济废弃-产量 经济废弃-套压\n')          # 中文
        # f.write('name bi_class dyc_class '
        #         'start_date sd_dgos sd_dgo sd_cp '
        #         'critical_date cd_dgos cd_dgo cd_cp '
        #         'measure_date md_dgos md_dgo md_cp '
        #         'abandon_date ad_dgos ad_dgo ad_cp\n')                    # 英文
    f.write(str(name) + "  ")
    f.write(str(bi_class) + "  ")
    f.write(str(dyc_class) + "  ")
    f.write(str(start_date) + "  ")
    f.write(str(round(dgo_sum_[start_date], 4)) + "  ")
    f.write(str(round(dgo[measu_date], 4)) + "  ")
    f.write(str(round(cp[start_date], 4)) + "  ")
    f.write(str(criti_date) + "  ")
    f.write(str(round(dgo_sum_[criti_date], 4)) + "  ")
    f.write(str(round(dgo[criti_date], 4)) + "  ")
    f.write(str(round(cp[criti_date], 4)) + "  ")
    f.write(str(measu_date) + "  ")
    f.write(str(round(dgo_sum_[measu_date], 4)) + "  ")
    f.write(str(round(dgo[measu_date], 4)) + "  ")
    f.write(str(round(cp[measu_date], 4)) + "  ")
    f.write(str(aband_date) + "  ")
    f.write(str(round(dgo_sum_[aband_date], 4)) + "  ")
    f.write(str(round(dgo[aband_date], 4)) + "  ")
    f.write(str(round(cp[aband_date], 4)) + "  ")
    f.write("\n")
    f.close()

    info = np.loadtxt(info_txt_address, dtype=str)
    columns = info[0, :].tolist()
    values = info[1:, :]
    info_df = pd.DataFrame(values, columns=columns)
    info_df.to_csv(info_df_address)


print()
plt.show()
print()
