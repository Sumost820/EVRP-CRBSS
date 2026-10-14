import sys

# 💡 如果是通过命令行批量运行，切换 matplotlib 为无界面模式，防止 plt.show() 阻塞
if len(sys.argv) > 1:
    import matplotlib

    matplotlib.use('Agg')

from gurobipy import *
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import datetime

model = Model("EVRPTW")

num_s, num_c, num_depot = 0, 0, 0
xc, yc, d, e, l, s, st = [], [], [], [], [], [], []
stations_segments_length, stations_segments_slope = [], []

COPY_COUNT = 1
M = 10000

# =========================================================
# 1. 提取全局参数
# =========================================================
Q = 79.69
g_inv = 3.39
C = 200.0
h = 1.0
V = 1.0

# ⚠️ 优先从命令行参数获取算例路径（方便批量运行），若无参数则使用默认本地路径
if len(sys.argv) > 1:
    FILE_LOCATION = sys.argv[1]
else:
    FILE_LOCATION = r"data/c101_21_25.txt"

with open(FILE_LOCATION) as f:
    for line in f:
        line = line.strip()
        if not line or line[0] in "DCS":
            continue
        if '/' in line:
            parts = line.split('/')
            if len(parts) >= 2:
                try:
                    val = float(parts[1].strip())
                    if line.startswith("Q"):
                        Q = val
                    elif line.startswith("C"):
                        C = val
                    elif line.startswith("r"):
                        h = val
                    elif line.startswith("g"):
                        g_inv = val
                    elif line.startswith("v"):
                        V = val
                except ValueError:
                    pass

# =========================================================
# 2. 计算动态充电函数 (Fast) 及换电时间
# =========================================================
Q_old = 16000.0
fast_old = [(13600, 0.000023), (15200, 0.000050), (16000, 0.000150)]
normal_old = [(13600, 0.000046), (15200, 0.000094), (16000, 0.000300)]

t_normal_old = 0.0
prev_b_old = 0.0
for b_old, r_old in normal_old:
    t_normal_old += (b_old - prev_b_old) * r_old
    prev_b_old = b_old

t_normal_new = Q * g_inv
K_q = Q / Q_old
K_t = t_normal_new / t_normal_old
K_rate = K_t / K_q

scaled_fast_lengths = []
scaled_fast_slopes = []
p_b = 0.0

for b_old, r_old in fast_old:
    b_diff = b_old - p_b
    ln = b_diff * K_q
    sp = r_old * K_rate
    scaled_fast_lengths.append(ln)
    scaled_fast_slopes.append(sp)
    p_b = b_old

fast_full_charge_time = sum([ln * sp for ln, sp in zip(scaled_fast_lengths, scaled_fast_slopes)])
new_swap_time = fast_full_charge_time / 5.0

# =========================================================
# 3. 读取节点数据
# =========================================================
with open(FILE_LOCATION) as f:
    meet_empty_line = False
    for line in f.readlines()[1:]:
        line = line.strip()
        if not line:
            meet_empty_line = True
            continue

        if not meet_empty_line:
            parts = line.split()
            if parts and parts[0][0] in "DCS":
                name, type_, x_coor, y_coor, demand, ready_time, due_time, service_time = parts[0], parts[1], parts[2], \
                parts[3], parts[4], parts[5], parts[6], parts[7]

                current_lengths = scaled_fast_lengths if type_ == 'f' else []
                current_slopes = scaled_fast_slopes if type_ == 'f' else []
                dynamic_swap_time = new_swap_time if type_ == 'f' else 0.0

                repeat_count = COPY_COUNT if type_ == 'f' else 1
                for _ in range(repeat_count):
                    xc.append(float(x_coor))
                    yc.append(float(y_coor))
                    d.append(float(demand))
                    e.append(float(ready_time))
                    l.append(float(due_time))
                    s.append(float(service_time))
                    st.append(dynamic_swap_time)
                    if type_ == "f":
                        stations_segments_length.append(current_lengths)
                        stations_segments_slope.append(current_slopes)

                if type_ == "f":
                    num_s += 1
                elif type_ == "c":
                    num_c += 1
                else:
                    num_depot += 1

    xc.append(xc[0])
    yc.append(yc[0])
    d.append(d[0])
    e.append(e[0])
    l.append(l[0])
    s.append(s[0])
    st.append(st[0])

"""模型集合"""
depot_0 = [0]
depot_n = [COPY_COUNT * num_s + num_c + 1]
stations = [i for i in range(1, COPY_COUNT * num_s + 1)]
stations_0 = depot_0 + stations
customers = [i for i in range(COPY_COUNT * num_s + 1, COPY_COUNT * num_s + num_c + 1)]
customers_0 = depot_0 + customers
stations_customers = stations + customers
stations_customers_0 = depot_0 + stations_customers
stations_customers_n = stations_customers + depot_n
total = depot_0 + stations + customers + depot_n

# =========================================================
# 🚀 优化：图的稀疏化 (去除无效边)
# =========================================================
A = []
for i in stations_customers_0:
    for j in stations_customers_n:
        if i == j: continue
        if i in stations and j in stations: continue

        dist_ij = np.hypot(xc[i] - xc[j], yc[i] - yc[j])
        time_ij = dist_ij / V

        if e[i] + s[i] + time_ij > l[j]: continue
        if i in customers and j in customers and (d[i] + d[j] > C): continue

        A.append((i, j))

Dist = {(i, j): np.hypot(xc[i] - xc[j], yc[i] - yc[j]) for i, j in A}
Time = {(i, j): Dist[i, j] / V for i, j in A}

station_pairs = [(i, j) for i in stations for j in stations if i != j]
stations_segments_indices = [(i, m) for i in stations for m in range(1, len(stations_segments_length[i - 1]) + 1)]
q = {(i, m): stations_segments_length[i - 1][m - 1] for i, m in stations_segments_indices}
rho = {(i, m): stations_segments_slope[i - 1][m - 1] for i, m in stations_segments_indices}

"""决策变量 (内化边界)"""
x = model.addVars(A, vtype=GRB.BINARY, name="arcs")
w = model.addVars(stations, vtype=GRB.CONTINUOUS, name="recharging/swapping start time")
sigma = model.addVars(stations, vtype=GRB.CONTINUOUS, name="departure time from station")
tau = model.addVars(total, lb={i: e[i] for i in total}, ub={i: l[i] for i in total}, vtype=GRB.CONTINUOUS,name="arrive time")
u = model.addVars(total, lb=0.0, ub=C, vtype=GRB.CONTINUOUS, name="remaining cargo")
y = model.addVars(total, lb=0.0, ub=Q, vtype=GRB.CONTINUOUS, name="remaining battery")
z = model.addVars(stations, lb=0.0, ub=Q, vtype=GRB.CONTINUOUS, name="recharging level")
epsilon = model.addVars(stations, vtype=GRB.BINARY, name="swapping or not")
mu = model.addVars(stations, vtype=GRB.BINARY, name="recharging or not")
delta = model.addVars(stations_customers, vtype=GRB.BINARY, name="visited by vehicle_1 or not")
Delta = model.addVars(stations, vtype=GRB.CONTINUOUS, name="recharging time")
lambda_arr = model.addVars(stations_segments_indices, vtype=GRB.CONTINUOUS, name="arrive segment weight")
lambda_dep = model.addVars(stations_segments_indices, vtype=GRB.CONTINUOUS, name="departure segment weight")
theta_arr = model.addVars(stations_segments_indices, vtype=GRB.BINARY, name="arrive segment weight = 0 or not")
theta_dep = model.addVars(stations_segments_indices, vtype=GRB.BINARY, name="departure segment weight = 0 or not")
alpha = model.addVars(station_pairs, vtype=GRB.BINARY, name="visit i before j or not for swapping")
beta = model.addVars(station_pairs, vtype=GRB.BINARY, name="visit i before j or not for recharging")

"""约束条件 (化简版)"""
model.addConstrs((x.sum(i, '*') == 1 for i in customers), name="c0")
model.addConstrs((x.sum(i, '*') <= 1 for i in stations), name="c1")
model.addConstrs((x.sum('*', j) - x.sum(j, '*') == 0 for j in stations_customers), name="c2")
model.addConstrs((epsilon[i] + mu[i] == x.sum(i, '*') for i in stations), name="c3")
model.addConstrs((epsilon[i] <= delta[i] for i in stations), name='c4')
model.addConstrs(
    (y[i] == quicksum(lambda_arr[i, m] * q[i, m] for m in range(1, len(stations_segments_length[i - 1]) + 1)) for i in
     stations), name='c51')
model.addConstrs(
    (z[i] == quicksum(lambda_dep[i, m] * q[i, m] for m in range(1, len(stations_segments_length[i - 1]) + 1)) for i in
     stations), name='c52')
model.addConstrs((theta_arr[i, m] >= lambda_arr[i, m] for i, m in stations_segments_indices), name='c53')
model.addConstrs((theta_arr[i, m + 1] <= lambda_arr[i, m] for i, m in stations_segments_indices if
                  m != len(stations_segments_length[i - 1])), name='c54')
model.addConstrs((theta_dep[i, m] >= lambda_dep[i, m] for i, m in stations_segments_indices), name='c55')
model.addConstrs((theta_dep[i, m + 1] <= lambda_dep[i, m] for i, m in stations_segments_indices if
                  m != len(stations_segments_length[i - 1])), name='c56')
model.addConstrs(
    (y[j] <= y[i] - h * Dist[i, j] + 2 * Q * (1 - x[i, j]) for i in customers_0 for j in stations_customers_n if
     (i, j) in A), name="c57")
model.addConstrs(
    (y[j] <= z[i] - h * Dist[i, j] + 2 * Q * (1 - x[i, j]) for i in stations for j in stations_customers_n if
     (i, j) in A), name="c58")
model.addConstrs((z[i] >= y[i] for i in stations), name="c512")
model.addConstrs((z[i] >= Q - Q * (1 - epsilon[i]) for i in stations), name="c513")
model.addConstrs((u[j] <= u[i] - d[i] + C * (1 - x[i, j]) for i in stations_customers_0 for j in stations_customers_n if
                  (i, j) in A), name="c61")
model.addConstrs(
    (delta[j] >= delta[i] - (1 - x[i, j]) for i in stations_customers for j in stations_customers if (i, j) in A),
    name='c71')
model.addConstrs(
    (delta[j] <= delta[i] + (1 - x[i, j]) for i in stations_customers for j in stations_customers if (i, j) in A),
    name='c72')

T_MAX = l[depot_n[0]]
model.addConstrs(
    (tau[j] >= tau[i] + Time[i, j] + s[i] - T_MAX * (1 - x[i, j]) for i in customers_0 for j in stations_customers_n if
     (i, j) in A), name="c81")
model.addConstrs(
    (tau[j] >= sigma[i] + Time[i, j] - T_MAX * (1 - x[i, j]) for i in stations for j in stations_customers_n if
     (i, j) in A), name="c84")
model.addConstrs((tau[i] <= w[i] for i in stations), name="c85")
model.addConstrs((Delta[i] == quicksum((lambda_dep[i, m] - lambda_arr[i, m]) * q[i, m] * rho[i, m] for m in
                                       range(1, len(stations_segments_length[i - 1]) + 1)) for i in stations),
                 name='c86')
model.addConstrs((sigma[i] >= w[i] + Delta[i] - T_MAX * (1 - mu[i]) for i in stations), name='c87')
model.addConstrs((sigma[i] <= w[i] + Delta[i] + T_MAX * (1 - mu[i]) for i in stations), name='c88')
model.addConstrs((sigma[i] >= w[i] + st[i] - T_MAX * (1 - epsilon[i]) for i in stations), name='c89')
model.addConstrs((sigma[i] <= w[i] + st[i] + T_MAX * (1 - epsilon[i]) for i in stations), name='c810')

for k in range(num_s):
    model.addConstrs((alpha[i, j] <= epsilon[i] for i in stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] for j in
                      stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] if i != j), name='c101')
    model.addConstrs((alpha[i, j] <= epsilon[j] for i in stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] for j in
                      stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] if i != j), name='c102')
    model.addConstrs(
        (alpha[i, j] + alpha[j, i] >= epsilon[i] + epsilon[j] - 1 for i in stations[k * COPY_COUNT:(k + 1) * COPY_COUNT]
         for j in stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] if i != j), name="c103")
    model.addConstrs(
        (sigma[i] <= w[j] + T_MAX * (1 - alpha[i, j]) for i in stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] for j in
         stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] if i != j), name="c104")
    model.addConstrs((beta[i, j] <= mu[i] for i in stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] for j in
                      stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] if i != j), name='c111')
    model.addConstrs((beta[i, j] <= mu[j] for i in stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] for j in
                      stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] if i != j), name='c112')
    model.addConstrs(
        (beta[i, j] + beta[j, i] >= mu[i] + mu[j] - 1 for i in stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] for j in
         stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] if i != j), name="c113")
    model.addConstrs(
        (sigma[i] <= w[j] + T_MAX * (1 - beta[i, j]) for i in stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] for j in
         stations[k * COPY_COUNT:(k + 1) * COPY_COUNT] if i != j), name="c114")

for k in range(num_s):
    station_copies = stations[k * COPY_COUNT: (k + 1) * COPY_COUNT]
    for m in range(len(station_copies) - 1):
        curr_copy = station_copies[m]
        next_copy = station_copies[m + 1]
        model.addConstr((epsilon[next_copy] + mu[next_copy]) <= (epsilon[curr_copy] + mu[curr_copy]),
                        name=f"sym_visit_s{k}_c{m}")

"""目标函数"""
c_t, c_d1, c_d2, c_r, c_s, c_f = 1, 0.1, 0, 0, 0.1, 0
dist = quicksum(c_t * Dist[i, j] * x[i, j] for i, j in A)
departure = quicksum((c_d1 * delta[i] + c_d2 * (1 - delta[i])) * x[0, i] for i in stations_customers if (0, i) in A)
recharging_and_swapping = quicksum(
    c_r * mu[i] * (z[i] - y[i]) + c_s * epsilon[i] + c_f * (mu[i] + epsilon[i]) for i in stations)
obj = dist + departure + recharging_and_swapping
model.setObjective(obj, GRB.MINIMIZE)

# =====================================================================
# 🟢 PHASE 2: 从0开始完整求解整个模型
# =====================================================================
print("\n" + "=" * 50)
print("🟢 阶段 2: 释放变量，开始 Gurobi 完整精确求解")
print("=" * 50)

# 1. 恢复变量 x 的正常上下界 [0, 1]
for (i, j) in A:
    x[i, j].LB = 0.0
    x[i, j].UB = 1.0

# 2. 清除刚刚验证产生的内部求解状态（包括解和 IIS）
model.reset()

# 3. 恢复求解器参数 (恢复默认或设为你需要的积极参数)
model.resetParams()
model.Params.TimeLimit = 3600

# =========================================================
# 💡 修复后的日志配置：必须放在 resetParams 之后
# =========================================================
instance_name = os.path.splitext(os.path.basename(FILE_LOCATION))[0]
LOG_DIR = os.path.join(os.path.dirname(FILE_LOCATION), "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 获取当前时间生成时间戳，例如 "20260522_153022"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 将时间戳拼接到文件名中
log_file_path = os.path.join(LOG_DIR, f"{instance_name}_{timestamp}.log")

# 重新将 LogFile 参数赋给模型
model.Params.LogFile = log_file_path
# =========================================================

# 4. 再次执行优化 (由于重置了边界，这次是全图搜索)
model.optimize()

# ====================================================================
# 🚀 增强输出与绘图 (针对 Phase 2 的最终最优解)
# ====================================================================

if model.Status == GRB.OPTIMAL or model.SolCount > 0:
    print("找到可行解或最优解")

    # --- 1. 输出统计结果 ---
    print("\n" + "=" * 40)
    print("📈 模型求解统计")
    print("=" * 40)
    print(f"**求解状态**: {model.Status} ")
    print(f"**目标函数值**: {model.ObjVal:.2f}")
    print(f"**CPU 运行时间**: {model.Runtime:.2f} 秒")

    print("\n💰 **目标函数子项成本分解**")

    dist_val = dist.getValue()
    departure_val = departure.getValue()
    recharging_and_swapping_val = recharging_and_swapping.getValue()

    print(f"  1. 🛣️ 行驶距离成本 (dist): {dist_val:.2f}")
    print(f"  2. 🚌 车辆启用成本 (departure): {departure_val:.2f}")
    print(f"  3. ⚡ 充/换电服务成本 (recharging_and_swapping): {recharging_and_swapping_val:.2f}")
    print("-" * 35)
    print(f"  **目标函数值**: {dist_val + departure_val + recharging_and_swapping_val:.2f}")
    print("=" * 40)

    # --- 2. 增强路径详情输出 ---
    optimal_x = model.getAttr('X', x)
    optimal_z = model.getAttr('X', z)
    optimal_b = model.getAttr('X', y)
    optimal_t = model.getAttr('X', tau)
    optimal_w = model.getAttr('X', w)
    optimal_sigma = model.getAttr('X', sigma)
    optimal_delta = model.getAttr('X', delta)
    optimal_epsilon = model.getAttr('X', epsilon)
    optimal_mu = model.getAttr('X', mu)
    optimal_u = model.getAttr('X', u)

    adj = defaultdict(list)
    for i, j in A:
        if optimal_x[i, j] > 0.5:
            adj[i].append(j)

    routes = []
    route_vehicle_types = []
    for start in depot_0:
        for next_node in adj[start]:
            route = [start]
            current = next_node
            is_swappable = optimal_delta[current] > 0.5
            route_vehicle_types.append("可换电 (Swappable)" if is_swappable else "仅充电 (Charge Only)")
            route.append(current)
            while current not in depot_n:
                current = adj[current][0]
                route.append(current)
            routes.append(route)

    print("\n" + "=" * 80)
    print("车辆路径详细报告")
    print("=" * 80)

    for idx, route in enumerate(routes, 1):
        vehicle_type = route_vehicle_types[idx - 1]
        print(f"\n🚀 **【路径 {idx}】**")
        print(f"  **节点序列**: {' -> '.join(map(str, route))}")
        print(f"  **车辆类型**: {vehicle_type}")

        route_dist = sum(Dist.get((route[i], route[i + 1]), 0) for i in range(len(route) - 1))
        route_travel_time = sum(Time.get((route[i], route[i + 1]), 0) for i in range(len(route) - 1))
        print(f"  **总距离**: {route_dist:.2f} / {route_travel_time:.2f}")

        print("\n  🛣️ **行程记录:**")
        for i in range(len(route) - 1):
            from_node, to_node = route[i], route[i + 1]
            arr_t = optimal_t.get(to_node)
            bat_lvl = optimal_b.get(to_node)

            out_line = f"  - 从节点 {from_node} (电量: {optimal_b.get(from_node):.2f}) 驶向 {to_node} (到达时间: {arr_t:.2f}, 到达电量: {bat_lvl:.2f})"

            if to_node in stations:
                w_t = optimal_w.get(to_node)
                sig_t = optimal_sigma.get(to_node)
                total_t = sig_t - arr_t  # type: ignore
                wait_t = w_t - arr_t  # type: ignore
                act_t = sig_t - w_t  # type: ignore

                action = "🔄 换电" if optimal_epsilon.get(to_node, 0) > 0.5 else (
                    "🔋 充电" if optimal_mu.get(to_node, 0) > 0.5 else "停留")

                out_line += f"\n      **{action}** 至 {optimal_z.get(to_node):.2f}, **总耗时**: {total_t:.2f} (排队/等待: {max(0, wait_t):.2f}, 充/换电: {act_t:.2f}), 离开时间: {sig_t:.2f}"

            elif to_node in customers:
                dep_time = arr_t + s[to_node]
                out_line += f", 服务/等待至: {dep_time:.2f}, 卸货后货物量: {optimal_u.get(to_node):.2f}"

            print(out_line)

    print("\n" + "=" * 80)
    print(f"✅ 总共使用了 {len(routes)} 辆车")
    print("=" * 80)

    # --- 5. 绘制路径图 ---
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    prop_cycle = plt.cm.tab20.colors  # type: ignore
    color_map = {idx: prop_cycle[idx % len(prop_cycle)] for idx in range(len(routes))}

    for idx, route in enumerate(routes):
        ax.plot([xc[n] for n in route], [yc[n] for n in route], color=color_map[idx], linewidth=2.5, marker='o',
                markersize=5, label=f'Route {idx + 1}')

    ax.scatter(xc[0], yc[0], c='black', marker='*', s=500, label='Depot', zorder=5, edgecolors='white', linewidths=1)
    ax.scatter([xc[i] for i in customers], [yc[i] for i in customers], c='dodgerblue', s=100, label='Customer',
               zorder=5, edgecolors='white', linewidths=1)
    ax.scatter([xc[i] for i in stations], [yc[i] for i in stations], c='crimson', marker='^', s=120, label='Station',
               zorder=5, edgecolors='white', linewidths=1)

    for i in total:
        if i != depot_n[0] or i == 0:
            ax.annotate(str(i), (xc[i], yc[i]), xytext=(5, 5), textcoords='offset points', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5))

    ax.set_title("EVRPTW Optimal Routes (Full Solve)", fontsize=14, fontweight='bold')
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 💡 自动将路线图保存到日志文件夹中，命名与日志对应
    fig_path = os.path.join(LOG_DIR, f"{instance_name}_{timestamp}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"🎨 路线图已保存至: {fig_path}")

    # 💡 只有在单独手动运行脚本时才弹窗显示图片；批量运行时跳过弹窗，防止阻塞
    if len(sys.argv) == 1:
        plt.show()

else:
    print("\n❌ PHASE 2 完整求解阶段未能找到可行解。")

# 💡 释放模型资源，强制刷新缓冲流并解除文件占用
model.dispose()