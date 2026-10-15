"""EVRP-CRBSS arc-based 模型"""

from itertools import combinations
from pathlib import Path
import json

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


# 入参读取
def readInstance(instancePath, copyCount=2):
    instancePath = Path(instancePath).resolve()

    batteryCapacity, loadCapacity = 79.69, 200.0
    inverseChargeRate, consumptionRate, speed = 3.39, 1.0, 1.0
    with open(instancePath) as file:
        for line in file:
            line = line.strip()
            if not line or line[0] in "DCS":
                continue
            if '/' in line:
                parts = line.split('/')
                if len(parts) >= 2:
                    try:
                        value = float(parts[1].strip())
                        if line.startswith("Q"):
                            batteryCapacity = value
                        elif line.startswith("C"):
                            loadCapacity = value
                        elif line.startswith("r"):
                            consumptionRate = value
                        elif line.startswith("g"):
                            inverseChargeRate = value
                        elif line.startswith("v"):
                            speed = value
                    except ValueError:
                        pass

    oldBatteryCapacity = 16000.0
    fastCurve = [(13600, 0.000023), (15200, 0.000050), (16000, 0.000150)]
    normalCurve = [(13600, 0.000046), (15200, 0.000094), (16000, 0.000300)]
    normalTime, previousEnergy = 0.0, 0.0
    for endEnergy, slope in normalCurve:
        normalTime += (endEnergy - previousEnergy) * slope
        previousEnergy = endEnergy
    energyScale = batteryCapacity / oldBatteryCapacity
    timeScale = batteryCapacity * inverseChargeRate / normalTime
    rateScale = timeScale / energyScale
    segmentWidths, segmentSlopes = [], []
    previousEnergy = 0.0
    for endEnergy, slope in fastCurve:
        segmentWidths.append((endEnergy - previousEnergy) * energyScale)
        segmentSlopes.append(slope * rateScale)
        previousEnergy = endEnergy
    swapTime = sum([width * slope for width, slope in zip(segmentWidths, segmentSlopes)]) / 5.0

    xc, yc, demand, readyTime, dueTime, serviceTime, swapTimes = [], [], [], [], [], [], []
    stationWidths, stationSlopes, stationNames, nodeNames = [], [], [], []
    stationCount, customerCount, depotCount = 0, 0, 0
    with open(instancePath) as file:
        meetEmptyLine = False
        for line in file.readlines()[1:]:
            line = line.strip()
            if not line:
                meetEmptyLine = True
                continue
            if not meetEmptyLine:
                parts = line.split()
                if parts and parts[0][0] in "DCS":
                    name, nodeType, xValue, yValue, dValue, eValue, lValue, sValue = parts[:8]
                    repeatCount = copyCount if nodeType == 'f' else 1
                    for visit in range(repeatCount):
                        xc.append(float(xValue))
                        yc.append(float(yValue))
                        demand.append(float(dValue))
                        readyTime.append(float(eValue))
                        dueTime.append(float(lValue))
                        serviceTime.append(float(sValue))
                        swapTimes.append(swapTime if nodeType == 'f' else 0.0)
                        nodeNames.append(name)
                        if nodeType == 'f':
                            stationWidths.append(segmentWidths)
                            stationSlopes.append(segmentSlopes)
                    if nodeType == 'f':
                        stationCount += 1
                        stationNames.append(name)
                    elif nodeType == 'c':
                        customerCount += 1
                    else:
                        depotCount += 1
    for values in [xc, yc, demand, readyTime, dueTime, serviceTime, swapTimes]:
        values.append(values[0])
    nodeNames.append(nodeNames[0] + "_sink")

    stations = list(range(1, copyCount * stationCount + 1))
    customers = list(range(copyCount * stationCount + 1, copyCount * stationCount + customerCount + 1))
    sink = copyCount * stationCount + customerCount + 1
    stationCopies = {}
    for index, name in enumerate(stationNames):
        stationCopies[name] = list(range(1 + index * copyCount, 1 + (index + 1) * copyCount))
    return {
        "instancePath": str(instancePath), "batteryCapacity": batteryCapacity,
        "loadCapacity": loadCapacity, "consumptionRate": consumptionRate,
        "inverseChargeRate": inverseChargeRate, "speed": speed,
        "stationCount": stationCount, "customerCount": customerCount, "depotCount": depotCount,
        "xc": xc, "yc": yc, "demand": demand, "readyTime": readyTime, "dueTime": dueTime,
        "serviceTime": serviceTime, "swapTimes": swapTimes,
        "stationWidths": stationWidths, "stationSlopes": stationSlopes,
        "stations": stations, "customers": customers, "sink": sink,
        "stationCopies": stationCopies, "nodeNames": nodeNames,
    }


# 模型构建
def buildModel(data, chargerCount=2, chargeVehicleCount=None, swapVehicleCount=None,
               distanceCost=1.0, swapCost=0.1, env=None):
    # 点集
    stations, customers, sink = data["stations"], data["customers"], data["sink"]
    internalNodes = stations + customers
    allNodes = [0] + internalNodes + [sink]

    batteryCapacity, loadCapacity = data["batteryCapacity"], data["loadCapacity"]
    # depot最晚时间窗
    maxTime = data["dueTime"][0]

    stationSet, customerSet = set(stations), set(customers)
    distance = {}
    for i in [0] + internalNodes:
        for j in internalNodes + [sink]:
            if i == j or (i, j) == (0, sink):
                continue
            length = float(np.hypot(data["xc"][i] - data["xc"][j], data["yc"][i] - data["yc"][j]))    # 直线距离
            serviceDuration = 0 if i in stationSet else data["serviceTime"][i]
            earliestDeparture = max(0, data["readyTime"][i]) + serviceDuration
            # 弧的预剪枝一：时间窗不可行
            if earliestDeparture + length / data["speed"] > data["dueTime"][j] + 1e-9:
                continue
            # 弧的预剪枝二：电量不可行
            if length * data["consumptionRate"] > batteryCapacity + 1e-9:
                continue
            # 弧的预剪枝二：容量不可行
            if i in customerSet and j in customerSet:
                if data["demand"][i] + data["demand"][j] > loadCapacity + 1e-9:
                    continue  # 两个客户的总需求超过车辆容量。
            distance[i, j] = length

    arcs = list(distance)
    # 距离矩阵和能耗矩阵
    travelTime = {arc: value / data["speed"] for arc, value in distance.items()}
    energy = {arc: value * data["consumptionRate"] for arc, value in distance.items()}
    # startNodes表示能从车场出发到达的点，用于检查发车数量（其实不是很重要）
    startNodes = [i for i in internalNodes if (0, i) in distance]
    if chargeVehicleCount is None:
        chargeVehicleCount = data["customerCount"]
    if swapVehicleCount is None:
        swapVehicleCount = data["customerCount"]

    # 保存每个换电桩副本（访问次数）可以选择的充电桩
    chargers = {}
    for name in data["stationCopies"]:
        count = chargerCount
        # count = chargerCount[name] if isinstance(chargerCount, dict) else chargerCount
        chargers[name] = list(range(1, count + 1))

    # 非线性充电函数
    segments = [(i, b) for i in stations for b in range(1, len(data["stationWidths"][i - 1]) + 1)]
    q = {(i, b): data["stationWidths"][i - 1][b - 1] for i, b in segments}
    rho = {(i, b): data["stationSlopes"][i - 1][b - 1] for i, b in segments}
    fullChargeTimes = {i: sum(width * slope for width, slope in zip(data["stationWidths"][i - 1], data["stationSlopes"][i - 1])) for i in stations}

    # 用于顺序约束的索引
    stationPairs, chargerPairs, chargerKeys = [], [], []
    for name, copies in data["stationCopies"].items():
        stationPairs.extend(combinations(copies, 2))                                                      # 电站访问对：换电顺序
        chargerPairs.extend((i, j, ell) for i, j in combinations(copies, 2) for ell in chargers[name])
        chargerKeys.extend((i, ell) for i in copies for ell in chargers[name])                              # 充电桩访问对：充电顺序

    model = gp.Model("EVRP-CRBSS", env=env)
    # 决策变量
    # 弧启用变量
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    # 节点是否被S-type车访问
    delta = model.addVars(internalNodes, vtype=GRB.BINARY, name="delta")
    # 电站副本节点是否执行换电
    epsilon = model.addVars(stations, vtype=GRB.BINARY, name="epsilon")
    # 电站副本节点是否执行充电
    mu = model.addVars(stations, vtype=GRB.BINARY, name="mu")
    # 节点可以开始服务的时间（考虑时间窗+前序路线时间）
    tau = model.addVars(allNodes, lb={i: max(0, data["readyTime"][i]) for i in allNodes}, ub={i: data["dueTime"][i] for i in allNodes}, name="tau")
    # 到达节点时的剩余容量
    u = model.addVars(allNodes, lb={i: data["demand"][i] if i in customerSet else 0 for i in allNodes}, ub=loadCapacity, name="u")
    # 到达节点时的剩余电量
    y = model.addVars(allNodes, lb=0, ub=batteryCapacity, name="y")
    # y[0].LB = batteryCapacity  # 从仓库出发时满电
    # 离开电站节点时的电量
    z = model.addVars(stations, lb=0, ub=batteryCapacity, name="z")
    # 在电站节点实际开始服务的时间（考虑排队）
    w = model.addVars(stations, lb=0, ub=maxTime, name="w")
    # 离开电站的时间
    sigma = model.addVars(stations, lb=0, ub=maxTime, name="sigma")
    # 在电站节点的充电时间
    chargeTime = model.addVars(stations, lb=0, ub=fullChargeTimes, name="chargeTime")
    # 到达和离开电站的相关辅助变量
    lambdaArr = model.addVars(segments, lb=0, name="lambdaArr")
    lambdaDep = model.addVars(segments, lb=0, name="lambdaDep")
    thetaArr = model.addVars(segments, vtype=GRB.BINARY, name="thetaArr")
    thetaDep = model.addVars(segments, vtype=GRB.BINARY, name="thetaDep")
    # 电站副本是否选择某个充电桩
    v = model.addVars(chargerKeys, vtype=GRB.BINARY, name="v")
    gammaS = model.addVars(stationPairs, vtype=GRB.BINARY, name="gammaS")
    gammaC = model.addVars(chargerPairs, vtype=GRB.BINARY, name="gammaC")
    # 该路线是否为S-type车
    dispatchS = model.addVars(startNodes, vtype=GRB.BINARY, name="dispatchS")

    outgoing, incoming = {}, {}
    for i in internalNodes:
        outgoing[i] = x.sum(i, '*')
        incoming[i] = x.sum('*', i)

    # con:1--4：访问、流平衡、车队数量
    for i in customers:
        model.addConstr(outgoing[i] == 1, name=f"con1[{i}]")
    for i in stations:
        model.addConstr(outgoing[i] <= 1, name=f"con2[{i}]")
    for i in internalNodes:
        model.addConstr(incoming[i] == outgoing[i], name=f"con3[{i}]")
    for i in startNodes:
        # dispatchS[i] = delta[i] * x[0,i] 线性化
        model.addConstr(dispatchS[i] <= delta[i], name=f"con4_product_delta[{i}]")
        model.addConstr(dispatchS[i] <= x[0, i], name=f"con4_product_arc[{i}]")
        model.addConstr(dispatchS[i] >= delta[i] + x[0, i] - 1, name=f"con4_product_lower[{i}]")
    model.addConstr(gp.quicksum(dispatchS.values()) <= swapVehicleCount, name="con4_S")
    model.addConstr(gp.quicksum(x[0, i] - dispatchS[i] for i in startNodes) <= chargeVehicleCount, name="con4_C")

    # con:5、10--11、16--18：沿弧传播载荷、电量、时间和车型
    for i, j in arcs:
        energyM = batteryCapacity + energy[i, j]         # 电量相关的大M
        earliestArrival = max(0, data["readyTime"][j])
        model.addConstr(u[j] <= u[i] - data["demand"][i] + loadCapacity * (1 - x[i, j]), name=f"con5[{i},{j}]")
        if i in stationSet:
            timeM = max(0, maxTime + travelTime[i, j] - earliestArrival)
            model.addConstr(y[j] <= z[i] - energy[i, j] + energyM * (1 - x[i, j]), name=f"con11[{i},{j}]")
            model.addConstr(tau[j] >= sigma[i] + travelTime[i, j] - timeM * (1 - x[i, j]), name=f"con18[{i},{j}]")
        else:
            timeM = max(0, data["dueTime"][i] + data["serviceTime"][i] + travelTime[i, j] - earliestArrival)
            model.addConstr(y[j] <= y[i] - energy[i, j] + energyM * (1 - x[i, j]), name=f"con10[{i},{j}]")
            model.addConstr(tau[j] >= tau[i] + data["serviceTime"][i] + travelTime[i, j] - timeM * (1 - x[i, j]), name=f"con17[{i},{j}]")

        if i in internalNodes and j in internalNodes:
            model.addConstr(delta[j] - delta[i] >= x[i, j] - 1, name=f"con16_lower[{i},{j}]")
            model.addConstr(delta[j] - delta[i] <= 1 - x[i, j], name=f"con16_upper[{i},{j}]")

    # con:6--9、12--15、20--23、31--32：补能模式、充电曲线与服务时间。
    for i in stations:
        segmentIds = range(1, len(data["stationWidths"][i - 1]) + 1)
        # 到达和离开电量
        model.addConstr(y[i] == gp.quicksum(lambdaArr[i, b] * q[i, b] for b in segmentIds), name=f"con6[{i}]")
        model.addConstr(z[i] == gp.quicksum(lambdaDep[i, b] * q[i, b] for b in segmentIds), name=f"con8[{i}]")
        for b in segmentIds:
            model.addConstr(thetaArr[i, b] >= lambdaArr[i, b], name=f"con7_upper[{i},{b}]")
            model.addConstr(thetaDep[i, b] >= lambdaDep[i, b], name=f"con9_upper[{i},{b}]")
            nextArr = thetaArr[i, b + 1] if b < segmentIds[-1] else 0
            nextDep = thetaDep[i, b + 1] if b < segmentIds[-1] else 0
            model.addConstr(lambdaArr[i, b] >= nextArr, name=f"con7_lower[{i},{b}]")
            model.addConstr(lambdaDep[i, b] >= nextDep, name=f"con9_lower[{i},{b}]")

        model.addConstr(y[i] <= z[i], name=f"con12[{i}]")
        model.addConstr(z[i] >= batteryCapacity * epsilon[i], name=f"con13[{i}]")
        model.addConstr(epsilon[i] + mu[i] == outgoing[i], name=f"con14[{i}]")
        model.addConstr(epsilon[i] <= delta[i], name=f"con15[{i}]")

        model.addConstr(w[i] >= tau[i], name=f"con20[{i}]")
        # 充电时间
        model.addConstr(chargeTime[i] == gp.quicksum((lambdaDep[i, b] - lambdaArr[i, b]) * q[i, b] * rho[i, b] for b in segmentIds), name=f"con21[{i}]")
        # 这里用maxTime也可以，不过fullChargeTimes[i]会更紧
        model.addConstr(sigma[i] - w[i] - chargeTime[i] >= -fullChargeTimes[i] * (1 - mu[i]), name=f"con22_lower[{i}]")
        model.addConstr(sigma[i] - w[i] - chargeTime[i] <= maxTime * (1 - mu[i]), name=f"con22_upper[{i}]")
        swapUpperM = max(0, maxTime - data["swapTimes"][i])
        model.addConstr(sigma[i] - w[i] - data["swapTimes"][i] >= -data["swapTimes"][i] * (1 - epsilon[i]), name=f"con23_lower[{i}]")
        model.addConstr(sigma[i] - w[i] - data["swapTimes"][i] <= swapUpperM * (1 - epsilon[i]), name=f"con23_upper[{i}]")
        model.addConstr(w[i] <= sigma[i], name=f"con_station_time_bounds[{i}]")
        model.addConstr(delta[i] <= outgoing[i], name=f"con31[{i}]")
        model.addConstr(y[i] <= batteryCapacity * (epsilon[i] + mu[i]), name=f"con32_arr[{i}]")
        model.addConstr(z[i] <= batteryCapacity * (epsilon[i] + mu[i]), name=f"con32_dep[{i}]")

    # con:25--31：充电桩分配、设备排序与副本激活次序
    for name, copies in data["stationCopies"].items():
        # 电站副本是否选择充电
        for i in copies:
            model.addConstr(gp.quicksum(v[i, ell] for ell in chargers[name]) == mu[i], name=f"con_charger_assignment[{i}]")
        # 换电顺序约束
        for i, j in combinations(copies, 2):
            model.addConstr(gammaS[i, j] <= epsilon[i], name=f"con24_i[{i},{j}]")
            model.addConstr(gammaS[i, j] <= epsilon[j], name=f"con24_j[{i},{j}]")
            model.addConstr(w[j] >= sigma[i] - maxTime * (1 - gammaS[i, j]) - maxTime * (2 - epsilon[i] - epsilon[j]), name=f"con25[{i},{j}]")
            model.addConstr(w[i] >= sigma[j] - maxTime * gammaS[i, j] - maxTime * (2 - epsilon[i] - epsilon[j]), name=f"con26[{i},{j}]")
            for ell in chargers[name]:
                model.addConstr(gammaC[i, j, ell] <= v[i, ell], name=f"con27_i[{i},{j},{ell}]")
                model.addConstr(gammaC[i, j, ell] <= v[j, ell], name=f"con27_j[{i},{j},{ell}]")
                model.addConstr(w[j] >= sigma[i] - maxTime * (1 - gammaC[i, j, ell]) - maxTime * (2 - v[i, ell] - v[j, ell]), name=f"con28[{i},{j},{ell}]")
                model.addConstr(w[i] >= sigma[j] - maxTime * gammaC[i, j, ell] - maxTime * (2 - v[i, ell] - v[j, ell]), name=f"con29[{i},{j},{ell}]")
        for i, j in zip(copies, copies[1:]):
            model.addConstr(epsilon[j] + mu[j] <= epsilon[i] + mu[i], name=f"con30[{i},{j}]")


    # 保留论文的距离成本与换电费用，不增加车辆启用费用。
    model.setObjective(gp.quicksum(distanceCost * distance[arc] * x[arc] for arc in arcs) + swapCost * gp.quicksum(epsilon.values()), GRB.MINIMIZE)
    model.update()
    variables = {"x": x, "delta": delta, "epsilon": epsilon, "mu": mu,
                 "tau": tau, "u": u, "y": y, "z": z, "w": w, "sigma": sigma,
                 "chargeTime": chargeTime, "lambdaArr": lambdaArr, "lambdaDep": lambdaDep,
                 "thetaArr": thetaArr, "thetaDep": thetaDep, "v": v,
                 "gammaS": gammaS, "gammaC": gammaC, "dispatchS": dispatchS}
    return model, variables


# 画求解结果的路线图-AI可视化
def saveRoutePlot(data, variables, result, outputDir):
    figure = Figure(figsize=(14, 9))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    figure.subplots_adjust(left=0.07, right=0.67, bottom=0.08, top=0.90)
    xc, yc = data["xc"], data["yc"]
    customers = data["customers"]
    axes.scatter([xc[i] for i in customers], [yc[i] for i in customers],
                 s=45, facecolors="white", edgecolors="#253746", label="Customer", zorder=4)
    stationNodes = [copies[0] for copies in data["stationCopies"].values()]
    axes.scatter([xc[i] for i in stationNodes], [yc[i] for i in stationNodes],
                 marker="^", s=85, color="#b8b8b8", edgecolors="#555555",
                 label="Integrated station", zorder=4)
    axes.scatter(xc[0], yc[0], marker="s", s=130, color="#222222", label="Depot", zorder=5)
    for i in [0] + customers + stationNodes:
        axes.annotate(data["nodeNames"][i], (xc[i], yc[i]), xytext=(5, 5),
                      textcoords="offset points", fontsize=8, zorder=6)

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
              "#56B4E9", "#6A3D9A", "#8C564B", "#7F7F7F", "#BCBD22"]
    for routeIndex, route in enumerate(result["routes"]):
        color = colors[routeIndex % len(colors)]
        vehicleType = "S" if variables["delta"][route[1]].X > 0.5 else "C"
        operations = []
        for i in route:
            if i in variables["epsilon"]:
                mode = "swap" if variables["epsilon"][i].X > 0.5 else "charge"
                operations.append(data["nodeNames"][i] + ":" + mode)
        label = f"R{routeIndex + 1} ({vehicleType})"
        if operations:
            label += "\n" + ", ".join(operations)
        axes.plot([xc[i] for i in route], [yc[i] for i in route], color=color,
                  linewidth=1.5, alpha=0.8, label=label, zorder=2)
        for i, j in zip(route, route[1:]):
            if xc[i] == xc[j] and yc[i] == yc[j]:
                continue
            axes.annotate("", xy=(xc[i] + 0.65 * (xc[j] - xc[i]), yc[i] + 0.65 * (yc[j] - yc[i])),
                          xytext=(xc[i] + 0.40 * (xc[j] - xc[i]), yc[i] + 0.40 * (yc[j] - yc[i])),
                          arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=3)

    title = Path(data["instancePath"]).stem + f" | copyCount={result['copyCount']}, chargerCount={result['chargerCount']}"
    title += "\n" + result["status"]
    if result["hasSolution"]:
        title += f" | Best solution={result['objective']:.2f}, gap={result['gap']:.2%}"
    else:
        axes.text(0.5, 0.5, "No feasible solution found", transform=axes.transAxes,
                  ha="center", fontsize=14, bbox=dict(facecolor="white", alpha=0.9))
    if result["issues"]:
        title += "\nWarning: solution validation issues; see result.json"
    axes.set_title(title, fontsize=12)
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_aspect("equal", adjustable="datalim")
    axes.margins(0.08)
    axes.grid(alpha=0.15)
    axes.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9,
                title="C: charging-only EV | S: swap-capable EV")
    figure.savefig(outputDir / "routes.png", dpi=220, bbox_inches="tight")
    figure.clear()


# 模型求解 - test入口
def runModel(instancePath, copyCount=2, chargerCount=2, chargeVehicleCount=None,
             swapVehicleCount=None, timeLimit=60, distanceCost=1.0, swapCost=0.1, outputDir=None):
    # 入参读取
    data = readInstance(instancePath, copyCount)
    # 结果文件夹（results）
    resultDir = Path(outputDir).resolve()

    # 算例输出结果文件夹（c101_21_25_copyCount1_chargerCount1）
    # 格式一：各电站充电桩数量相同
    chargerLabel = str(chargerCount)
    # # 格式二：各电站充电桩数量不同，字典格式
    # if isinstance(chargerCount, dict):
    #     chargerLabel = "-".join(f"{name}x{count}" for name, count in sorted(chargerCount.items()))
    outputDir = resultDir / f"{Path(instancePath).stem}_copyCount{copyCount}_chargerCount{chargerLabel}"
    outputDir.mkdir(parents=True, exist_ok=True)

    # 求解模型时防止之前跑的结果干扰
    for fileName in ["gurobi.log", "result.json", "routes.png"]:
        (outputDir / fileName).unlink(missing_ok=True)

    # 启动Gurobi环境
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        # 构建模型
        model, variables = buildModel(data, chargerCount, chargeVehicleCount, swapVehicleCount, distanceCost, swapCost, env)
        with model:
            model.Params.OutputFlag = 1
            model.Params.LogToConsole = 1
            model.Params.DisplayInterval = 5     # 5秒更新求解进度
            model.Params.TimeLimit = timeLimit
            model.Params.Threads = 0             # 线程拉满
            model.Params.MIPGap = 0.01
            model.Params.LogFile = str(outputDir / "gurobi.log")   # 存日志
            model.optimize()
            runtime = model.Runtime

            statusNames = {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
                           GRB.TIME_LIMIT: "TIME_LIMIT", GRB.INF_OR_UNBD: "INF_OR_UNBD",
                           GRB.UNBOUNDED: "UNBOUNDED"}

            # 求解结果 - 以字典返回（后续最好可以改成标准化的对象）
            result = {
                "instancePath": data["instancePath"],
                "outputDir": str(outputDir),
                "status": statusNames.get(model.Status, str(model.Status)),
                "hasSolution": model.SolCount > 0,
                "objective": None,
                "lowerBound": None,
                "gap": None,
                "runtime": runtime,
                "arcCount": len(variables["x"]),
                "variableCount": model.NumVars,
                "constraintCount": model.NumConstrs,
                "copyCount": copyCount,
                "chargerCount": chargerCount,
                "timeLimit": timeLimit,
                "chargeVehicleCount": data["customerCount"] if chargeVehicleCount is None else chargeVehicleCount,
                "swapVehicleCount": data["customerCount"] if swapVehicleCount is None else swapVehicleCount,
                "distanceCost": distanceCost, "swapCost": swapCost,
                "routes": [], "issues": [], "iisConstraints": [], "iisError": None,
            }

            if abs(model.ObjBound) < GRB.INFINITY:
                result["lowerBound"] = model.ObjBound
            if model.SolCount > 0:
                result["objective"], result["gap"] = model.ObjVal, model.MIPGap
                selectedArcs = [(i, j) for (i, j), value in variables["x"].items() if value.X > 0.5]
                remainingArcs = set(selectedArcs)
                nextNodes = {i: j for i, j in selectedArcs if i != 0}
                for start, nextNode in selectedArcs:
                    if start != 0:
                        continue
                    route = [0, nextNode]
                    remainingArcs.discard((0, nextNode))
                    while nextNode != data["sink"] and nextNode in nextNodes:
                        currentNode = nextNode
                        nextNode = nextNodes[currentNode]
                        remainingArcs.discard((currentNode, nextNode))
                        if nextNode in route:
                            break
                        route.append(nextNode)
                    result["routes"].append(route)
                    if route[-1] != data["sink"]:
                        result["issues"].append("路线未正常返回仓库")
                if remainingArcs:
                    result["issues"].append("存在脱离仓库路线的弧或站点环")
                for node, value in variables["y"].items():
                    if value.X > data["batteryCapacity"] + 1e-5:
                        result["issues"].append(f"节点 {node} 电量 {value.X:.4f} 超过电池容量")
                for node, value in variables["u"].items():
                    if value.X > data["loadCapacity"] + 1e-5:
                        result["issues"].append(f"节点 {node} 载荷 {value.X:.4f} 超过载荷容量")
            if model.Status == GRB.INFEASIBLE:
                model.Params.TimeLimit = min(timeLimit, 30)
                try:
                    model.computeIIS()
                    result["iisConstraints"] = [constraint.ConstrName for constraint in model.getConstrs()
                                                if constraint.IISConstr]
                except gp.GurobiError as error:
                    result["iisError"] = str(error)

            # 存求解结果
            (outputDir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            # 画结果图
            saveRoutePlot(data, variables, result, outputDir)
            return result
