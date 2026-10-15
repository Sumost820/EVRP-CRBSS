"""批测"""

from pathlib import Path

import pytest
from gurobiModel import runModel


# 路径和参数
projectDir = Path(__file__).resolve().parent
dataDir = projectDir / "data"
instancePaths = sorted(dataDir.glob("*.txt"))
resultDir = projectDir / "results"
copyCount = 2                # 每个物理电站的复制数，表示可以被访问的次数（如果充电，每次访问后选择1个充电桩）
chargerCount = 2             # 每个电站的充电桩数量
chargeVehicleCount = 10      # C-type 车辆数（上限），None 表示不限制
swapVehicleCount = 10        # S-type 车辆数（上限），None 表示不限制
timeLimit = 30


@pytest.mark.parametrize("instancePath", instancePaths, ids=[path.stem for path in instancePaths])
def testGurobiModel(instancePath):
    result = runModel(
        instancePath,
        copyCount=copyCount,
        chargerCount=chargerCount,
        chargeVehicleCount=chargeVehicleCount,
        swapVehicleCount=swapVehicleCount,
        timeLimit=timeLimit,
        outputDir=resultDir,
    )

    assert result["status"] in ["OPTIMAL", "TIME_LIMIT"], f"{result['status']}，诊断：{result['iisConstraints']}"
    assert result["hasSolution"], "时限内未找到可行解"
    assert not result["issues"], result["issues"]
