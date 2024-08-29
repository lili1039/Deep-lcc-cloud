# import asyncio
# import numpy as np
# import pickle
# import time
# import redis
# import math
# import websockets

# def StopCriteria(k,rs,n_cav,timestep):
#     error_pri1 = 0
#     tolerance_pri1 = 0

#     error_dual1 = 0
#     tolerance_dual1 = 0

#     error_pri2 = 0
#     tolerance_pri2 = 0

#     error_dual2 = 0
#     tolerance_dual2 = 0

#     error_pri3 = 0
#     tolerance_pri3 = 0

#     error_dual3 = 0
#     tolerance_dual3 = 0

#     error_pri4 = 0
#     tolerance_pri4 = 0

#     error_dual4 = 0
#     tolerance_dual4 = 0    

#     for i in range(n_cav):
#         while True:
#             error_tolerance_bytes = rs.mget(f'error_tolerance_{i}_{timestep}_{k}')[0]
#             if error_tolerance_bytes != None:
#                 break
#         error_tolerance = pickle.loads(error_tolerance_bytes)

#         error_pri1 = error_pri1 + error_tolerance[0][0]
#         tolerance_pri1 = _pri1 + error_tolerance[0][1]
#         error_dual1 = error_dual1 + error_tolerance[0][2]
#         tolerance_dual1 = tolerance_dual1 + error_tolerance[0][3]

#     if error_pri1 > tolerance_pri1 or error_dual1 > tolerance_dual1:
#         return False
    
#     for i in range(n_cav-1):
#         while True:
#             error_tolerance_bytes = rs.mget(f'error_tolerance_{i}_{timestep}_{k}')[0]
#             if error_tolerance_bytes != None:
#                 break
#         error_tolerance = pickle.loads(error_tolerance_bytes)

#         error_pri2 = error_pri2 + error_tolerance[1][0]
#         tolerance_pri2 = tolerance_pri2 + error_tolerance[1][1]
#         error_dual2 = error_dual2 + error_tolerance[1][2]
#         tolerance_dual2 = tolerance_dual2 + error_tolerance[1][3]

#     if error_pri2 > tolerance_pri2 or error_dual2 > tolerance_dual2:
#         return False
    
#     for i in range(n_cav):
#         while True:
#             error_tolerance_bytes = rs.mget(f'error_tolerance_{i}_{timestep}_{k}')[0]
#             if error_tolerance_bytes != None:
#                 break
#         error_tolerance = pickle.loads(error_tolerance_bytes)

#         error_pri3 = error_pri3 + error_tolerance[2][0]
#         tolerance_pri3 = tolerance_pri3 + error_tolerance[2][1]
#         error_dual3 = error_dual3 + error_tolerance[2][2]
#         tolerance_dual3 = tolerance_dual3 + error_tolerance[2][3]

#     if error_pri3 > tolerance_pri3 or error_dual3 > tolerance_dual3:
#         return False   

#     for i in range(n_cav):
#         while True:
#             error_tolerance_bytes = rs.mget(f'error_tolerance_{i}_{timestep}_{k}')[0]
#             if error_tolerance_bytes != None:
#                 break
#         error_tolerance = pickle.loads(error_tolerance_bytes)
                
#         error_pri4 = error_pri4 + error_tolerance[3][0]
#         tolerance_pri4 = tolerance_pri4 + error_tolerance[3][1]
#         error_dual4 = error_dual4 + error_tolerance[3][2]
#         tolerance_dual4 = tolerance_dual4 + error_tolerance[3][3]

#     if error_pri4 > tolerance_pri4 or error_dual4 > tolerance_dual4:
#         return False
    
#     return True


#=================================================================
# import asyncio
# import pickle

# async def fetch_error_tolerance(rs, i, timestep, k):
#     while True:
#         error_tolerance_bytes = rs.mget(f'error_tolerance_{i}_{timestep}_{k}')[0]
#         if error_tolerance_bytes is not None:
#             return pickle.loads(error_tolerance_bytes)
#         await asyncio.sleep(0)  # 让出执行权，避免阻塞

# async def check_criteria(error_tolerance, idx, error_pri, tolerance_pri, error_dual, tolerance_dual):
#     error_pri[idx] += error_tolerance[idx][0]
#     tolerance_pri[idx] += error_tolerance[idx][1]
#     error_dual[idx] += error_tolerance[idx][2]
#     tolerance_dual[idx] += error_tolerance[idx][3]

#     if error_pri[idx] > tolerance_pri[idx] or error_dual[idx] > tolerance_dual[idx]:
#         return False
#     return True

# async def StopCriteria(k, rs, n_cav, timestep):
#     error_pri = [0, 0, 0, 0]
#     tolerance_pri = [0, 0, 0, 0]
#     error_dual = [0, 0, 0, 0]
#     tolerance_dual = [0, 0, 0, 0]

#     # 并行获取所有 error_tolerance 结果
#     tasks = [fetch_error_tolerance(rs, i, timestep, k) for i in range(n_cav)]
#     error_tolerances = await asyncio.gather(*tasks)

#     # 逐个处理第一个条件，必须在所有n_cav中完成
#     for i in range(n_cav):
#         if not await check_criteria(error_tolerances[i], 0, error_pri, tolerance_pri, error_dual, tolerance_dual):
#             return False

#     # 逐个处理第二个条件，仅在n_cav-1中完成
#     for i in range(n_cav-1):
#         if not await check_criteria(error_tolerances[i], 1, error_pri, tolerance_pri, error_dual, tolerance_dual):
#             return False

#     # 逐个处理第三个条件，必须在所有n_cav中完成
#     for i in range(n_cav):
#         if not await check_criteria(error_tolerances[i], 2, error_pri, tolerance_pri, error_dual, tolerance_dual):
#             return False

#     # 逐个处理第四个条件，必须在所有n_cav中完成
#     for i in range(n_cav):
#         if not await check_criteria(error_tolerances[i], 3, error_pri, tolerance_pri, error_dual, tolerance_dual):
#             return False

#     return True


#=================================================================
import asyncio
import pickle



async def fetch_error_tolerance(rs, i, timestep, k):
    while True:
        error_tolerance_bytes = rs.mget(f'error_tolerance_{i}_{timestep}_{k}')[0]
        if error_tolerance_bytes is not None:
            return pickle.loads(error_tolerance_bytes)
        await asyncio.sleep(0)  # 让出执行权，避免阻塞

async def check_criteria(n_cav, error_tolerances, idx):
    error_pri, tolerance_pri, error_dual, tolerance_dual = 0, 0, 0, 0

    if idx == 1:
        range_limit = n_cav - 1
    else:
        range_limit = n_cav

    for i in range(range_limit):
        error_pri += error_tolerances[i][idx][0]
        tolerance_pri += error_tolerances[i][idx][1]
        error_dual += error_tolerances[i][idx][2]
        tolerance_dual += error_tolerances[i][idx][3]

        if error_pri > tolerance_pri or error_dual > tolerance_dual:
            return False
    return True

async def StopCriteria(k, rs, n_cav, timestep):
    tasks = [fetch_error_tolerance(rs, i, timestep, k) for i in range(n_cav)]
    error_tolerances = await asyncio.gather(*tasks)

    criteria_tasks = [
        check_criteria(n_cav, error_tolerances, 0),
        check_criteria(n_cav, error_tolerances, 1),
        check_criteria(n_cav, error_tolerances, 2),
        check_criteria(n_cav, error_tolerances, 3),
    ]

    results = await asyncio.gather(*criteria_tasks)

    return all(results)