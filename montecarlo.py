import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
number_of_processes = comm.Get_size()

rt_mean = 1.08 ** (1 / 365)
# rt_mean = 1.0
total_years = 40
# kt_mean = 1.02**(1/365)
kt_mean = 1.0
transaction_cost = 10
start_capital = 100000
leverage = 1.25
simulation_runs = 200
rebalance_period = 30

total_days = 365 * total_years

rt = np.random.laplace(rt_mean, 0.007, (simulation_runs, total_days)).astype(dtype=np.double)
lev_depot = np.zeros((simulation_runs, total_days), dtype=np.double)
normal_depot = np.zeros((simulation_runs, total_days), dtype=np.double)
rebalance = np.zeros((simulation_runs, total_days), dtype=np.double)
nonlev_depot = np.zeros((simulation_runs, total_days), dtype=np.double)

for j in range(simulation_runs):

    lev_depot[j, 0] = start_capital * (leverage - 1)
    normal_depot[j, 0] = start_capital - lev_depot[j,0]
    nonlev_depot[j, 0] = start_capital
    for i in range(total_days - 1):
        is_rebalance_day = (i + 1) % rebalance_period == 0
        if is_rebalance_day:
            rebalance[j, i] = lev_depot[j, 0] - lev_depot[j, i] * (2 * rt[j, i] - kt_mean)
            normal_depot[j, i + 1] = normal_depot[j, i] * rt[j, i] - rebalance[j, i] - transaction_cost
            lev_depot[j, i + 1] = lev_depot[j, i] * (2 * rt[j, i] - kt_mean) + rebalance[j, i]
        else:
            rebalance[j, i] = 0
            normal_depot[j, i + 1] = normal_depot[j, i] * rt[j, i]
            lev_depot[j, i + 1] = lev_depot[j, i] * (2 * rt[j, i] - kt_mean)

        nonlev_depot[j, i + 1] = nonlev_depot[j, i] * rt[j, i]
    print('simulation_run: {}, rank: {}'.format(j,rank))
    
all_nonlev_depot = None
all_lev_depot = None
all_normal_depot = None
all_rebalance = None
        
if rank == 0:
    all_nonlev_depot = np.empty(simulation_runs*number_of_processes)
    all_lev_depot = np.empty(simulation_runs*number_of_processes)
    all_normal_depot = np.empty(simulation_runs*number_of_processes)
    all_rebalance = np.empty(simulation_runs*number_of_processes)
    

comm.Gather(np.ascontiguousarray(nonlev_depot[:,-1]), all_nonlev_depot, root=0)
comm.Gather(np.ascontiguousarray(lev_depot[:,-1]), all_lev_depot, root=0)
comm.Gather(np.ascontiguousarray(normal_depot[:,-1]), all_normal_depot, root=0)
comm.Gather(np.ascontiguousarray(rebalance[:,-1]), all_rebalance, root=0)


if rank == 0:
    assets_lev_with_rebalance = all_normal_depot + all_lev_depot
    assets_nonlev = all_nonlev_depot

    print('levWithRebalanceResult')
    print('number > 1: {}, mean: {}, var: {}, median: {}, max: {}'.format(
        np.sum(assets_lev_with_rebalance > start_capital) / assets_lev_with_rebalance.size,
        np.mean(assets_lev_with_rebalance), np.var(assets_lev_with_rebalance, ddof=1),
        np.median(assets_lev_with_rebalance), np.max(assets_lev_with_rebalance)))
    print('nonLevResult')
    print('number > 1: {}, mean: {}, var: {}, median: {}, max: {}'.format(
        np.sum(assets_nonlev > start_capital) / assets_nonlev.size,
        np.mean(assets_nonlev),
        np.var(assets_nonlev, ddof=1),
        np.median(assets_nonlev), np.max(assets_nonlev)))

