import ray
import numpy as np
import matplotlib.pyplot as plt

ray.init(object_store_memory=32e09)

rt_mean = 1.08 ** (1 / 365)
# rt_mean = 1.0
total_years = 40
# kt_mean = 1.02**(1/365)
kt_mean = 1.0
transaction_cost = 10
start_capital = 100000
leverage = 1.25
simulation_runs = 32000
rebalance_period = 30

total_days = 365 * total_years

@ray.remote
def run_simulation(j):
    print('running simulation ' + str(j))
    rt = np.random.laplace(rt_mean, 0.007, total_days).astype(dtype=np.double)
    lev_depot = np.zeros(total_days)
    normal_depot = np.zeros(total_days)
    rebalance = np.zeros(total_days)
    nonlev_depot = np.zeros(total_days)

    lev_depot[0] = start_capital * (leverage - 1)
    normal_depot[0] = start_capital - lev_depot[0]
    nonlev_depot[0] = start_capital
    for i in range(total_days - 1):
        is_rebalance_day = (i + 1) % rebalance_period == 0
        if is_rebalance_day:
            rebalance[i] = lev_depot[0] - lev_depot[i] * (2 * rt[i] - kt_mean)
            normal_depot[i + 1] = normal_depot[i] * rt[i] - rebalance[i] - transaction_cost
            lev_depot[i + 1] = lev_depot[i] * (2 * rt[i] - kt_mean) + rebalance[i]
        else:
            rebalance[i] = 0
            normal_depot[i + 1] = normal_depot[i] * rt[i]
            lev_depot[i + 1] = lev_depot[i] * (2 * rt[i] - kt_mean)

        nonlev_depot[i + 1] = nonlev_depot[i] * rt[i]
    return lev_depot, normal_depot, rebalance, nonlev_depot

futures = [run_simulation.remote(j) for j in range(simulation_runs)]
ray_result = ray.get(futures)

lev_depot = np.vstack(tuple((result[0] for result in ray_result)))
normal_depot = np.vstack(tuple((result[1] for result in ray_result)))
leverage = np.vstack(tuple((result[2] for result in ray_result)))
nonlev_depot = np.vstack(tuple((result[3] for result in ray_result)))

# plt.plot(np.arange(total_days), lev_depot[0, :], label='lev_depot')
# plt.plot(np.arange(total_days), normal_depot[0, :], label='normal_depot')
# plt.plot(np.arange(total_days), normal_depot[0, :] + lev_depot[0, :], label='normal_depot + lev_depot')
# plt.plot(np.arange(total_days), nonlev_depot[0, :], label='nonlev_depot')

assets_lev_with_rebalance = normal_depot[:, -1] + lev_depot[:, -1]
assets_nonlev = nonlev_depot[:, -1]

plt.hist(assets_lev_with_rebalance, 1000)
plt.title('assets lev with rebalance')
plt.figure()
plt.hist(assets_nonlev, 1000)
plt.title('assets nonlev')
# print(normal_depot[0, -1] + lev_depot[0, -1] - nonlev_depot[0, -1])

plt.show()

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

#
# # levResult = np.prod(2 * rt - kt_mean, axis=1)
#
# print('levWithRebalanceResult')
# print('number > 1: {}, mean: {}, var: {}, median: {}, max: {}'.format(np.sum(levResult > 1) / levResult.size,
#                                                                       np.mean(levResult), np.var(levResult, ddof=1),
#                                                                       np.median(levResult), np.max(levResult)))
# # print('levResult')
# # print('number > 1: {}, mean: {}, var: {}, median: {}, max: {}'.format(np.sum(levResult > 1) / levResult.size,
# #                                                                       np.mean(levResult), np.var(levResult, ddof=1),
# #                                                                       np.median(levResult), np.max(levResult)))
# print('nonLevResult')
# print('number > 1: {}, mean: {}, var: {}, median: {}, max: {}'.format(np.sum(nonLevResult > 1) / nonLevResult.size,
#                                                                       np.mean(nonLevResult),
#                                                                       np.var(nonLevResult, ddof=1),
#                                                                       np.median(nonLevResult), np.max(nonLevResult)))
#
# plt.figure()
# plt.title('levResult')
# y, x, _ = plt.hist(levResult, 1000, range=(0.0, 400))
# print('levResult argmax: {}'.format(x[np.argmax(y)]))
#
# plt.figure()
# plt.title('nonLevResult')
# y, x, _ = plt.hist(nonLevResult, 1000, range=(0.0, 400))
# print('nonLevResult argmax: {}'.format(x[np.argmax(y)]))
# plt.show(block=True)
