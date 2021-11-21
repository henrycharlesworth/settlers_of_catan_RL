import numpy as np
import matplotlib.pyplot as plt

def gaussian_pdf(x, mu, sigma):
    return (1.0 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

# x = np.arange(100)
# p = gaussian_pdf(x, 100.0, 30.0) + 0.005
# # p = p / np.sum(p)
# # plt.plot(x, p)
# # plt.show()

def update_opponent_policies(earlier_policies, rollout_manager, args):
    # base_p = gaussian_pdf(np.arange(args.num_policies_to_store),
    #                       args.num_policies_to_store, (3.0/10) * args.num_policies_to_store) + 0.005
    # num_policies = len(earlier_policies)
    # p = base_p[-num_policies:]
    # p = p / np.sum(p)

    p = get_prob_dist(num_policies=len(earlier_policies))

    for i in range(len(rollout_manager.processes)):
        policy_dicts = np.random.choice(earlier_policies, 3, p=p)

        rollout_manager.update_policy(policy_dicts[0], process_id=i, policy_id=1)
        rollout_manager.update_policy(policy_dicts[1], process_id=i, policy_id=2)
        rollout_manager.update_policy(policy_dicts[2], process_id=i, policy_id=3)

def get_prob_dist(num_policies, linear_num=800, linear_prob=0.5):
    p = ((1-linear_prob) / num_policies) * np.ones((num_policies,))

    num_aux = min(linear_num, num_policies)
    h = (2 * linear_prob) / (num_aux + 1)
    grad = h / num_aux
    aux_p = np.zeros((num_aux,))
    for i in range(num_aux):
        aux_p[i] += i * grad

    p[-num_aux:] += aux_p

    p = p / np.sum(p)

    return p

# num_pols = 1000
# x = np.arange(num_pols)
# p = get_prob_dist(num_pols)
#
# plt.plot(x, p)
# plt.show()
