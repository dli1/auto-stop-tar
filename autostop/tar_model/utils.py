
def calculate_ap(pid2label, ranked_pids, cutoff=0.5):
    num_rel = 0
    total_precision = 0.0
    for i, pid in enumerate(ranked_pids):
        label = pid2label[pid]
        if label >= cutoff:
            num_rel += 1
            total_precision += num_rel / (i + 1.0)

    return (total_precision / num_rel) if num_rel > 0 else 0.0


def calculate_losser(recall_cost, cost, N, R):
    return (1-recall_cost)**2 + (100/N)**2 * (cost/(R+100))**2