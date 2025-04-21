import rosbag
import pickle

import numpy as np
import pandas as pd

def read_bag(data_path: str, q_idxs=np.s_[4:16], qd_idxs=np.s_[22:34]):
    bag = rosbag.Bag(data_path, 'r')
    topics = ['observation', 'pd_target', 'action']
    ts = []
    obses = []
    qs = []
    qds = []
    q_deses = []
    qd_deses = []
    Kps = []
    Kds = []
    actions = []
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == 'observation':
            obs = np.array(msg.observation)
            obses.append(obs)
            qs.append(obs[q_idxs])
            qds.append(obs[qd_idxs])
            ts.append(t.to_sec())
        elif topic == 'pd_target':
            q_deses.append(msg.q_des[:12])
            qd_deses.append(msg.qd_des[:12])
            Kps.append(msg.Kp[:12])
            Kds.append(msg.Kd[:12])
        elif topic == 'action':
            actions.append(np.array(msg.action))
    bag.close()
    assert len(obses) == len(q_deses) == len(actions)
    ts = [t - ts[0] for t in ts]

    return (ts, obses, qs, qds, q_deses, qd_deses, Kps, Kds, actions)

data = read_bag('bag_files/subrollout_00.bag')

# Создаём словарь для DataFrame
data_dict = {
    "time": data[0],          # ts
    "observation": data[1],   # obses
    "q": data[2],            # qs
    "qd": data[3],           # qds
    "q_des": data[4],        # q_deses
    "qd_des": data[5],       # qd_deses
    "Kp": data[6],           # Kps
    "Kd": data[7],           # Kds
    "action": data[8]        # actions
}

# Создаём DataFrame
df = pd.DataFrame(data_dict)

df['torque'] = df.apply(lambda row: np.array(row['Kp']*(row['q_des']-row['q']) + row['Kd']*row['qd']), axis=1)
# Сохраняем в pickle
with open(f"/app/output/output.pkl", 'wb') as f:
    pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)



