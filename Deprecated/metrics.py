import os
import json
from utils import Utils

class Metrics:

    @staticmethod
    def get_actions_metric(dict1, dict2, sparse=False):
        actions1 = dict1["actions"]
        actions2 = dict2["actions"]
        assert len(actions1) == len(actions2), "Different number of actions taken in wiki and guess"
        n = len(actions1)
        score = 0
        for a1, a2 in zip(actions1, actions2):
            score+=Metrics.compare_action(a1, a2, sparse)
        return score/n

    @staticmethod
    def get_action_specific_metrics(normal_dict, sim_dict, sparse=False):
        normal_actions = normal_dict["actions"]
        sim_actions = sim_dict["actions"]
        if not len(normal_actions) == len(sim_actions):
            print("")
            # print(f"Different number of actions taken in wiki and guess\nNormal: {len(normal_actions)}\n{normal_actions}\nSim: {len(sim_actions)}\n{sim_actions}\n\n")
        metric_dict = {"general": 0}
        count_dict = {"general": 0}
        flag = True
        for na, sa in zip(normal_actions, sim_actions):
            if flag:
                flag=False
                continue
            metric_dict["general"] += Metrics.compare_action(na, sa, sparse)
            count_dict["general"] += 1

            na_name = Metrics.get_action_name(na)
            sa_name = Metrics.get_action_name(sa)
            if na_name in metric_dict.keys():
                metric_dict[na_name] += Metrics.compare_action(na, sa, sparse)
                count_dict[na_name] += 1
            else:
                metric_dict[na_name] = Metrics.compare_action(na, sa, sparse)
                count_dict[na_name] = 1
        
        for key in metric_dict.keys():
            metric_dict[key]/= count_dict[key]
        return metric_dict
        
        
    
    @staticmethod
    def compare_action(action1, action2, sparse=False):
        action1 = action1.lower()
        action2 = action2.lower()
        if sparse:
            if Metrics.get_action_name(action1) == Metrics.get_action_name(action2):
                return 1
            else:
                return 0
        else:            
            if action1 == action2:
                return 1
            else:
                return 0
    
    @staticmethod
    def get_action_name(full_action):
        full_action = full_action.strip()
        ind = full_action.find('[')
        if ind < 0:
            return None
        else:
            return full_action[:ind]
    
    @staticmethod
    def get_action_specific_avg_metric(dir_path):
        avg_metrics_dict = {"general":0, "Search":0, "Lookup":0, "Finish":0}
        metric_names = ["general", "Search", "Lookup", "Finish"]
        n_metrics = 0
        for direc in os.listdir(dir_path):
            try:
                metrics_dict = Utils.read_json(os.path.join(dir_path, direc, "metrics.json"))
            except FileNotFoundError:
                continue
            for metric in metric_names:
                avg_metrics_dict[metric] += metrics_dict.get(metric, 0)
            n_metrics+=1
        
        for key in avg_metrics_dict.keys():
            avg_metrics_dict[key]/=n_metrics
        return avg_metrics_dict, n_metrics
    
    @staticmethod
    def recalculate_metrics(base_traj_path):
        for folder_name in os.listdir(base_traj_path):
            save_dir = os.path.join(base_traj_path, str(folder_name))
            normal_observations_dict = Utils.read_json(os.path.join(save_dir, "normalobs.json"))
            sim_observations_dict = Utils.read_json(os.path.join(save_dir, "simobs.json"))

            metrics_file_path = os.path.join(save_dir, "metrics.json")
            if os.path.exists(metrics_file_path):
                os.remove(metrics_file_path)
            metric_dict = Metrics.get_action_specific_metrics(normal_observations_dict, sim_observations_dict, sparse=False)
            Utils.save_json(metric_dict, metrics_file_path)