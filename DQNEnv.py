import numpy as np
import random

def eval_performance(tp_n, fp_n, fn_n):
    if tp_n == fp_n == fn_n == 0:
        precision = recall = f1 = 1
    elif tp_n == 0 and np.sum(fp_n + fn_n) > 0:
        precision = recall = f1 = 0
    else:
        precision = tp_n / (tp_n + fp_n)
        recall = tp_n / (tp_n + fn_n)
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


# define an environment class
class Env(object):
    def __init__(self, action_set, y_window_labels, score_list):
        # initialization
        self.actions = action_set
        self.y = y_window_labels
        self.score_list = score_list
        self.t_env = 0
        self.state = [0,0, 0, 0, 0, 0]  # avr_score, var_score, TP_rate (per window), FP_rate, TN_rate, FN_rate
        self.done = False
        self.tp_n = 0
        self.tn_n = 0
        self.fp_n = 0
        self.fn_n = 0
        self.total_reward = 0
        self.result_hist = []

    # reset the environment
    def reset(self):
        self.t_env = 0
        self.total_reward = 0
        self.done = False
        self.state = [0,0,0, 0, 0, 0]
        self.tp_n = 0
        self.tn_n = 0
        self.fp_n = 0
        self.fn_n = 0
        self.result_hist = []
        return self.state

    # action-state function
    def do_step(self, action, t_env, k_state, e, Episodes):
        self.t_env = t_env
        # generate reward
        if self.score_list[self.t_env] >= action:
            pred_i = 1
        else:
            pred_i = 0
        if self.y[self.t_env] == 0 and pred_i == 0:  # true negative
            self.tn_n += 1
            self.result_hist.append("TN")
            reward = 1
        elif self.y[self.t_env] == 1 and pred_i == 1:  # true positive
            self.tp_n += 1
            self.result_hist.append("TP")
            reward = 1
        elif self.y[self.t_env] == 0 and pred_i == 1:  # false positive
            self.fp_n += 1
            self.result_hist.append("FP")
            reward = -1
        elif self.y[self.t_env] == 1 and pred_i == 0:  # false negative
            self.fn_n += 1
            self.result_hist.append("FN")
            reward = -1
        self.total_reward += reward

        # generate new state, which is based on the observations of previous k windows
        if self.t_env < 1:
            self.state = [0,0,0, 0, 0, 0]
        elif 1 <= self.t_env < len(self.y) - 1:
            # 1.feedback - no delay
            if self.t_env < k_state - 1:
                score_windows = self.score_list[:self.t_env + 1]
                result_windows = self.result_hist[:self.t_env + 1]
            else:
                score_windows = self.score_list[self.t_env - k_state + 1:self.t_env + 1]
                result_windows = self.result_hist[self.t_env - k_state + 1:self.t_env + 1]

            # # 2.feedback - no delay, partial
            # partial_par=3 # the feedback of last 3 timesetps invisible
            # if self.t_env < k_state - 1:
            #     score_windows = self.score_list[:self.t_env + 1]
            #     result_windows = self.result_hist[:self.t_env + 1-partial_par]
            # else:
            #     score_windows = self.score_list[self.t_env - k_state + 1:self.t_env + 1]
            #     result_windows = self.result_hist[self.t_env - k_state + 1:self.t_env + 1-partial_par]

            # 3.feedback - delay
            # delay_t = 100
            # if self.t_env < k_state - 1:
            #     score_windows = self.score_list[:self.t_env + 1]
            #     result_windows = self.result_hist[:self.t_env + 1 - delay_t]
            # else:
            #     score_windows = self.score_list[self.t_env - k_state + 1:self.t_env + 1]
            #     result_windows = self.result_hist[self.t_env - k_state + 1 - delay_t:self.t_env + 1 - delay_t]

            # #4.noisy feedback
            # if self.t_env < k_state - 1:
            #     score_windows = self.score_list[:self.t_env + 1]
            #     result_windows = self.result_hist[:self.t_env + 1]
            # else:
            #     score_windows = self.score_list[self.t_env - k_state + 1:self.t_env + 1]
            #     result_windows = self.result_hist[self.t_env - k_state + 1:self.t_env + 1]
                # pos = random.sample(range(0, 10), 1)
                # for i in pos:
                #     if result_windows[i] == "TP":
                #         result_windows[i] = "FP"
                #     elif result_windows[i] == "TN":
                #         result_windows[i] = "FN"
                #     elif result_windows[i] == "FN":
                #         result_windows[i] = "TN"
                #     elif result_windows[i] == "FP":
                #         result_windows[i] = "TP"

            mean_score = np.mean(score_windows)
            var_score = np.var(score_windows)
            TP_rate = result_windows.count("TP") / len(score_windows)
            FP_rate = result_windows.count("FP") / len(score_windows)
            TN_rate = result_windows.count("TN") / len(score_windows)
            FN_rate = result_windows.count("FN") / len(score_windows)
            self.state = [mean_score, var_score, TP_rate, FP_rate, TN_rate, FN_rate]
        else:
            self.done = True

        # get performance
        if self.t_env == len(self.y) - 1:
            precision, recall, f1 = eval_performance(self.tp_n, self.fp_n, self.fn_n)
            if e % 200 == 0 or e == Episodes - 1:
                print("episode:{}/{} precision:{} recall:{} F1: {}".format(e + 1, Episodes, precision, recall, f1))

        return reward, self.state, self.done
