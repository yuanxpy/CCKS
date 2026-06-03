import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

import torch
import os
import torch.nn.functional as F
import math
import random
import numpy as np
import torch.nn as nn
REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env,t_ep,test_mode=False,ask_budget=None,batch=None,hidden_states=None,agent_obs=None,epi_obs=None,episode_counts=None,ohter_outputs=None,n_agents=None,n_actions=None,ohter_colas=None,obss=None):
        agent_inputs = th.nn.functional.softmax(agent_inputs, dim=-1)
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

#---------------------------------cons----------------------------------
        # avail_actions_ind = np.nonzero(avail_actions)[0]

        # if np.random.uniform() >= self.epsilon:
        #         action = torch.argmax(q_value)
        # else:
        #     if avail_actions_ind.shape[0] <= 1:
        #         action = avail_actions_ind[0]
        #     else:
        #         if episode_counts > self.args.start_advice and ask_budget[agent_num] > 0:  # able to send a request
        #             action, ask_budget = self.ask_advice(q_value, hidden_state,
        #                                                                         obs, last_action, agent_num,
        #                                                                         agent_obs, epi_obs, ask_budget,
        #                                                                         episode_counts)
        #             if (action is None) or (action not in avail_actions_ind): # knowledge is not available
        #                 # perform epsilon-greedy
        #                 if np.random.uniform() < epsilon:
        #                     action = np.random.choice(avail_actions_ind)
        #                 else:
        #                     action = torch.argmax(q_value)
        #         else:  # unable to send a request, perform epsilon-greedy
        #             if np.random.uniform() < epsilon:
        #                 action = np.random.choice(avail_actions_ind)
        #             else:
        #                 action = torch.argmax(q_value)
#----------------------------------cons---------------------------------------
        # if 0:
        if np.random.uniform() >= self.epsilon:
            picked_actions = masked_q_values.max(dim=2)[1]
            teacher_scores = [0.0] * n_agents
        else:
            picked_actions = []      
            teacher_scores = []        
            dim0, dim1, dim2 = masked_q_values.shape
            # for o in range(dim0):
            for i in range(dim1):
                q_values_i = masked_q_values[0][i]
                avail_actions_ind = np.nonzero(avail_actions[0][i])
                avail_actions_ind = torch.squeeze(avail_actions_ind,1)
                teacher_score_i = 0
                if avail_actions_ind.shape[0] <= 1:
                    action_i = avail_actions_ind[0]
                else:
                    if t_env > self.args.start_advice and ask_budget[i] > 0:  # able to send a request
                    # if 1:
                        other_avail_actions = avail_actions[0][i]
                        for count_avail in range(dim1-1):
                                ohter_outputs[i][count_avail][other_avail_actions == 0.0] = -float("inf")
                        action_i, ask_budget, teacher_score_i = self.ask_advice2(q_values_i, hidden_states,
                                                                            obss[i], i,
                                                                            agent_obs, epi_obs, ask_budget,
                                                                            episode_counts,ohter_outputs,n_agents,n_actions,ohter_colas)
                        if (action_i is None) or (action_i not in avail_actions_ind): #  knowledge is not available
                            if np.random.uniform() < self.epsilon:
                                action_i = np.random.choice(avail_actions_ind.cpu())
                            else:
                                action_i = torch.argmax(q_values_i)
                    else:  # unable to send a request, perform epsilon-greedy
                        if np.random.uniform() < self.epsilon:
                            action_i = np.random.choice(avail_actions_ind.cpu())
                        else:
                            action_i = torch.argmax(q_values_i)
                teacher_scores.append(teacher_score_i)    
                picked_actions.append(action_i)
            picked_actions = torch.tensor([picked_actions], dtype=torch.long).cuda()
                    
        return picked_actions, ask_budget, teacher_scores
                    
            # for i in dim0:
            #     for j in dim1: 
            # if avail_actions_ind.shape[0] <= 1:
            #     action = avail_actions_ind[0]
#----------------------------------cons---------------------------------------




        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions,ask_budget



    def ask_advice2(self, q_value, hidden_state, obs, i, agent_obs, epi_obs, ask_budget, episode_counts, ohter_outputs,
                n_agents, n_actions, ohter_colas):
        out = F.softmax(q_value, dim=-1).unsqueeze(dim=0)
        count_avail = torch.count_nonzero(out[0]).item() #有效动作的数量
        agent_id = np.zeros(n_agents)
        agent_id[i] = 1.
        # obs = obs.tolist()
        teacher_score = None
        # if random.random() < (math.pow((1 + self.args.variable_a), -math.sqrt(agent_obs[i][tuple(obs)]))) and obs not in np.array(epi_obs[i]):
        if 1:
            s, s_key = [], []
            for j in range(n_agents):
                s.append(obs)
                s_key.append(obs)
                # if self.args.reuse_network:
                #     s[j] = np.hstack((s[j], np.eye(n_agents)[j]))
            obs_adv = torch.Tensor(np.array(s))
            if self.args.device == 'cuda':
                obs_adv = obs_adv.cuda()
                out_adv = torch.zeros([n_agents, n_actions]).cuda()
                q_values = torch.zeros([n_agents, n_actions]).cuda()
            else:
                out_adv = torch.zeros([n_agents, n_actions])
                q_values = torch.zeros([n_agents, n_actions])

            # if self.args.reuse_network:
            #     q_values, _ = self.eval_rnn(obs_adv, hidden_state.repeat(n_agents, 1))
            # else:
            if 1:
                for advisor_i in range(n_agents):
                    q_values[advisor_i] = ohter_outputs[i][advisor_i].unsqueeze(dim=0)

            out_adv = F.softmax(q_values, dim=-1).detach()
            # std = torch.std(out_adv, dim=-1, unbiased=False)
            # # Normalize std using Min-Max normalization to obtain the policy confidence
            # trust = std * (n_actions / math.sqrt(n_actions - 1))

            dist = Categorical(probs=out_adv)
            entropy = dist.entropy()
            # temp_entropy = entropy.clone()
            # temp_entropy[i] = -float('inf')
            # trust_weight = F.softmax(temp_entropy, dim=-1).detach()
            pdist = nn.PairwiseDistance(p=2)
            cola_cos = []
            for j in range(n_agents):
                cola_cos.append(-pdist(ohter_colas[i],ohter_colas[j]))
            cola_weight = torch.stack(cola_cos,dim=0)
            # cola_weight[i] = -float('inf')
            # cola_weight = F.softmax(cola_weight, dim=-1).detach()
            final_value = []
            for j in range(n_agents):
                final_value.append(cola_weight[j] * entropy[j])
            final_weight = torch.stack(final_value,dim=0)
            # final_weight = -final_weight
            final_weight[i] = -float('inf')
            final_weight = F.softmax(final_weight, dim=-1).detach()


            # K = self.args.K
            K = int(count_avail/3)+1
            # a_best_worst = []  # Store the best & worst actions
            # for j in range(n_agents):
            #     a_best_worst.append([-1] * K)
            #     # a_best_worst.append([-1, -1])
            #     if i == j:
            #         continue
            #     else:
            #         for num1 in range(K):
            #             a_best_worst[j][num1] = out_adv[j].argsort()[num1].item()
            #         # a_best_worst[j][1] = out_adv[j].argmin().item()

            # Identify which agents are eligible for knowledge sharing
            give_adv_list = [_ for _ in range(n_agents)]
            for z in range(n_agents):
                # 没见过obs或z就是学生智能体的编号
                # if (z == i) or (tuple(s_key[z]) not in agent_obs[z]):
                if z == i:
                    give_adv_list.remove(z)
                    continue
                # if (agent_obs[z][tuple(s_key[z])] <= agent_obs[i][tuple(s_key[i])]) and (q_values[z].max() <= q_value.max()):
                #     give_adv_list.remove(z)
                #     continue
                # 当老师智能体q函数 最大值不比学生的大，最小值不比学生的小则没有学习的价值
                if (q_values[z].min() >= q_value.min()) and (q_values[z].max() <= q_value.max()):
                    give_adv_list.remove(z)
                    continue
                if entropy[i] < entropy[z]:
                    give_adv_list.remove(z)
                    continue


            if len(give_adv_list) > 0:  # There are no agents eligible for knowledge sharing
                get_advice = False
                k_values = []
                for k in range(n_actions):  # for each action
                    record = []
                    for l in give_adv_list:
                        record.append((out_adv[l, k].item())*final_weight[l].item())
                    #动作k的得分为老师们的指导的均值乘上标准差,意义为当均值越大时得分越高，标准差越大时证明老师们意见不统一则降低得分
                    #这里发现标准差太大了除上去会导致分数太大
                    # k_value = np.sum(record) / np.std(record)
                    k_value = np.sum(record)
                    k_values.append(k_value)
                k_mean = (sum(k_values)/count_avail)
                for k in range(n_actions):  # for each action
                    k_values[k] = k_values[k] - k_mean
                k_values = np.array(k_values)
                k_indexs = k_values.argsort()
                if len(k_values) != 0:
                    get_advice = True
                if get_advice:  # Reference others' knowledge for decision-making
                    out = F.softmax(out / 0.3, dim=-1)  # obtains the new policy by performing softmax normalization
                    # cautiously absorbing the knowledge and sample an action
                    # sort = torch.sort(out)
                    # std = torch.std(out, unbiased=False)
                    # adv_trust = std * (n_actions / math.sqrt(n_actions - 1))
                    # if self.args.name.find('rdm') > -1:  # sample randomly
                    if 1:
                        action = random.choices([_ for _ in range(n_actions)], weights=out[0], k=1)[0]
                    # else:  # sample by targeted exploration
                    #     action = out.argmax() if random.random() < adv_trust else self._targeted_exploration(adv_trust,
                    #                                                                                         sort)
                        if action in k_indexs[:K].tolist(): #这里或改成以1/k_values的概率重新采样一次
                            action = random.choices([_ for _ in range(n_actions)], weights=out[0], k=1)[0]
                    ask_budget[i] = ask_budget[i] - 1
                    teacher_score = k_values[action]
                else:  # There is no need to reference others' knowledge for decision-making.
                    action = None
                    teacher_score = 0
            else:  # There are no agents eligible for knowledge sharing
                action = None
                teacher_score = 0
            return action, ask_budget, teacher_score

        action = None
        teacher_score = 0
        return action, ask_budget, teacher_score
REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
