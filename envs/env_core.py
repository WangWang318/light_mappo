import numpy as np


class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self, agent_num=6, enemy_num=1, reward=1, penalty=-0.5):
        self.agent_num = agent_num  # 设置智能体(小飞机)的个数，这里设置为两个 # set the number of agents(aircrafts), here set to two
        self.teammate_num = agent_num - enemy_num
        self.obs_dim = 2  # 设置智能体的观测维度 # set the observation dimension of agents
        self.action_dim = 2  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional
        self.reward = reward
        self.enemy_num = enemy_num    # 敌机的数量
        self.penalty = penalty

        # 初始化飞机的位置
        self.location = []
        for i in range(self.teammate_num):
            sub_obs = np.random.uniform(0,2,size=(self.obs_dim,))
            self.location.append(sub_obs)
        for i in range(self.enemy_num):
            sub_obs = np.random.uniform(-2,2, size=(self.obs_dim, )) + 6   # 敌机的初始位置
            self.location.append(sub_obs)

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        # 敌机的位置
        """

        sub_agent_obs = []
        for i in range(self.teammate_num):
            sub_obs = np.random.uniform(0,2, size=(self.obs_dim,))
            sub_agent_obs.append(sub_obs)
        for i in range(self.enemy_num):
            sub_obs = np.random.uniform(-2,2, size=(self.obs_dim, )) + 6   # 敌机的初始位置
            sub_agent_obs.append(sub_obs)

        self.location = sub_agent_obs

        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        sub_agent_reward = np.zeros((self.agent_num, 1))
        sub_agent_done = [False] * self.agent_num
        sub_agent_info = [{}] * self.agent_num

        print("location")
        print(self.location)
        # print("actions")
        # print(actions)
        actions = np.clip(actions, -1, 1)
        print("actions_clip")
        print(actions)

        # 更新位置
        self.location = np.clip(np.add(self.location, actions),0, 10)


        sub_agent_obs = self.location
        # 攻打敌机
        for i in range(self.teammate_num, self.agent_num):
            sum = 0
            tmp_buffer = []
            # 同伴攻击敌机
            for j in range(self.teammate_num):
                # 计算欧氏距离
                if np.linalg.norm(np.array(self.location[i]) - np.array(self.location[j])) < 2:
                    sum += 1
                    tmp_buffer.append(j)    # 记录谁击落了敌机
                    # print("{} 击打了敌机".format(j))
                    sub_agent_reward[j] += self.reward / 20
            if sum >= 2:    # 击落敌机
                print("击落了飞机")
                # sub_agent_done[i] = True # 游戏结束
                for k in range(len(tmp_buffer)):
                    sub_agent_reward[tmp_buffer[k]] += self.reward
                sub_agent_reward[i] += self.penalty / 10

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]


