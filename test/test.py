from tensorboardX import SummaryWriter

# 指定事件文件的路径
log_dir = "../results/MyEnv/MyEnv/mappo/check/run2/logs/agent0/value_loss/agent0/value_loss/events.out.tfevents.1697457771.autodl-container-1d9444a656-57a2ed27"


# 创建一个 SummaryWriter 实例
writer = SummaryWriter(log_dir=log_dir)
