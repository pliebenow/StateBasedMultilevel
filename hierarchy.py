
import torch.optim as optim
from resNet152 import ResNet152


class multilevel_hierarchy():

    def __init__(self, level_list: list):
        num_level = len(level_list)
        model_hierarchy = []
        optim_hierarchy = []

        for n in range(num_level):
            model_hierarchy.append(ResNet152(level_list[n][0], level_list[n][1], level_list[n][2]))
            optim_hierarchy.append(optim.SGD(model_hierarchy[n].parameters(), lr=0.0001))


multilevel_hierarchy([[3, 8, 36, 3], [2, 4, 18, 2], [1, 4, 9, 1]])
