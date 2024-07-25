import torch
import random


class Context:
    def __init__(self, objects, attributes, data):
        self.objects = objects
        self.attributes = attributes
        self.data = data

    @classmethod
    def from_matrix(cls, data, objects=None, attributes=None):
        if objects is None:
            objects = torch.arange(0, data.shape[0])

        if attributes is None:
            attributes = torch.arange(0, data.shape[1])

        data = (data > 0).float()
        return cls(objects, attributes, data)

    def reduce(self):
        reduced_columns = self.dis_columns()
        self.data = self.data[:, reduced_columns]
        self.attributes = self.attributes[reduced_columns]

    # reduce C
    def dis_columns(self):
        num_rows, num_cols = self.data.shape
        all_columns = torch.arange(num_cols)
        selected_columns = []
        column_sums = torch.sum(self.data, dim=0)
        constant_columns = torch.where((column_sums == 0) | (column_sums == self.data.size(0)))[0]
        non_constant_columns = torch.tensor(list(set(all_columns.cpu().numpy()) - set(constant_columns.cpu().numpy())))

        for col in range(non_constant_columns.shape[0]):
            random_index = random.choice(non_constant_columns)
            selected_columns.append(int(random_index))
            non_constant_columns = torch.masked_select(non_constant_columns, non_constant_columns != random_index)
            re_matrix = self.data[:, selected_columns]
            if torch.unique(re_matrix, dim=0).shape[0] == num_rows:
                break

        return selected_columns

    def __repr__(self):
        return f"Context({self.objects}, {self.attributes}, {self.data})"
