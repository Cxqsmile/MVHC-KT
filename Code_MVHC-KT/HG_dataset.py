# coding=utf-8
import torch

from utils import *
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class KTDataset(Dataset):
    def __init__(self, HGcq_filepath, HGqc_filepath, GCN_filepath, similar_filepath, lstm_filepath, device):
        super(KTDataset, self).__init__()

        self.device = device

        self.lstm_data = np.load(lstm_filepath,allow_pickle=True)  # (num_student,3)
        # print(self.lstm_data.shape)
        self.num_student = self.lstm_data.shape[0]

        ### The first channel data  tensor
        self.HG_cq = torch.load(HGcq_filepath).to(device)
        self.HG_qc = torch.load(HGqc_filepath).to(device)

        ### The second channel data tensor
        self.GCN_adj = torch.load(GCN_filepath).to(device)

        ### The third channel data tensor
        self.HG_similar_graph = torch.load(similar_filepath).to(device)

    def __len__(self):
        return self.num_student

    def __getitem__(self, student_idx):
        student_question_seq = self.lstm_data[student_idx][0]
        student_skill_seq = self.lstm_data[student_idx][1]
        student_correct_seq = self.lstm_data[student_idx][2]
        student_seq_len = len(student_question_seq)
        student_seq_mask = [1] * student_seq_len
        # print(student_seq_len)

        sample = {
            "student_idx": torch.tensor(student_idx).to(self.device),
            "student_question_seq": torch.tensor(student_question_seq).to(self.device),
            "student_skill_seq": torch.tensor(student_skill_seq).to(self.device),
            "student_seq_len": torch.tensor(student_seq_len).to(self.device),
            "student_seq_mask": torch.tensor(student_seq_mask).to(self.device),
            "student_correct_seq": torch.tensor(student_correct_seq).to(self.device),
        }

        return sample


def collate_fn_4sq(batch, padding_value):
    """
    Pad sequence in the batch into a fixed length
    batch: list obj
    """
    # get each item in the batch
    batch_student_idx = []
    batch_student_question_seq = []
    batch_student_skill_seq = []
    batch_student_seq_len = []
    batch_student_seq_mask = []
    batch_student_correct_seq = []
    for item in batch:
        batch_student_idx.append(item["student_idx"])
        batch_student_question_seq.append(item["student_question_seq"])
        batch_student_skill_seq.append(item["student_skill_seq"])
        batch_student_seq_len.append(item["student_seq_len"])
        batch_student_seq_mask.append(item["student_seq_mask"])
        batch_student_correct_seq.append(item["student_correct_seq"])




    # pad seq
    pad_student_question_seq = pad_sequence(batch_student_question_seq, batch_first=True, padding_value=1084)
    pad_student_skill_seq = pad_sequence(batch_student_skill_seq, batch_first=True, padding_value=342)
    # pad_student_correct_seq = pad_sequence(batch_student_correct_seq, batch_first=True, padding_value=2)
    pad_student_correct_seq = pad_sequence(batch_student_correct_seq, batch_first=True, padding_value=0)
    pad_student_seq_mask = pad_sequence(batch_student_seq_mask, batch_first=True, padding_value=0)



    # print(torch.isnan(pad_student_question_seq).any())
    # print(torch.isnan(pad_student_skill_seq).any())
    # print(torch.isnan(pad_student_correct_seq).any())
    # print(torch.isnan(pad_student_seq_mask).any())


    # stack list obj to a torch.tensor
    batch_student_idx = torch.stack(batch_student_idx)
    batch_student_seq_len = torch.stack(batch_student_seq_len)

    collate_sample = {
        "student_idx": batch_student_idx,
        "student_question_seq": pad_student_question_seq,
        "student_skill_seq": pad_student_skill_seq,
        "student_seq_len": batch_student_seq_len,
        "student_seq_mask": pad_student_seq_mask,
        "student_correct_seq": pad_student_correct_seq,
    }

    return collate_sample

