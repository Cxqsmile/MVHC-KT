import torch
from HG_dataset import KTDataset
from HG_dataset import collate_fn_4sq
from torch.utils.data import DataLoader

from model_cl import HGC_lstm

import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

save_dir = 'logs'
dataset = 'algebra2005'

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
HGcq_filepath = '.\\data\\'+dataset+'\\HG1\\HG_cq.pt'
HGqc_filepath = '.\\data\\'+dataset+'\\HG1\\HG_qc.pt'
GCN_filepath = '.\\data\\'+dataset+'\\concept_cooccurrence\\edge_index.pt'
similar_filepath = '.\\data\\'+dataset+'\\similar\\HG_question_similarity_graph.pt'
test_filepath = '.\\data\\'+dataset+'\\lstm_data\\test_set.npy'

# algebra2005
num_question = 1084
num_skill = 342
num_student = 3340

emb_dim = 128 # Dimensions of problem and conceptual representation
correct_emb_dim = 16  # The representation dimension of the answer
ht_dim = 32  # The dimension of the out output by LSTM, which is the student feature representation vector
num_mv_layers = 1  # Number of layers in the first channel
num_gcn_layers = 3  # Number of layers in the second channel
num_question_layers = 6  #Number of layers in the third channel
num_lstm_layers = 3  # lstm layer
temperature = 0.5
dropout = 0.1
lambda_cl = 0.1 #Proportion and weight of loss in contrastive learning

batch_size = 256
PADDING_IDX = 18000 ##This value is not being used anymore


test_dataset = KTDataset(HGcq_filepath=HGcq_filepath,
                               HGqc_filepath=HGqc_filepath,
                               GCN_filepath=GCN_filepath,
                               similar_filepath=similar_filepath,
                               lstm_filepath=test_filepath,
                               device=device)

test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX))

# Load Model
model_filename = 'trained_model_algebra2005.pt'
current_save_dir = os.path.join(save_dir, model_filename)

model = HGC_lstm(num_question, num_skill, emb_dim, correct_emb_dim, ht_dim, num_mv_layers, num_gcn_layers, num_question_layers, num_lstm_layers, temperature, dropout, device)
model.load_state_dict(torch.load(current_save_dir))
model = model.to(device)

criterion = nn.BCELoss(reduction='none').to(device)

test_loss_list = []
test_acc_list = []
test_auc_list = []
#test
model.eval()
with torch.no_grad():
    for idx, batch in enumerate(test_dataloader):

        print("Test. Batch {}/{}".format(idx, len(test_dataloader)))
        predictions, loss_cl_skill, loss_cl_question = model(test_dataset, batch)

        # calculate loss
        true_label = torch.unsqueeze(batch["student_correct_seq"], 2)[:, 1:, :].float().to(device)
        loss_pre = criterion(predictions, true_label)
        loss_pre = torch.squeeze(loss_pre, 2).to(device)
        mask = batch["student_seq_mask"][:, 1:].to(device)
        masked_loss = loss_pre * mask

        masked_loss = masked_loss.sum(dim=1) / mask.sum(dim=1)

        loss_pre = masked_loss.mean()
        loss = loss_pre + lambda_cl * (loss_cl_skill + loss_cl_question)
        print("Test. loss_pre: {:.4f}; loss_cl_skill: {:.4f}; loss_cl_question: {:.4f}; "
                     "loss: {:.4f}".format(loss_pre.item(), loss_cl_skill, loss_cl_question, loss))

        test_loss_list.append(loss.item())


        mask_unsque = torch.unsqueeze(mask, 2).to(device)  # [batch, seqlen, 1]
        predictions_binary = (predictions >= 0.5).float()
        prediction_label_mask = (predictions_binary == true_label) * mask_unsque
        true_label_num = batch["student_seq_len"].sum() - batch_size
        test_acc_temp = (prediction_label_mask.sum().item()) / (true_label_num.item())
        test_acc_list.append(test_acc_temp)

        # valid auc
        y_pred_list = []
        y_true_list = []
        batch_seqlen = batch["student_seq_len"]
        for batch_idx in range(len(batch_seqlen)):
            for seqlen_stu in range(batch_seqlen[batch_idx] - 1):
                y_pred_list.append(predictions[batch_idx, seqlen_stu, 0].item())
                y_true_list.append(true_label[batch_idx, seqlen_stu, 0].item())

        auc_temp = roc_auc_score(y_true_list, y_pred_list)
        test_auc_list.append(auc_temp)

    test_loss = np.mean(test_loss_list)
    test_acc = np.mean(test_acc_list)
    test_auc = np.mean(test_auc_list)

print("Test finishes")
print("Test loss: {}".format(test_loss))
print("Test acc: {:.4f}".format(test_acc))
print("Test auc: {:.4f}".format(test_auc))





