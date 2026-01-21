import torch
from HG_dataset import KTDataset
from HG_dataset import collate_fn_4sq
from torch.utils.data import DataLoader
from early_stopping import EarlyStopping

from model_cl import HGC_lstm

import torch.optim as optim
import torch.nn as nn
import logging
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import datetime
import matplotlib.pyplot as plt

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

save_dir = 'logs'
dataset = 'algebra2005'
dataset_new = 'algebra2005_new'

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
HGcq_filepath = '.\\data\\'+dataset+'\\HG1\\HG_cq.pt'
HGqc_filepath = '.\\data\\'+dataset+'\\HG1\\HG_qc.pt'
GCN_filepath = '.\\data\\'+dataset+'\\concept_cooccurrence\\edge_index.pt'
similar_filepath = '.\\data\\'+dataset+'\\similar\\HG_question_similarity_graph.pt'
train_filepath = '.\\data\\'+dataset+'\\lstm_data\\train_set.npy'
valid_filepath = '.\\data\\'+dataset+'\\lstm_data\\valid_set.npy'



current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
current_save_dir = os.path.join(save_dir, current_time)
# create current save_dir
os.mkdir(current_save_dir)

# Setup logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=os.path.join(current_save_dir, f"log_training.txt"),
                    filemode='w+')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logging.getLogger('matplotlib.font_manager').disabled = True

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

num_epochs = 200
batch_size = 256
PADDING_IDX = 18000 ##This value is not being used anymore


train_dataset = KTDataset(HGcq_filepath=HGcq_filepath,
                               HGqc_filepath=HGqc_filepath,
                               GCN_filepath=GCN_filepath,
                               similar_filepath=similar_filepath,
                               lstm_filepath=train_filepath,
                               device=device)


valid_dataset = KTDataset(HGcq_filepath=HGcq_filepath,
                               HGqc_filepath=HGqc_filepath,
                               GCN_filepath=GCN_filepath,
                               similar_filepath=similar_filepath,
                               lstm_filepath=valid_filepath,
                               device=device)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX))

valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX))


# Load Model
model = HGC_lstm(num_question, num_skill, emb_dim, correct_emb_dim, ht_dim, num_mv_layers, num_gcn_layers, num_question_layers, num_lstm_layers, temperature, dropout, device)
model = model.to(device)

lr = 1e-3
# decay = 5e-4
decay = 5e-5
lr_scheduler_factor = 0.1
patience = 10
early_stopping = EarlyStopping(patience=patience)


optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
# optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='none').to(device)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', verbose=True, factor=lr_scheduler_factor)

# Train
monitor_loss = float('inf')
best_val_auc = 0.0
best_epoch = 0

# to save train_loss and val_loss train_acc train_auc val_acc val_auc
train_loss_all = []
train_acc_all = []
train_auc_all = []
val_loss_all = []
val_acc_all = []
val_auc_all = []

logging.info("================= emb_dim:{} correct_emb_dim:{} ht_dim:{} num_mv_layers:{} num_gcn_layers:{} num_question_layers:{} num_lstm_layers:{} temperature:{} dropout:{} lambda_cl:{} =================".format(emb_dim, correct_emb_dim,ht_dim,num_mv_layers,num_gcn_layers,num_question_layers,num_lstm_layers,temperature,dropout,lambda_cl))


train_start_time = time.time()
for epoch in range(num_epochs):
    logging.info("================= Epoch {}/{} =================".format(epoch, num_epochs))
    start_time = time.time()

    model.train()

    train_loss_list = []
    train_acc_list = []
    train_auc_list = []
    for idx, batch in enumerate(train_dataloader):
        logging.info("Train. Batch {}/{}".format(idx, len(train_dataloader)))

        batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}

        optimizer.zero_grad()

        # forward_start = time.time()
        predictions, loss_cl_skill, loss_cl_question = model(train_dataset, batch) # [batch, seqlen, 1]
        # forward_end = time.time()
        # print(forward_end - forward_start)

        # true_label = torch.unsqueeze(batch["student_correct_seq"], 2)[:,1:,:].float().to(device) # [batch, seqlen, 1]
        true_label = torch.unsqueeze(batch["student_correct_seq"], 2)[:, 1:, :].float()  # [batch, seqlen, 1]

        # Calculate loss. If there is a memory overflow, consider setting all the values of loss to 'item' before proceeding with the operation **************************************


        loss_pre = criterion(predictions, true_label)
        # print(torch.isnan(loss_pre).any().item())
        # if(torch.isnan(loss_pre).any().item()):
        #     print(predictions)
        #     print(true_label)


        # loss_pre = torch.squeeze(loss_pre, 2).to(device)
        loss_pre = torch.squeeze(loss_pre, 2)
        # print(torch.isnan(loss_pre).any().item())
        # mask = batch["student_seq_mask"][:,1:].to(device)

        mask = batch["student_seq_mask"][:, 1:]
        masked_loss = loss_pre * mask

        masked_loss = masked_loss.sum(dim=1) / mask.sum(dim=1)

        loss_pre = masked_loss.mean()
        # print(torch.isnan(loss_pre).any().item())
        loss = loss_pre + lambda_cl * (loss_cl_skill + loss_cl_question)

        # backward_start = time.time()
        loss.backward()
        optimizer.step()
        # backward_end = time.time()
        # print(backward_end - backward_start)

        train_loss_list.append(loss.item())

        logging.info("Train. loss_pre: {:.4f}; loss_cl_skill: {:.4f}; loss_cl_question: {:.4f}; "
                     "loss: {:.4f}".format(loss_pre.item(), loss_cl_skill.item(), loss_cl_question.item(), loss.item()))

        # logging.info(
        #     f"Batch {idx}/{len(train_dataloader)} - Forward: {forward_end - forward_start:.4f}s, "
        #     f"Backward: {backward_end - backward_start:.4f}s"
        # )

        del loss, loss_pre, loss_cl_skill, loss_cl_question
        torch.cuda.empty_cache()

        # mask_unsque = torch.unsqueeze(mask, 2).to(device) # [batch, seqlen, 1]
        mask_unsque = torch.unsqueeze(mask, 2)  # [batch, seqlen, 1]
        predictions_binary = (predictions >= 0.5).float()
        prediction_label_mask = (predictions_binary == true_label)*mask_unsque
        true_label_num = batch["student_seq_len"].sum()-batch_size
        train_acc_temp = (prediction_label_mask.sum().item()) / (true_label_num.item())
        train_acc_list.append(train_acc_temp)

        y_pred_list = []
        y_true_list = []
        batch_seqlen = batch["student_seq_len"]
        for batch_idx in range(len(batch_seqlen)):
            for seqlen_stu in range(batch_seqlen[batch_idx]-1):
                y_pred_list.append(predictions[batch_idx,seqlen_stu,0].item())
                y_true_list.append(true_label[batch_idx,seqlen_stu,0].item())

        auc_temp = roc_auc_score(y_true_list, y_pred_list)
        train_auc_list.append(auc_temp)

        # print(torch.cuda.memory_allocated() / 1024 ** 2, "MB")

    train_loss = np.mean(train_loss_list)
    train_acc = np.mean(train_acc_list)
    train_auc = np.mean(train_auc_list)
    train_loss_all.append(train_loss)
    train_acc_all.append(train_acc)
    train_auc_all.append(train_auc)

    logging.info("Training finishes at this epoch. It takes {} min".format((time.time() - start_time) / 60))
    logging.info("Training loss: {:.4f}".format(train_loss))
    logging.info("Training acc: {:.4f}".format(train_acc))
    logging.info("Training auc: {:.4f}".format(train_auc))
    logging.info("Training Epoch {}/{} results:".format(epoch, num_epochs))
    logging.info("\n")
    logging.info("Valid")

    val_loss_list = []
    val_acc_list = []
    val_auc_list = []
    #The following is the test/validation set
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(valid_dataloader):

            logging.info("Valid. Batch {}/{}".format(idx, len(valid_dataloader)))
            predictions, loss_cl_skill, loss_cl_question = model(valid_dataset, batch)

            # calculate loss
            # true_label = torch.unsqueeze(batch["student_correct_seq"], 2)[:, 1:, :].float().to(device)
            true_label = torch.unsqueeze(batch["student_correct_seq"], 2)[:, 1:, :].float()
            loss_pre = criterion(predictions, true_label)
            # loss_pre = torch.squeeze(loss_pre, 2).to(device)
            # mask = batch["student_seq_mask"][:, 1:].to(device)
            loss_pre = torch.squeeze(loss_pre, 2)
            mask = batch["student_seq_mask"][:, 1:]
            masked_loss = loss_pre * mask

            masked_loss = masked_loss.sum(dim=1) / mask.sum(dim=1)

            loss_pre = masked_loss.mean()
            loss = loss_pre + lambda_cl * (loss_cl_skill + loss_cl_question)
            logging.info("Valid. loss_pre: {:.4f}; loss_cl_skill: {:.4f}; loss_cl_question: {:.4f}; "
                         "loss: {:.4f}".format(loss_pre.item(), loss_cl_skill.item(), loss_cl_question.item(), loss.item()))

            val_loss_list.append(loss.item())

            # mask_unsque = torch.unsqueeze(mask, 2).to(device)  # [batch, seqlen, 1]
            mask_unsque = torch.unsqueeze(mask, 2)  # [batch, seqlen, 1]
            predictions_binary = (predictions >= 0.5).float()
            prediction_label_mask = (predictions_binary == true_label) * mask_unsque
            true_label_num = batch["student_seq_len"].sum() - batch_size
            valid_acc_temp = (prediction_label_mask.sum().item()) / (true_label_num.item())
            val_acc_list.append(valid_acc_temp)

            # valid  auc
            y_pred_list = []
            y_true_list = []
            batch_seqlen = batch["student_seq_len"]
            for batch_idx in range(len(batch_seqlen)):
                for seqlen_stu in range(batch_seqlen[batch_idx] - 1):
                    y_pred_list.append(predictions[batch_idx, seqlen_stu, 0].item())
                    y_true_list.append(true_label[batch_idx, seqlen_stu, 0].item())

            auc_temp = roc_auc_score(y_true_list, y_pred_list)
            val_auc_list.append(auc_temp)

        val_loss = np.mean(val_loss_list)
        val_acc = np.mean(val_acc_list)
        val_auc = np.mean(val_auc_list)
        val_loss_all.append(val_loss)
        val_acc_all.append(val_acc)
        val_auc_all.append(val_auc)

    logging.info("Valid finishes")
    logging.info("Valid loss: {}".format(val_loss))
    logging.info("Valid acc: {:.4f}".format(val_acc))
    logging.info("Valid auc: {:.4f}".format(val_auc))
    logging.info("\n")


    # Check monitor loss and monitor score for updating
    monitor_loss = min(monitor_loss, val_loss)
    # Learning rate schuduler
    lr_scheduler.step(monitor_loss)

    # update best_value
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch
        logging.info("Update valid results and save model at epoch{}".format(epoch))

        # define saved_model_path
        saved_model_path = os.path.join(current_save_dir, "{}.pt".format(dataset))
        torch.save(model.state_dict(), saved_model_path)
    else:
        logging.info("Best valid results{} and save model at epoch{}".format(best_val_auc,best_epoch))

    # update new_model
    saved_model_path = os.path.join(current_save_dir, "{}.pt".format(dataset_new))
    torch.save(model.state_dict(), saved_model_path)
    logging.info("==================================\n\n")

    early_stopping(val_loss,val_auc)
    if early_stopping.stop_training:
        train_end_time = time.time()
        training_time = train_end_time - train_start_time
        print(f"Total training time: {training_time:.2f} seconds")
        print("Early stopping triggered.")
        break

epochs = range(1, epoch + 2)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss_all, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss_all, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

