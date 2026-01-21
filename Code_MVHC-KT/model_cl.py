import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Multi view Supergraph Convolutional Layer
class MultiViewHyperConvLayer(nn.Module):
    def __init__(self, emb_dim):
        super(MultiViewHyperConvLayer, self).__init__()
        # self.dropout = nn.Dropout(0.3)
        self.emb_dim = emb_dim

    def forward(self, skill_embs, HG_qc, HG_cq):
        # skill_embs = [cn, d]
        # HG_cq = [cn, qn]
        # HG_qc = [qn, cn]

        # 1. node -> hyperedge message
        # 1) poi node aggregation
        msg_poi_agg = torch.sparse.mm(HG_qc, skill_embs)  # [qn, d]

        # 2. propagation: hyperedge -> node
        propag_pois_embs = torch.sparse.mm(HG_cq, msg_poi_agg)  # [123, d]
        # propag_pois_embs = self.dropout(propag_pois_embs)

        return propag_pois_embs

# Multi view Supergraph Convolutional Network
class MultiViewHyperConvNetwork(nn.Module):
    def __init__(self, num_layers, emb_dim, dropout):
        super(MultiViewHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.mv_hconv_layer = MultiViewHyperConvLayer(emb_dim)
        self.dropout = dropout

    def forward(self, skill_embs, HG_qc, HG_cq):
        final_skill_embs = [skill_embs]
        for layer_idx in range(self.num_layers):
            skill_embs = self.mv_hconv_layer(skill_embs, HG_qc, HG_cq)  # [cn, d]
            # add residual connection to alleviate over-smoothing issue
            skill_embs = skill_embs + final_skill_embs[-1]
            skill_embs = F.dropout(skill_embs, self.dropout)
            final_skill_embs.append(skill_embs)
        final_skill_embs = torch.mean(torch.stack(final_skill_embs), dim=0)  # [cn, d]

        return final_skill_embs


class GConvNetwork(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, dropout):
        super(GConvNetwork, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        # Define n-layer GCN convolution
        self.conv1 = GCNConv(emb_dim, emb_dim)
        self.conv2 = GCNConv(emb_dim, emb_dim)
        self.conv3 = GCNConv(emb_dim, emb_dim)
        #
        # self.conv4 = GCNConv(emb_dim, emb_dim)
        # self.conv5 = GCNConv(emb_dim, emb_dim)
        #
        # self.conv6 = GCNConv(emb_dim, emb_dim)
        # self.conv7 = GCNConv(emb_dim, emb_dim)
        #
        # self.conv8 = GCNConv(emb_dim, emb_dim)
        # self.conv9 = GCNConv(emb_dim, emb_dim)

    def forward(self, skill_embs, edge_index):
        x = skill_embs

        x = self.conv1(x, edge_index)
        x = F.dropout(x, self.dropout)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.dropout(x, self.dropout)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.dropout(x, self.dropout)
        x = F.relu(x)


        # x = self.conv4(x, edge_index)
        # x = F.dropout(x, self.dropout)
        # x = F.relu(x)

        # x = self.conv5(x, edge_index)
        # x = F.dropout(x, self.dropout)
        # x = F.relu(x)
        #

        # x = self.conv6(x, edge_index)
        # x = F.dropout(x, self.dropout)
        # x = F.relu(x)

        # x = self.conv7(x, edge_index)
        # x = F.dropout(x, self.dropout)
        # x = F.relu(x)
        #

        # x = self.conv8(x, edge_index)
        # x = F.dropout(x, self.dropout)
        # x = F.relu(x)

        # x = self.conv9(x, edge_index)
        # x = F.dropout(x, self.dropout)
        # x = F.relu(x)

        # return skill_embs
        return x

class QuestionConvNetwork(nn.Module):
    def __init__(self, num_layers, dropout):
        super(QuestionConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, question_embs, similar_graph):
        final_question_embs = [question_embs]
        for _ in range(self.num_layers):
            # pois_embs = geo_graph @ pois_embs
            question_embs = torch.sparse.mm(similar_graph, question_embs)
            question_embs = question_embs + final_question_embs[-1]
            question_embs = F.dropout(question_embs, self.dropout)
            final_question_embs.append(question_embs)
        output_question_embs = torch.mean(torch.stack(final_question_embs), dim=0)  # [num_question, d]

        return output_question_embs


class HGC_lstm(nn.Module):
    def __init__(self, num_question, num_skill, emb_dim, correct_emb_dim, ht_dim, num_mv_layers, num_gcn_layers, num_question_layers, num_lstm_layers, temperature, dropout, device):
        super(HGC_lstm, self).__init__()

        # definition
        self.num_question = num_question
        self.num_skill = num_skill
        self.device = device
        self.emb_dim = emb_dim
        self.correct_emb_dim = correct_emb_dim
        self.lstm_dim = 2 * emb_dim + correct_emb_dim
        self.ht_dim = ht_dim
        self.pre_fc_input_dim = ht_dim + emb_dim * 2
        self.num_mv_layers = num_mv_layers  # #The number of layers in the first channel
        self.num_gcn_layers = num_gcn_layers  # The number of layers in the second channel
        self.num_question_layers = num_question_layers # The number of layers in the third channel
        self.num_lstm_layers = num_lstm_layers  # lstm layer
        self.ssl_temp = temperature


        self.question_embedding = nn.Embedding(num_question + 1, self.emb_dim, padding_idx=num_question) # Padding_idx needs to be set based on the maximum length in the dataset
        self.skill_embedding = nn.Embedding(num_skill + 1, self.emb_dim, padding_idx=num_skill)
        self.correct_embedding = nn.Embedding(2+1, correct_emb_dim, padding_idx=2)

        # embedding init
        nn.init.xavier_uniform_(self.question_embedding.weight)
        nn.init.xavier_uniform_(self.skill_embedding.weight)
        nn.init.xavier_uniform_(self.correct_embedding.weight)

        # network
        self.mv_hconv_network = MultiViewHyperConvNetwork(num_mv_layers, emb_dim, dropout)
        self.gcn_network = GConvNetwork(num_gcn_layers, emb_dim, dropout) #
        self.question_conv_network = QuestionConvNetwork(num_question_layers, dropout)
        self.pre_lstm = nn.LSTM(input_size=self.lstm_dim, hidden_size=ht_dim, num_layers=num_lstm_layers, batch_first=True, bidirectional=False)
        # lstm
        self.pre_fc1 = nn.Linear(self.pre_fc_input_dim, ht_dim)
        self.pre_relu = nn.ReLU()
        self.pre_fc2 = nn.Linear(ht_dim, 1)
        self.pre_sigmoid = nn.Sigmoid()

        # gating before
        self.w_gate_one_channal = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))
        self.b_gate_one_channal = nn.Parameter(torch.FloatTensor(1, emb_dim))
        self.w_gate_two_channal = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))
        self.b_gate_two_channal = nn.Parameter(torch.FloatTensor(1, emb_dim))
        self.w_gate_three_channal = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))
        self.b_gate_three_channal = nn.Parameter(torch.FloatTensor(1, emb_dim))
        nn.init.xavier_normal_(self.w_gate_one_channal.data)
        nn.init.xavier_normal_(self.b_gate_one_channal.data)
        nn.init.xavier_normal_(self.w_gate_two_channal.data)
        nn.init.xavier_normal_(self.b_gate_two_channal.data)
        nn.init.xavier_normal_(self.w_gate_three_channal.data)
        nn.init.xavier_normal_(self.b_gate_three_channal.data)

        # gate for adaptive fusion with skill embeddings
        self.one_channal_skill_gate = nn.Sequential(nn.Linear(emb_dim, 1), nn.Sigmoid())
        self.two_channal_skill_gate = nn.Sequential(nn.Linear(emb_dim, 1), nn.Sigmoid())

        # gate for adaptive fusion with question embeddings
        self.one_channal_question_gate = nn.Sequential(nn.Linear(emb_dim, 1), nn.Sigmoid())
        self.three_channal_question_gate = nn.Sequential(nn.Linear(emb_dim, 1), nn.Sigmoid())

        # dropout
        self.dropout = nn.Dropout(dropout)


    def cal_loss_infonce_new(self, emb1, emb2):

        total_loss = 0.0

        #Traverse each time step
        for t in range(emb1.shape[1]):  # seq_len
            emb1_t = emb1[:, t, :]  # (batch_size, embedding_dim)
            emb2_t = emb2[:, t, :]  # (batch_size, embedding_dim)

            pos_score = torch.exp(torch.sum(emb1_t * emb2_t, dim=1) / self.ssl_temp)

            neg_score = torch.sum(torch.exp(torch.mm(emb1_t, emb2_t.T) / self.ssl_temp), axis=1)

            # InfoNCE
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            total_loss += loss / pos_score.shape[0]

        # #Calculate the average loss for each time step
        total_loss /= emb1.shape[1]
        return total_loss

    def cal_loss_infonce(self, emb1, emb2): # input data: (batch，seqlen，embs)
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]

        return loss

    def cal_loss_cl_skill(self, batch_one_channal_skill_embs, batch_two_channal_skill_embs,one_batch_skill_embs, two_batch_skill_embs):

        # normalization
        norm_batch_one_channal_skill_embs = F.normalize(batch_one_channal_skill_embs, p=2, dim=1)
        norm_batch_two_channal_skill_embs = F.normalize(batch_two_channal_skill_embs, p=2, dim=1)

        norm_one_batch_skill_embs = F.normalize(one_batch_skill_embs, p=2, dim=2)
        norm_two_batch_skill_embs = F.normalize(two_batch_skill_embs, p=2, dim=2)

        # calculate loss
        loss_cl_skill = 0.0
        loss_cl_skill += self.cal_loss_infonce(norm_batch_one_channal_skill_embs, norm_batch_two_channal_skill_embs)
        loss_cl_skill += self.cal_loss_infonce_new(norm_one_batch_skill_embs, norm_two_batch_skill_embs)
        loss_cl_skill = loss_cl_skill/2

        return loss_cl_skill

    def cal_loss_cl_question(self, batch_one_channal_question_embs, batch_three_channal_question_embs, one_batch_question_embs, three_batch_question_embs):
        # normalization
        norm_batch_one_channal_question_embs = F.normalize(batch_one_channal_question_embs, p=2, dim=1)
        norm_batch_three_channal_question_embs = F.normalize(batch_three_channal_question_embs, p=2, dim=1)

        norm_one_batch_question_embs = F.normalize(one_batch_question_embs, p=2, dim=2)
        norm_three_batch_question_embs = F.normalize(three_batch_question_embs, p=2, dim=2)

        # calculate loss
        loss_cl_question = 0.0
        loss_cl_question += self.cal_loss_infonce(norm_batch_one_channal_question_embs, norm_batch_three_channal_question_embs)
        loss_cl_question += self.cal_loss_infonce_new(norm_one_batch_question_embs, norm_three_batch_question_embs)
        loss_cl_question = loss_cl_question/2

        return loss_cl_question

    def forward(self, dataset, batch):

        one_gate_skill_embs = torch.multiply(self.skill_embedding.weight[:-1],
                                             torch.sigmoid(torch.matmul(self.skill_embedding.weight[:-1],
                                                                        self.w_gate_one_channal) + self.b_gate_one_channal))

        two_gate_skill_embs = torch.multiply(self.skill_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.skill_embedding.weight[:-1],
                                                                       self.w_gate_two_channal) + self.b_gate_two_channal))

        three_gate_question_embs = torch.multiply(self.question_embedding.weight[:-1],
                                                  torch.sigmoid(torch.matmul(self.question_embedding.weight[:-1],
                                                                             self.w_gate_three_channal) + self.b_gate_three_channal))


        one_channal_skill_embs = self.mv_hconv_network(one_gate_skill_embs, dataset.HG_qc, dataset.HG_cq) # [cn,d]

        one_structural_question_embs = torch.sparse.mm(dataset.HG_qc, one_channal_skill_embs)  # [qn d]

        skill_padding_weight = torch.unsqueeze(self.skill_embedding.weight[-1], 0)
        one_channal_skill_embs = torch.cat((one_channal_skill_embs[:, :], skill_padding_weight[:,:]), dim=0)
        question_padding_weight = torch.unsqueeze(self.question_embedding.weight[-1], 0)
        one_structural_question_embs = torch.cat((one_structural_question_embs[:, :], question_padding_weight[:,:]), dim=0)

        one_batch_skill_embs = one_channal_skill_embs[batch["student_skill_seq"]]  # [BS, len, d]

        one_batch_question_embs = one_structural_question_embs[batch["student_question_seq"]]  # [BS, len, d]

        # skill_embs graph convolutional network
        two_channal_skill_embs = self.gcn_network(two_gate_skill_embs, dataset.GCN_adj)  # [cn, d]

        two_channal_skill_embs = torch.cat((two_channal_skill_embs[:, :], skill_padding_weight[:, :]), dim=0)

        two_batch_skill_embs = two_channal_skill_embs[batch["student_skill_seq"]]  # [BS, len, d]

        # question_embs
        three_channal_question_embs = self.question_conv_network(three_gate_question_embs, dataset.HG_similar_graph) # [qn, d]
        three_channal_question_embs = torch.cat((three_channal_question_embs[:, :], question_padding_weight[:, :]),dim=0)

        three_batch_question_embs = three_channal_question_embs[batch["student_question_seq"]]  # [BS, len, d]

        loss_cl_skill = self.cal_loss_cl_skill(one_channal_skill_embs, two_channal_skill_embs,one_batch_skill_embs, two_batch_skill_embs)
        loss_cl_question = self.cal_loss_cl_question(one_structural_question_embs, three_channal_question_embs, one_batch_question_embs, three_batch_question_embs)

        # normalization
        norm_one_channal_skill_embs = F.normalize(one_channal_skill_embs, p=2, dim=1)
        norm_two_channal_skill_embs = F.normalize(two_channal_skill_embs, p=2, dim=1)

        norm_one_structural_question_embs = F.normalize(one_structural_question_embs, p=2, dim=1)
        norm_three_channal_question_embs = F.normalize(three_channal_question_embs, p=2, dim=1)

        # adaptive fusion for skill embeddings
        one_channal_skill_coef = self.one_channal_skill_gate(norm_one_channal_skill_embs)
        two_channal_skill_coef = self.two_channal_skill_gate(norm_two_channal_skill_embs)

        # adaptive fusion for question embeddings
        one_channal_question_coef = self.one_channal_question_gate(norm_one_structural_question_embs)
        three_channal_question_coef = self.three_channal_question_gate(norm_three_channal_question_embs)

        # final fusion for question and skill embeddings
        fusion_skill_embs = one_channal_skill_coef * norm_one_channal_skill_embs + two_channal_skill_coef * norm_two_channal_skill_embs
        fusion_question_embs = one_channal_question_coef * norm_one_structural_question_embs + three_channal_question_coef * norm_three_channal_question_embs

        batch_skill_embs = fusion_skill_embs[batch["student_skill_seq"]]  # [BS, len, d]
        batch_question_embs = fusion_question_embs[batch["student_question_seq"]]  # [BS, len, d]

        batch_correct_embs = self.correct_embedding(batch["student_correct_seq"])  # [BS, len, correct_emb_dim]
        input_data_lstm = torch.cat((batch_question_embs[:,:-1,:], batch_skill_embs[:,:-1,:], batch_correct_embs[:,:-1,:]), dim=2) # [BS, len-1, 2*d+correct_emb_dim]

        output_lstm, (hn, cn) = self.pre_lstm(input_data_lstm) # (batch_size, seq_len, hidden_len)

        input_data_fc = torch.cat((batch_question_embs[:, 1:, :], batch_skill_embs[:, 1:, :], output_lstm[:, :, :]), dim=2) #[BS, len-1, 2*d+correct_emb_dim]

        pre_predict = self.pre_relu(self.pre_fc1(input_data_fc))
        predict = self.pre_sigmoid(self.pre_fc2(pre_predict))

        return predict, loss_cl_skill, loss_cl_question