import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class GCNLayer1(nn.Module):
    def __init__(self, in_feats, out_feats, use_topic=False, new_graph=True):
        super(GCNLayer1, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.use_topic = use_topic
        self.new_graph = new_graph

    def forward(self, inputs, dia_len, topicLabel):
        if self.new_graph:
            # pdb.set_trace()
            adj = self.message_passing_directed_speaker(inputs, dia_len, topicLabel)
        else:
            adj = self.message_passing_wo_speaker(inputs, dia_len, topicLabel)
        x = torch.matmul(adj, inputs)
        x = self.linear(x)
        return x

    def cossim(self, x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)

    def atom_calculate_edge_weight(self, x, y):
        f = self.cossim(x, y)
        if f > 1 and f < 1.05:
            f = 1
        elif f < -1 and f > -1.05:
            f = -1
        elif f >= 1.05 or f <= -1.05:
            print('cos = {}'.format(f))
        return f

    def message_passing_wo_speaker(self, x, dia_len, topicLabel):
        adj = torch.zeros((x.shape[0], x.shape[0])) + torch.eye(x.shape[0])
        start = 0

        for i in range(len(dia_len)):  #
            for j in range(dia_len[i] - 1):
                for pin in range(dia_len[i] - 1 - j):
                    xz = start + j
                    yz = xz + pin + 1
                    f = self.cossim(x[xz], x[yz])
                    if f > 1 and f < 1.05:
                        f = 1
                    elif f < -1 and f > -1.05:
                        f = -1
                    elif f >= 1.05 or f <= -1.05:
                        print('cos = {}'.format(f))
                    Aij = 1 - math.acos(f) / math.pi
                    adj[xz][yz] = Aij
                    adj[yz][xz] = Aij
            start += dia_len[i]

        if self.use_topic:
            for (index, topic_l) in enumerate(topicLabel):
                xz = index
                yz = x.shape[0] + topic_l - 7
                f = self.cossim(x[xz], x[yz])
                if f > 1 and f < 1.05:
                    f = 1
                elif f < -1 and f > -1.05:
                    f = -1
                elif f >= 1.05 or f <= -1.05:
                    print('cos = {}'.format(f))
                Aij = 1 - math.acos(f) / math.pi
                adj[xz][yz] = Aij
                adj[yz][xz] = Aij

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj

    def message_passing_directed_speaker(self, x, dia_len, qmask):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        start = 0
        use_utterance_edge = False
        for (i, len_) in enumerate(dia_len):
            speaker0 = []
            speaker1 = []
            for (j, speaker) in enumerate(qmask[start:start + len_]):
                # if speaker[0] == 1:
                if speaker[0][0] == 1:
                    speaker0.append(j)
                else:
                    speaker1.append(j)
            if use_utterance_edge:
                for j in range(len_ - 1):
                    f = self.atom_calculate_edge_weight(x[start + j], x[start + j + 1])
                    Aij = 1 - math.acos(f) / math.pi
                    adj[start + j][start + j + 1] = Aij
                    adj[start + j + 1][start + j] = Aij
            for k in range(len(speaker0) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker0[k]], x[start + speaker0[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker0[k]][start + speaker0[k + 1]] = Aij
                adj[start + speaker0[k + 1]][start + speaker0[k]] = Aij
            for k in range(len(speaker1) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker1[k]], x[start + speaker1[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker1[k]][start + speaker1[k + 1]] = Aij
                adj[start + speaker1[k + 1]][start + speaker1[k]] = Aij

            start += dia_len[i]

        return adj


class GCN_2Layers(nn.Module):
    def __init__(self, lstm_hid_size, gcn_hid_dim, num_class, dropout, use_topic=False, use_residue=True, return_feature=False):
        super(GCN_2Layers, self).__init__()

        self.lstm_hid_size = lstm_hid_size
        self.gcn_hid_dim = gcn_hid_dim
        self.num_class = num_class
        self.dropout = dropout
        self.use_topic = use_topic
        self.return_feature = return_feature

        self.gcn1 = GCNLayer1(self.lstm_hid_size, self.gcn_hid_dim, self.use_topic)
        self.use_residue = use_residue
        if self.use_residue:
            self.gcn2 = GCNLayer1(self.gcn_hid_dim, self.gcn_hid_dim, self.use_topic)
            self.linear = nn.Linear(self.lstm_hid_size + self.gcn_hid_dim, self.num_class)
        else:
            self.gcn2 = GCNLayer1(self.gcn_hid_dim, self.num_class, self.use_topic)

    def forward(self, x, dia_len, topicLabel):
        x_graph = self.gcn1(x, dia_len, topicLabel)
        if not self.use_residue:
            x = self.gcn2(x_graph, dia_len, topicLabel)
            if self.return_feature:
                print("Error, you should change the state of use_residue")
        else:
            x_graph = self.gcn2(x_graph, dia_len, topicLabel)
            x = torch.cat([x, x_graph], dim=-1)
            if self.return_feature:
                return x
            x = self.linear(x)
        log_prob = F.log_softmax(x, 1)

        return log_prob


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support

        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class TextCNN(nn.Module):
    def __init__(self, input_dim, emb_size=128, in_channels=1, out_channels=128, kernel_heights=[3, 4, 5], dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.embd = nn.Sequential(
            nn.Linear(len(kernel_heights) * out_channels, emb_size),
            nn.ReLU(inplace=True),
        )

    def conv_block(self, input, conv_layer):
        # (batch_size, out_channels, dim, 1)
        conv_out = conv_layer(input)
        activation = F.relu(conv_out.squeeze(3))
        # (batch_size, out_channels)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        return max_out

    def forward(self, frame_x):
        batch_size, seq_len, feat_dim = frame_x.size()  # dia_len, utt_len, batch_size, feat_dim
        frame_x = frame_x.view(batch_size, 1, seq_len, feat_dim)
        max_out1 = self.conv_block(frame_x, self.conv1)
        max_out2 = self.conv_block(frame_x, self.conv2)
        max_out3 = self.conv_block(frame_x, self.conv3)
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        fc_in = self.dropout(all_out)
        embd = self.embd(fc_in)
        return embd


class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, new_graph=False, reason_flag=False):
        super(GCNII, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        if not return_feature:
            self.fcs.append(nn.Linear(nfeat + nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

        self.rnn_layer = 1
        self.rnn = torch.nn.LSTM(nhidden, nhidden, self.rnn_layer)  # 400,200,1
        self.reason_flag = reason_flag

    def cossim(self, x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)

    def forward(self, x, dia_len, qmask):
        if self.new_graph:
            adj = self.message_passing_directed_speaker(x, dia_len, qmask)
        else:
            adj = self.create_big_adj(x, dia_len)
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        h = (torch.zeros_like(layer_inner).unsqueeze(0).repeat(self.rnn_layer, 1, 1),
             torch.zeros_like(layer_inner).unsqueeze(0).repeat(self.rnn_layer, 1, 1))  # [0]  # v1
        for i, con in enumerate(self.convs):
            if self.reason_flag:
                q = layer_inner
                # output, (hn, cn) = rnn(input, (h0, c0))
                layer_inner, h = self.rnn(q.unsqueeze(0), h)
                layer_inner = layer_inner.squeeze(0)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
            # layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            if self.reason_flag:
                layer_inner += q

        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        if self.use_residue:
            layer_inner = torch.cat([x, layer_inner], dim=-1)
        if not self.return_feature:
            layer_inner = self.fcs[-1](layer_inner)
            layer_inner = F.log_softmax(layer_inner, dim=1)
        return layer_inner

    def create_big_adj(self, x, dia_len):
        adj = torch.zeros((x.shape[0], x.shape[0]))
        start = 0
        for i in range(len(dia_len)):
            sub_adj = torch.zeros((dia_len[i], dia_len[i]))
            temp = x[start:start + dia_len[i]]
            temp_len = torch.sqrt(torch.bmm(temp.unsqueeze(1), temp.unsqueeze(2)).squeeze(-1).squeeze(-1))
            temp_len_matrix = temp_len.unsqueeze(1) * temp_len.unsqueeze(0)
            cos_sim_matrix = torch.matmul(temp, temp.permute(1, 0)) / temp_len_matrix
            sim_matrix = torch.acos(cos_sim_matrix * 0.99999)
            sim_matrix = 1 - sim_matrix / math.pi

            sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix

            m_start = start
            n_start = start
            adj[m_start:m_start + dia_len[i], n_start:n_start + dia_len[i]] = sub_adj

            start += dia_len[i]
        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj

    def message_passing_wo_speaker(self, x, dia_len, topicLabel):
        adj = torch.zeros((x.shape[0], x.shape[0])) + torch.eye(x.shape[0])
        start = 0
        for i in range(len(dia_len)):  #
            for j in range(dia_len[i] - 1):
                for pin in range(dia_len[i] - 1 - j):
                    xz = start + j
                    yz = xz + pin + 1
                    f = self.cossim(x[xz], x[yz])
                    if f > 1 and f < 1.05:
                        f = 1
                    elif f < -1 and f > -1.05:
                        f = -1
                    elif f >= 1.05 or f <= -1.05:
                        print('cos = {}'.format(f))
                    Aij = 1 - math.acos(f) / math.pi
                    adj[xz][yz] = Aij
                    adj[yz][xz] = Aij
            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj

    def atom_calculate_edge_weight(self, x, y):
        f = self.cossim(x, y)
        if f > 1 and f < 1.05:
            f = 1
        elif f < -1 and f > -1.05:
            f = -1
        elif f >= 1.05 or f <= -1.05:
            print('cos = {}'.format(f))
        return f

    def message_passing_directed_speaker(self, x, dia_len, qmask):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        start = 0
        use_utterance_edge = False
        for (i, len_) in enumerate(dia_len):
            speaker0 = []
            speaker1 = []
            for (j, speaker) in enumerate(qmask[i][0:len_]):
                if speaker[0] == 1:
                    speaker0.append(j)
                else:
                    speaker1.append(j)
            if use_utterance_edge:
                for j in range(len_ - 1):
                    f = self.atom_calculate_edge_weight(x[start + j], x[start + j + 1])
                    Aij = 1 - math.acos(f) / math.pi
                    adj[start + j][start + j + 1] = Aij
                    adj[start + j + 1][start + j] = Aij
            for k in range(len(speaker0) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker0[k]], x[start + speaker0[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker0[k]][start + speaker0[k + 1]] = Aij
                adj[start + speaker0[k + 1]][start + speaker0[k]] = Aij
            for k in range(len(speaker1) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker1[k]], x[start + speaker1[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker1[k]][start + speaker1[k + 1]] = Aij
                adj[start + speaker1[k + 1]][start + speaker1[k]] = Aij

            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj.cuda()

    def message_passing_relation_graph(self, x, dia_len):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        window_size = 10
        start = 0
        for (i, len_) in enumerate(dia_len):
            edge_set = []
            for k in range(len_):
                left = max(0, k - window_size)
                right = min(len_ - 1, k + window_size)
                edge_set = edge_set + [str(i) + '_' + str(j) for i in range(left, right) for j in range(i + 1, right + 1)]
            edge_set = [[start + int(str_.split('_')[0]), start + int(str_.split('_')[1])] for str_ in list(set(edge_set))]
            for left, right in edge_set:
                f = self.atom_calculate_edge_weight(x[left], x[right])
                Aij = 1 - math.acos(f) / math.pi
                adj[left][right] = Aij
                adj[right][left] = Aij
            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj.cuda()


class GCNII_lyc(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, new_graph=False, reason_flag=False):
        super(GCNII_lyc, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        if not return_feature:
            self.fcs.append(nn.Linear(nfeat + nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

        self.rnn_layer = 1
        self.rnn = torch.nn.LSTM(nhidden, nhidden, self.rnn_layer)  # 400,200,1
        self.reason_flag = reason_flag

    def cossim(self, x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)

    def forward(self, x, dia_len, topicLabel, adj=None, test_label=False):
        if adj is None:
            if self.new_graph:
                adj = self.message_passing_relation_graph(x, dia_len)
            else:
                adj = self.message_passing_wo_speaker(x, dia_len, topicLabel)
        else:
            adj = adj
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))  # 4971,100 << 4971,200
        _layers.append(layer_inner)
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)  # 873,100

        h = (torch.zeros_like(layer_inner).unsqueeze(0).repeat(self.rnn_layer, 1, 1),
             torch.zeros_like(layer_inner).unsqueeze(0).repeat(self.rnn_layer, 1, 1))  # [0]  # v1

        for i, con in enumerate(self.convs):
            if self.reason_flag:
                q = layer_inner
                # (4419,100), ((1,4419,100)*2) << (1,4419,100), ((1,4419,100)*2)
                # output, (hn, cn) = rnn(input, (h0, c0))
                layer_inner, h = self.rnn(q.unsqueeze(0), h)
                layer_inner = layer_inner.squeeze(0)

            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            if self.reason_flag:
                layer_inner += q
                # layer_inner = torch.cat([layer_inner, q], dim=-1)
            if test_label:
                print('# deepGCN layer ' + str(i))
                print(layer_inner.size())
                import numpy as np
                import os
                if not os.path.isdir('../outputs/iemocap/'): os.makedirs('../outputs/iemocap/')
                np.save("../outputs/iemocap/1080_v1_test_output_layer_{}".format(i), layer_inner.data.cpu().numpy())

        if self.use_residue:
            layer_inner = torch.cat([x, layer_inner], dim=-1)  # 300 << 200,100

        if not self.return_feature:
            layer_inner = self.fcs[-1](layer_inner)
            layer_inner = F.log_softmax(layer_inner, dim=1)
        return layer_inner

    def message_passing_wo_speaker(self, x, dia_len, topicLabel):
        adj = torch.zeros((x.shape[0], x.shape[0]))
        start = 0
        for i in range(len(dia_len)):
            sub_adj = torch.zeros((dia_len[i], dia_len[i]))
            temp = x[start:start + dia_len[i]]
            vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1))
            norm_temp = (temp.permute(1, 0) / vec_length)
            cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)
            cos_sim_matrix = cos_sim_matrix * 0.99999
            sim_matrix = torch.acos(cos_sim_matrix)

            d = sim_matrix.sum(1)
            D = torch.diag(torch.pow(d, -0.5))

            sub_adj[:dia_len[i], :dia_len[i]] = D.mm(sim_matrix).mm(D)
            adj[start:start + dia_len[i], start:start + dia_len[i]] = sub_adj
            start += dia_len[i]

        adj = adj.cuda()

        return adj

    def atom_calculate_edge_weight(self, x, y):
        f = self.cossim(x, y)
        if f > 1 and f < 1.05:
            f = 1
        elif f < -1 and f > -1.05:
            f = -1
        elif f >= 1.05 or f <= -1.05:
            print('cos = {}'.format(f))
        return f

    def message_passing_directed_speaker(self, x, dia_len, qmask):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        start = 0
        use_utterance_edge = False
        for (i, len_) in enumerate(dia_len):
            speaker0 = []
            speaker1 = []
            for (j, speaker) in enumerate(qmask[i][0:len_]):
                if speaker[0] == 1:
                    speaker0.append(j)
                else:
                    speaker1.append(j)
            if use_utterance_edge:
                for j in range(len_ - 1):
                    f = self.atom_calculate_edge_weight(x[start + j], x[start + j + 1])
                    Aij = 1 - math.acos(f) / math.pi
                    adj[start + j][start + j + 1] = Aij
                    adj[start + j + 1][start + j] = Aij
            for k in range(len(speaker0) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker0[k]], x[start + speaker0[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker0[k]][start + speaker0[k + 1]] = Aij
                adj[start + speaker0[k + 1]][start + speaker0[k]] = Aij
            for k in range(len(speaker1) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker1[k]], x[start + speaker1[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker1[k]][start + speaker1[k + 1]] = Aij
                adj[start + speaker1[k + 1]][start + speaker1[k]] = Aij

            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj.cuda()

    def message_passing_relation_graph(self, x, dia_len):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        window_size = 10
        start = 0
        for (i, len_) in enumerate(dia_len):
            edge_set = []
            for k in range(len_):
                left = max(0, k - window_size)
                right = min(len_ - 1, k + window_size)
                edge_set = edge_set + [str(i) + '_' + str(j) for i in range(left, right) for j in range(i + 1, right + 1)]
            edge_set = [[start + int(str_.split('_')[0]), start + int(str_.split('_')[1])] for str_ in list(set(edge_set))]
            for left, right in edge_set:
                f = self.atom_calculate_edge_weight(x[left], x[right])
                Aij = 1 - math.acos(f) / math.pi
                adj[left][right] = Aij
                adj[right][left] = Aij
            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj.cuda()
