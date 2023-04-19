import argparse
import numpy as np
import datetime
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import IEMOCAPDataset, MELDDataset
from model import LSTMModel, GRUModel, DialogRNNModel, DialogueGNNModel
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from loss import FocalLoss, MaskedNLLLoss


def seed_everything(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(data_path=None, batch_size=32, valid_rate=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset(data_path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset(data_path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(data_path=None, batch_size=32, valid_rate=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path=data_path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path=data_path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_f, dataloader, epoch=0, train_flag=False, optimizer=None, cuda_flag=False, target_names=None, tensorboard=False):
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []

    assert not train_flag or optimizer != None
    if train_flag:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train_flag:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)
        loss = loss_f(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train_flag:
            loss.backward()
            if tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return [], [], float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    all_each = metrics.classification_report(labels, preds, target_names=target_names, digits=4)
    all_acc = ["ACC"]
    for i in range(len(target_names)):
        all_acc.append("{}: {:.4f}".format(target_names[i], accuracy_score(labels[labels == i], preds[labels == i])))

    return all_each, all_acc, avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]


def train_or_eval_graph_model(model, loss_f, dataloader, epoch=0, train_flag=False, optimizer=None, cuda_flag=False, modals=None, target_names=None,
                              test_label=False, tensorboard=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda_flag: ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train_flag or optimizer != None
    if train_flag:
        model.train()
    else:
        model.eval()

    seed_everything(seed=args.seed)
    for data in dataloader:
        if train_flag:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        if args.multi_modal:
            if args.mm_fusion_mthd == 'concat':
                if modals == 'avl':
                    textf = torch.cat([acouf, visuf, textf], dim=-1)
                elif modals == 'av':
                    textf = torch.cat([acouf, visuf], dim=-1)
                elif modals == 'vl':
                    textf = torch.cat([visuf, textf], dim=-1)
                elif modals == 'al':
                    textf = torch.cat([acouf, textf], dim=-1)
                else:
                    raise NotImplementedError
            elif args.mm_fusion_mthd == 'gated':
                textf = textf
        else:
            if modals == 'a':
                textf = acouf
            elif modals == 'v':
                textf = visuf
            elif modals == 'l':
                textf = textf
            else:
                raise NotImplementedError

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        if args.multi_modal and args.mm_fusion_mthd in ['gated', 'mfn', 'mfn_only', 'concat_subsequently', 'tfn_only', 'lmf_only', 'concat_only']:
            log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths, acouf, visuf, test_label)
        else:
            log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths)

        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_f(log_prob, label)
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train_flag:
            loss.backward()
            if tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return [], [], float('nan'), float('nan'), [], [], float('nan'), []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    all_each = metrics.classification_report(labels, preds, target_names=target_names, digits=4)
    all_acc = ["ACC"]
    for i in range(len(target_names)):
        all_acc.append("{}: {:.4f}".format(target_names[i], accuracy_score(labels[labels == i], preds[labels == i])))

    return all_each, all_acc, avg_loss, avg_accuracy, labels, preds, avg_fscore, [vids, ei, et, en, el]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--dataset', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--data_dir', type=str, default='../data/iemocap/IEMOCAP_features.pkl', help='dataset dir')

    parser.add_argument('--multi_modal', action='store_true', default=True, help='whether to use multimodal information')

    parser.add_argument('--modals', default='avl', help='modals to fusion: avl')

    parser.add_argument('--mm_fusion_mthd', default='concat_subsequently',
                        help='method to use multimodal information: mfn, concat, gated, concat_subsequently,mfn_only,tfn_only,lmf_only')

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')

    parser.add_argument('--base_model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU/None')

    parser.add_argument('--graph_model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')

    parser.add_argument('--graph_type', default='GDF', help='relation/GCN3/DeepGCN/GF/GF2/GDF')

    parser.add_argument('--graph_construct', default='direct', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--nodal_attention', action='store_true', default=True, help='whether to use nodal attention in graph model')

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')

    parser.add_argument('--use_residue', action='store_true', default=True, help='whether to use residue information or not')

    parser.add_argument('--av_using_lstm', action='store_true', default=False, help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--active_listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--use_crn_speaker', action='store_true', default=True, help='whether to use use crn_speaker embedding')

    parser.add_argument('--speaker_weights', type=str, default='3-0-1', help='speaker weight 0-0-0')

    parser.add_argument('--use_speaker', action='store_true', default=False, help='whether to use speaker embedding')

    parser.add_argument('--reason_flag', action='store_true', default=False, help='reason flag')

    parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--valid_rate', type=float, default=0.0, metavar='valid_rate', help='valid rate, 0.0/0.1')

    parser.add_argument('--modal_weight', type=float, default=1.0, help='modal weight 1/0.7')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=16, help='Deep_GCN_nlayers')

    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec_dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.4, metavar='dropout', help='dropout rate')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha 0.1/0.2')

    parser.add_argument('--lamda', type=float, default=0.5, help='eta 0.5/0')

    parser.add_argument('--gamma', type=float, default=0.5, help='gamma 0.5/1/2')

    parser.add_argument('--windowp', type=int, default=10, help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10, help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')

    parser.add_argument('--loss', default="FocalLoss", help='loss function: FocalLoss/NLLLoss')

    parser.add_argument('--class_weight', action='store_true', default=False, help='use class weights')

    parser.add_argument('--save_model_dir', type=str, default='../outputs/iemocap_demo/', help='saved model dir')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--test_label', action='store_true', default=False, help='whether do test only')

    parser.add_argument('--load_model', type=str, default='../outputs/iemocap_demo/model_4.pkl', help='trained model dir')

    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    parser.add_argument('--patience', type=int, default=5, help='early stop')

    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)
    if args.av_using_lstm:
        name_ = args.mm_fusion_mthd + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + 'using_lstm_' + args.dataset
    else:
        name_ = args.mm_fusion_mthd + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + str(args.Deep_GCN_nlayers) + '_' + args.dataset

    if args.use_speaker:
        name_ = name_ + '_speaker'
    if args.use_modal:
        name_ = name_ + '_modal'

    cuda_flag = torch.cuda.is_available() and not args.no_cuda

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    n_epochs = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = feat2dim['textCNN'] if args.dataset == 'IEMOCAP' else feat2dim['MELD_text']

    if args.multi_modal:
        if args.mm_fusion_mthd == 'concat':
            if modals == 'avl':
                D_m = D_audio + D_visual + D_text
            elif modals == 'av':
                D_m = D_audio + D_visual
            elif modals == 'al':
                D_m = D_audio + D_text
            elif modals == 'vl':
                D_m = D_visual + D_text
            else:
                raise NotImplementedError
        else:
            D_m = D_text
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_text
        else:
            raise NotImplementedError
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100

    n_speakers, n_classes, class_weights, target_names = -1, -1, None, None
    if args.dataset == 'IEMOCAP':
        n_speakers, n_classes = 2, 6
        target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
        class_weights = torch.FloatTensor([1 / 0.086747,
                                           1 / 0.144406,
                                           1 / 0.227883,
                                           1 / 0.160585,
                                           1 / 0.127711,
                                           1 / 0.252668])
    if args.dataset == 'MELD':
        n_speakers, n_classes = 9, 7

        target_names = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']
        class_weights = torch.FloatTensor([1.0 / 0.466750766,
                                           1.0 / 0.122094071,
                                           1.0 / 0.027752748,
                                           1.0 / 0.071544422,
                                           1.0 / 0.171742656,
                                           1.0 / 0.026401153,
                                           1.0 / 0.113714183])

    seed_everything(seed=args.seed)
    if args.graph_model:
        model = DialogueGNNModel(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=n_speakers,
                                 max_seq_len=200,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=not cuda_flag,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 lamda=args.lamda,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 D_m_v=D_visual,
                                 D_m_a=D_audio,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                 dataset=args.dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal,
                                 reason_flag=args.reason_flag,
                                 multi_modal=args.multi_modal,
                                 use_crn_speaker=args.use_crn_speaker,
                                 speaker_weights=args.speaker_weights,
                                 modal_weight=args.modal_weight,
                                 )

        if args.graph_type == 'GDF':
            name = 'MM-DFN'
        elif args.graph_type == 'GF':
            name = 'MMGCN'
        else:
            name = 'GCN'
        print('{} with {} as base model'.format(name, args.base_model))

    else:
        if args.base_model == 'DialogRNN':
            model = DialogRNNModel(D_m, D_g, D_p, D_e, D_h, D_a,
                                   n_classes=n_classes,
                                   listener_state=args.active_listener,
                                   context_attention=args.attention,
                                   dropout_rec=args.rec_dropout,
                                   dropout=args.dropout)

            print('Basic Dialog RNN Model.')


        elif args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h,
                             n_classes=n_classes,
                             dropout=args.dropout)

            print('Basic GRU Model.')


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h,
                              n_classes=n_classes,
                              dropout=args.dropout)

            print('Basic LSTM Model.')

        else:
            print('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
            raise NotImplementedError

    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('Running on the {} features........'.format(modals))

    if cuda_flag:
        # torch.cuda.set_device(0)
        print('Running on GPU')
        class_weights = class_weights.cuda()
        model.cuda()
    else:
        print('Running on CPU')

    if args.loss == 'FocalLoss' and args.graph_model:
        # FocalLoss
        loss_f = FocalLoss(gamma=args.gamma, alpha=class_weights if args.class_weight else None)
    else:
        # NLLLoss
        loss_f = nn.NLLLoss(class_weights if args.class_weight else None) if args.graph_model \
            else MaskedNLLLoss(class_weights if args.class_weight else None)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(data_path=args.data_dir,
                                                                   valid_rate=args.valid_rate,
                                                                   batch_size=batch_size,
                                                                   num_workers=0)
    elif args.dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(data_path=args.data_dir,
                                                                      valid_rate=args.valid_rate,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    else:
        train_loader, valid_loader, test_loader = None, None, None
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    if args.test_label and args.graph_model:
        model = torch.load(args.load_model)
        all_each, all_acc, test_loss, test_acc, test_label, test_pred, test_fscore, _ = train_or_eval_graph_model(model=model,
                                                                                                                  loss_f=loss_f,
                                                                                                                  dataloader=test_loader,
                                                                                                                  train_flag=False,
                                                                                                                  cuda_flag=cuda_flag,
                                                                                                                  modals=args.modals,
                                                                                                                  target_names=target_names,
                                                                                                                  test_label=True)
        print('# test_label,test_pred', len(test_label), len(test_pred))
        # import numpy as np

        np.save("./save_model/iemocap/test_label", test_label)
        np.save("./save_model/iemocap/test_pred", test_pred)

        print(all_each)
        print(all_acc)
        exit(0)

    all_test_fscore, all_test_acc = [], []
    best_epoch, best_epoch2, patience, best_eval_fscore, best_eval_loss = -1, -1, 0, 0, None
    patience2 = 0
    for e in range(n_epochs):
        start_time = time.time()

        if args.graph_model:
            _, _, train_loss, train_acc, _, _, train_fscore, _ = train_or_eval_graph_model(model=model,
                                                                                           loss_f=loss_f,
                                                                                           dataloader=train_loader,
                                                                                           epoch=e,
                                                                                           train_flag=True,
                                                                                           optimizer=optimizer,
                                                                                           cuda_flag=cuda_flag,
                                                                                           modals=args.modals,
                                                                                           target_names=target_names)

            _, _, valid_loss, valid_acc, _, _, valid_fscore, _ = train_or_eval_graph_model(model=model,
                                                                                           loss_f=loss_f,
                                                                                           dataloader=valid_loader,
                                                                                           epoch=e,
                                                                                           cuda_flag=cuda_flag,
                                                                                           modals=args.modals,
                                                                                           target_names=target_names)
            all_each, all_acc, test_loss, test_acc, test_label, test_pred, test_fscore, _ = train_or_eval_graph_model(model=model,
                                                                                                                      loss_f=loss_f,
                                                                                                                      dataloader=test_loader,
                                                                                                                      epoch=e,
                                                                                                                      cuda_flag=cuda_flag,
                                                                                                                      modals=args.modals,
                                                                                                                      target_names=target_names)

        else:
            _, _, train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model=model,
                                                                                        loss_f=loss_f,
                                                                                        dataloader=train_loader,
                                                                                        epoch=e,
                                                                                        train_flag=True,
                                                                                        optimizer=optimizer,
                                                                                        cuda_flag=cuda_flag,
                                                                                        target_names=target_names
                                                                                        )

            _, _, valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model=model,
                                                                                        loss_f=loss_f,
                                                                                        dataloader=valid_loader,
                                                                                        epoch=e,
                                                                                        cuda_flag=cuda_flag,
                                                                                        target_names=target_names)
            all_each, all_acc, test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model=model,
                                                                                                                                    loss_f=loss_f,
                                                                                                                                    dataloader=test_loader,
                                                                                                                                    epoch=e,
                                                                                                                                    cuda_flag=cuda_flag,
                                                                                                                                    target_names=target_names)

        all_test_fscore.append(test_fscore)
        all_test_acc.append(test_acc)
        if args.valid_rate > 0:
            eval_loss, _, eval_fscore = valid_loss, valid_acc, valid_fscore
        else:
            eval_loss, _, eval_fscore = test_loss, test_acc, test_fscore
        if e == 0 or best_eval_fscore < eval_fscore:
            patience = 0
            best_epoch, best_eval_fscore = e, eval_fscore
        else:
            patience += 1
        if best_eval_loss is None:
            best_eval_loss = eval_loss
            best_epoch2 = 0
        else:
            if eval_loss < best_eval_loss:
                best_epoch2, best_eval_loss = e, eval_loss
                patience2 = 0
            else:
                patience2 += 1

        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore,
                       round(time.time() - start_time, 2)))

        print(all_each)
        print(all_acc)

        if patience >= args.patience and patience2 >= args.patience:
            print('Early stoping...', patience, patience2)
            break

    print('Final Test performance...')
    print('Early stoping...', patience, patience2)
    print('Eval-metric: F1, Epoch: {}, best_eval_fscore: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch, best_eval_fscore,
                                                                                                all_test_acc[best_epoch] if best_epoch >= 0 else 0,
                                                                                                all_test_fscore[best_epoch] if best_epoch >= 0 else 0))
