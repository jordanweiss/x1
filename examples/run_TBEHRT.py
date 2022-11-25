import sys
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import  pytorch_pretrained_bert as Bert

from  pytorch_pretrained_bert import optimizer
import sklearn.metrics as skm
from torch.utils.data.dataset import Dataset
from utils import *
from model import *
from data import *
from utils import *
import matplotlib as plt

from torch import optim as toptimizer

def get_beta(batch_idx, m, beta_type):
    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta


def trainunsup(e, sched, patienceMetric, MLM=True):
    sampled = datatrain.reset_index(drop=True)
    #

    Dset = TBEHRT_data_formation(token2idx=BertVocab['token2idx'], dataframe=sampled,
                                 max_len=global_params['max_len_seq'], max_age=global_params['max_age'],
                                 year=global_params['age_year'], age_symbol=global_params['age_symbol'],
                                 TestFlag=MLM, noMEM=False, yvocab=YearVocab['token2idx'], expColumn='explabel',
                                 outcomeColumn='label')

    trainload = DataLoader(dataset=Dset, batch_size=global_params['batch_size'], shuffle=True, num_workers=3,
                           sampler=None)

    model.train()
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    oldloss = 10 ** 10
    for step, batch in enumerate(trainload):

        batch = tuple(t.to(global_params['device']) for t in batch)

        age_ids, input_ids, input_idsMLM, posi_ids, segment_ids, year_ids, attMask, masked_label, outcome_label, treatment_label, vaelabel = batch

        masked_lm_loss, lossT, pred, label, treatOut, treatLabel, out, outLabel, treatindex, targreg, vaelosspure = model(
            input_idsMLM,
            age_ids,
            segment_ids,
            posi_ids,
            year_ids,

            attention_mask=attMask,
            masked_lm_labels=masked_label,
            outcomeT=outcome_label,
            treatmentCLabel=treatment_label,
            fullEval=False,
            vaelabel=vaelabel)
        vaeloss = vaelosspure['loss']

        totalL = masked_lm_loss
        if global_params['gradient_accumulation_steps'] > 1:
            totalL = totalL / global_params['gradient_accumulation_steps']
        totalL.backward()
        treatFull = treatOut
        treatLabelFull = treatLabel
        treatLabelFull = treatLabelFull.cpu().detach()

        outFull = out

        outLabelFull = outLabel
        treatindex = treatindex.cpu().detach().numpy()
        zeroind = np.where(treatindex == 0)
        outzero = outFull[0][zeroind]
        outzeroLabel = outLabelFull[zeroind]


        temp_loss += totalL.item()
        tr_loss += totalL.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        if step % 600 == 0:
            print([(keyvae, valvae) for (keyvae, valvae) in vaelosspure.items() if
                   keyvae in ['loss', 'Reconstruction_Loss', 'KLD']])
            if oldloss < vaelosspure['loss']:
                patienceMetric = patienceMetric + 1
                if patienceMetric >= 10:
                    sched.step()
                    print("LR: ", sched.get_lr())
                    patienceMetric = 0
            oldloss = vaelosspure['loss']

        if step % 200 == 0:
            precOut0 = -1
            if len(zeroind[0]) > 0:
                precOut0, _, _ = OutcomePrecision(outzero, outzeroLabel, False)

            print(
                "epoch: {0}| Loss: {1:6.5f}\t| MLM: {2:6.5f}\t| TOutP: {3:6.5f}\t|vaeloss: {4:6.5f}\t|ExpP: {5:6.5f}".format(
                    e, temp_loss / 200, cal_acc(label, pred), precOut0, vaeloss,
                    cal_acc(treatLabelFull, treatFull, False)))
            temp_loss = 0

        if (step + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()
            optim.zero_grad()

    # Save a trained model
    del sampled, Dset, trainload
    return sched, patienceMetric


def train_multi(e, MLM=True):
    sampled = datatrain.reset_index(drop=True)

    Dset = TBEHRT_data_formation(token2idx=BertVocab['token2idx'], dataframe=sampled,
                                 max_len=global_params['max_len_seq'], max_age=global_params['max_age'],
                                 year=global_params['age_year'], age_symbol=global_params['age_symbol'],
                                 TestFlag=MLM, noMEM=False, yvocab=YearVocab['token2idx'], expColumn='explabel',
                                 outcomeColumn='label')
    trainload = DataLoader(dataset=Dset, batch_size=global_params['batch_size'], shuffle=True, num_workers=3,
                           sampler=None)
    model.train()
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(trainload):

        batch = tuple(t.to(global_params['device']) for t in batch)

        age_ids, input_ids, input_idsMLM, posi_ids, segment_ids, year_ids, attMask, masked_label, outcome_label, treatment_label, vaelabel = batch
        masked_lm_loss, lossT, pred, label, treatOut, treatLabel, out, outLabel, treatindex, targreg, vaelosspure = model(
            input_idsMLM,
            age_ids,
            segment_ids,
            posi_ids,
            year_ids,

            attention_mask=attMask,
            masked_lm_labels=masked_label,
            outcomeT=outcome_label,
            treatmentCLabel=treatment_label,
            fullEval=False,
            vaelabel=vaelabel)

        vaeloss = vaelosspure['loss']
        totalL = 1 * (lossT) + 0 + (global_params['fac'] * masked_lm_loss)
        if global_params['gradient_accumulation_steps'] > 1:
            totalL = totalL / global_params['gradient_accumulation_steps']
        totalL.backward()
        treatFull = treatOut
        treatLabelFull = treatLabel
        treatLabelFull = treatLabelFull.cpu().detach()

        outFull = out

        outLabelFull = outLabel
        treatindex = treatindex.cpu().detach().numpy()
        zeroind = np.where(treatindex == 0)
        outzero = outFull[0][zeroind]
        outzeroLabel = outLabelFull[zeroind]

        temp_loss += totalL.item()
        tr_loss += totalL.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        if step % 200 == 0:
            precOut0 = -1

            if len(zeroind[0]) > 0:
                precOut0, _, _ = OutcomePrecision(outzero, outzeroLabel, False)

            print(
                "epoch: {0}| Loss: {1:6.5f}\t| MLM: {2:6.5f}\t| TOutP: {3:6.5f}\t|vaeloss: {4:6.5f}\t|ExpP: {5:6.5f}".format(
                    e, temp_loss / 200, cal_acc(label, pred), precOut0, vaeloss,
                    cal_acc(treatLabelFull, treatFull, False)))

            print([(keyvae, valvae) for (keyvae, valvae) in vaelosspure.items() if
                   keyvae in ['loss', 'Reconstruction_Loss', 'KLD']])
            temp_loss = 0

        if (step + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()
            optim.zero_grad()

    del sampled, Dset, trainload


def evaluation_multi_repeats():
    model.eval()
    y = []
    y_label = []
    t_label = []
    t_output = []
    count = 0
    totalL = 0
    for step, batch in enumerate(testload):
        model.eval()
        count = count + 1
        batch = tuple(t.to(global_params['device']) for t in batch)

        age_ids, input_ids, input_idsMLM, posi_ids, segment_ids, year_ids, attMask, masked_label, outcome_label, treatment_label, vaelabel = batch
        with torch.no_grad():

            masked_lm_loss, lossT, pred, label, treatOut, treatLabel, out, outLabel, treatindex, targreg, vaelosspure = model(
                input_idsMLM,
                age_ids,
                segment_ids,
                posi_ids,
                year_ids,

                attention_mask=attMask,
                masked_lm_labels=masked_label,
                outcomeT=outcome_label,
                treatmentCLabel=treatment_label, vaelabel=vaelabel)

        totalL = totalL + lossT.item() + 0 + (global_params['fac'] * masked_lm_loss)
        treatFull = treatOut
        treatLabelFull = treatLabel
        treatLabelFull = treatLabelFull.detach()
        outFull = out
        outLabelFull = outLabel
        treatindex = treatindex.cpu().detach().numpy()
        outPred = []
        outexpLab = []
        for el in range(global_params['treatments']):
            zeroind = np.where(treatindex == el)
            outPred.append(outFull[el][zeroind])
            outexpLab.append(outLabelFull[zeroind])


        y_label.append(torch.cat(outexpLab))

        y.append(torch.cat(outPred))

        treatOut = treatFull.cpu()
        treatLabel = treatLabelFull.cpu()
        if step % 200 == 0:
            print(step, "tempLoss:", totalL / count)

        t_label.append(treatLabel)
        t_output.append(treatOut)

    y_label = torch.cat(y_label, dim=0)
    y = torch.cat(y, dim=0)
    t_label = torch.cat(t_label, dim=0)
    treatO = torch.cat(t_output, dim=0)

    tempprc, output, label = precision_test(y, y_label, False)
    treatPRC = cal_acc(t_label, treatO, False)
    tempprc2, output2, label2 = roc_auc(y, y_label, False)

    print("LossEval: ", float(totalL) / float(count))

    return tempprc, tempprc2, treatPRC, float(totalL) / float(count)


def fullEval_4analysis_multi(tr, te, filetest):
    if tr:
        sampled = datatrain.reset_index(drop=True)

    if te:
        data = filetest

        if tr:
            sampled = pd.concat([sampled, data]).reset_index(drop=True)
        else:
            sampled = data
    Fulltset = TBEHRT_data_formation(token2idx=BertVocab['token2idx'],
                                     dataframe=sampled,
                                     max_len=global_params['max_len_seq'],
                                     max_age=global_params['max_age'],
                                     year=global_params['age_year'],
                                     age_symbol=global_params['age_symbol'],
                                     TestFlag=True, yvocab=YearVocab['token2idx'],
                                     expColumn='explabel',
                                     outcomeColumn='label')
    fullDataLoad = DataLoader(dataset=Fulltset, batch_size=int(global_params['batch_size']), shuffle=False,
                              num_workers=0)

    model.eval()
    y = []
    y_label = []
    t_label = []
    t_output = []
    count = 0
    totalL = 0
    eps_array = []

    for yyy in range(model_config['num_treatment']):
        y.append([yyy])
        y_label.append([yyy])

    print(y)
    for step, batch in enumerate(fullDataLoad):
        model.eval()

        count = count + 1
        batch = tuple(t.to(global_params['device']) for t in batch)

        age_ids, input_ids, input_idsMLM, posi_ids, segment_ids, year_ids, attMask, masked_label, outcome_label, treatment_label, vaelabel = batch

        with torch.no_grad():
            masked_lm_loss, lossT, pred, label, treatOut, treatLabel, out, outLabel, treatindex, targreg, vaeloss = model(
                input_idsMLM,
                age_ids,
                segment_ids,
                posi_ids,
                year_ids,

                attention_mask=attMask,
                masked_lm_labels=masked_label,
                outcomeT=outcome_label,
                treatmentCLabel=treatment_label, fullEval=True, vaelabel=vaelabel)



        outFull = out
        outLabelFull = outLabel


        for el in range(global_params['treatments']):
            y[el].append(outFull[el].cpu())
            y_label[el].append(outLabelFull.cpu())

        totalL = totalL + (1 * (lossT)).item()

        if step % 200 == 0:
            print(step, "tempLoss:", totalL / count)

        t_label.append(treatLabel)
        t_output.append(treatOut)

    for idd, elem in enumerate(y):
        elem = torch.cat(elem[1:], dim=0)
        y[idd] = elem
    for idd, elem in enumerate(y_label):
        elem = torch.cat(elem[1:], dim=0)
        y_label[idd] = elem

    t_label = torch.cat(t_label, dim=0)
    treatO = torch.cat(t_output, dim=0)
    treatPRC = cal_acc(t_label, treatO)

    print("LossEval: ", float(totalL) / float(count), "prec treat:", treatPRC)
    return y, y_label, t_label, treatO, treatPRC, eps_array


def fullCONV(y, y_label, t_label, treatO):
    def convert_multihot(label, pred):
        label = label.cpu().numpy()
        truepred = pred.detach().cpu().numpy()
        truelabel = label
        newpred = []
        for i, x in enumerate(truelabel):
            temppred = []
            temppred.append(truepred[i][0])
            temppred.append(truepred[i][x[0]])
            newpred.append(temppred)
        return truelabel, np.array(truepred)

    def convert_bin(logits, label, treatmentlabel2):

        output = logits
        label, output = label.cpu().numpy(), output.detach().cpu().numpy()
        label = label[treatmentlabel2[0]]

        return label, output

    treatmentlabel2, treatment2 = convert_multihot(t_label, treatO)
    y = torch.cat(y, dim=0).view(global_params['treatments'], -1)
    y = y.transpose(1, 0)
    y_label = torch.cat(y_label, dim=0).view(global_params['treatments'], -1)
    y_label = y_label.transpose(1, 0)
    y2 = []
    y2label = []
    for i, elem in enumerate(y):
        j, k = convert_bin(elem, y_label[i], treatmentlabel2[i])
        y2.append(k)
        y2label.append(j)
    y2 = np.array(y2)
    y2label = np.array(y2label)
    y2label = np.expand_dims(y2label, -1)

    return y2, y2label, treatmentlabel2, treatment2


file_config = {
    'data': 'test.csv',
}
optim_config = {
    'lr': 1e-4,
    'warmup_proportion': 0.1
}

BertVocab = {}
token2idx = {'MASK': 4,
             'CLS': 3,
             'SEP': 2,
             'UNK': 1,
             'PAD': 0,
             'disease1': 5,
             'disease2': 6,
             'disease3': 7,
             'disease4': 8,
             'disease5': 9,
             'disease6': 10,
             'medication1': 11,
             'medication2': 12,
             'medication3': 13,
             'medication4': 14,
             'medication5': 15,
             'medication6': 16,
             }
idx2token = {}
for x in token2idx:
    idx2token[token2idx[x]] = x
BertVocab['token2idx'] = token2idx
BertVocab['idx2token'] = idx2token

YearVocab = {'token2idx': {'PAD': 0,
                           '1987': 1,
                           '1988': 2,
                           '1989': 3,
                           '1990': 4,
                           '1991': 5,
                           '1992': 6,
                           '1993': 7,
                           '1994': 8,
                           '1995': 9,
                           '1996': 10,
                           '1997': 11,
                           '1998': 12,
                           '1999': 13,
                           '2000': 14,
                           '2001': 15,
                           '2002': 16,
                           '2003': 17,
                           '2004': 18,
                           '2005': 19,
                           '2006': 20,
                           '2007': 21,
                           '2008': 22,
                           '2009': 23,
                           '2010': 24,
                           '2011': 25,
                           '2012': 26,
                           '2013': 27,
                           '2014': 28,
                           '2015': 29,
                           'UNK': 30},
             'idx2token': {0: 'PAD',
                           1: '1987',
                           2: '1988',
                           3: '1989',
                           4: '1990',
                           5: '1991',
                           6: '1992',
                           7: '1993',
                           8: '1994',
                           9: '1995',
                           10: '1996',
                           11: '1997',
                           12: '1998',
                           13: '1999',
                           14: '2000',
                           15: '2001',
                           16: '2002',
                           17: '2003',
                           18: '2004',
                           19: '2005',
                           20: '2006',
                           21: '2007',
                           22: '2008',
                           23: '2009',
                           24: '2010',
                           25: '2011',
                           26: '2012',
                           27: '2013',
                           28: '2014',
                           29: '2015',
                           30: 'UNK'}}
global_params = {
    'batch_size': 128,
    'gradient_accumulation_steps': 1,
    'num_train_epochs': 3,
    'device': 'cuda:0',
    'output_dir': "/",
    'save_model': True,
    'max_len_seq': 250,
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'fac': 0.1,
    'diseaseI': 1,
    'treatments': 2
}

ageVocab, _ = age_vocab(max_age=global_params['max_age'], year=global_params['age_year'],
                        symbol=global_params['age_symbol'])

model_config = {
    'vocab_size': len(BertVocab['token2idx'].keys()),  # number of disease + symbols for word embedding
    'hidden_size': 150,  # word embedding and seg embedding hidden size
    'seg_vocab_size': 2,  # number of vocab for seg embedding
    'age_vocab_size': len(ageVocab.keys()),  # number of vocab for age embedding
    'max_position_embedding': global_params['max_len_seq'],  # maximum number of tokens
    'hidden_dropout_prob': 0.3,  # dropout rate
    'num_hidden_layers': 4,  # number of multi-head attention layers required
    'num_attention_heads': 6,  # number of attention heads
    'attention_probs_dropout_prob': 0.4,  # multi-head attention dropout rate
    'intermediate_size': 108,  # the size of the "intermediate" layer in the transformer encoder
    'hidden_act': 'gelu',
    'initializer_range': 0.02,  # parameter weight initializer range
    'num_treatment': global_params['treatments'],
    'device': global_params['device'],
    'year_vocab_size': len(YearVocab['token2idx'].keys()),

    'batch_size': global_params['batch_size'],
    'MLM': True,
    'poolingSize': 50,
    'unsupVAE': True,
    'unsupSize': ([[3, 2]] * 22),
    'vaelatentdim': 40,
    'vaehidden': 50,
    'vaeinchannels': 39,

}

data = pd.read_csv(file_config['data'])

kf = KFold(n_splits=5, shuffle=True, random_state=2)

print('Begin experiments....')

for cutiter in enumerate(range(5)):
    print("_________________\nfold___" + str(cutiter) + "\n_________________")

    result = next(kf.split(data), None)

    datatrain = data.iloc[result[0]]
    testdata = data.iloc[result[1]]

    tset = TBEHRT_data_formation(token2idx=BertVocab['token2idx'],
                                 dataframe=testdata,
                                 max_len=global_params['max_len_seq'],
                                 max_age=global_params['max_age'],
                                 year=global_params['age_year'],
                                 age_symbol=global_params['age_symbol'],
                                 TestFlag=True,
                                 yvocab=YearVocab['token2idx'],
                                 expColumn='explabel',
                                 outcomeColumn='label')
    testload = DataLoader(dataset=tset, batch_size=int(global_params['batch_size']), shuffle=False, num_workers=0)

    model_config['klpar'] = float(1.0 / (len(datatrain) / global_params['batch_size']))
    conf = BertConfig(model_config)
    model = TBEHRT(conf, 1)

    optim = optimizer.adam(params=list(model.named_parameters()), config=optim_config)

    model_to_save_name = 'TBEHRT_Test' + "__CUT" + str(cutiter) + ".bin"

    import warnings

    warnings.filterwarnings(action='ignore')
    scheduler = toptimizer.lr_scheduler.ExponentialLR(optim, 0.95, last_epoch=-1)
    patience = 0
    best_pre = -100000000000000000000
    LossC = 0.1
    #
    for e in range(5):
        scheduler, patience = trainunsup(e, scheduler, patience)

    for e in range(1):
        train_multi(e)
        auc, auroc, auc2, loss = evaluation_multi_repeats()
        aucreal = -1 * loss
        if aucreal > best_pre:
            patience = 0
            # Save a trained model
            print("** ** * Saving best fine - tuned model ** ** * ")
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(global_params['output_dir'], model_to_save_name)
            create_folder(global_params['output_dir'])
            if global_params['save_model']:
                torch.save(model_to_save.state_dict(), output_model_file)

            best_pre = aucreal
            print("auc-mean: ", aucreal)
        else:
            if patience % 2 == 0 and patience != 0:
                scheduler.step()
                print("LR: ", scheduler.get_lr())

            patience = patience + 1
        print('auprc : {}, auroc : {}, Treat-auc : {}, time: {}'.format(auc, auroc, auc2, "long....."))

    LossC = 0.1
    conf = BertConfig(model_config)
    model = TBEHRT(conf, 1)
    optim = optimizer.VAEadam(params=list(model.named_parameters()), config=optim_config)
    output_model_file = os.path.join(global_params['output_dir'], model_to_save_name)
    model = toLoad(model, output_model_file)

    y, y_label, t_label, treatO, tprc, eps = fullEval_4analysis_multi(False, True, testdata)

    y2, y2label, treatmentlabel2, treatment2 = fullCONV(y, y_label, t_label, treatO)

    NPSaveNAME = 'TBEHRT_Test' + "__CUT" + str(cutiter) + ".npz"

    np.savez('/' + NPSaveNAME,
             outcome=y2,
             outcome_label=y2label, treatment=treatment2, treatment_label=treatmentlabel2,
             epsilon=model.state_dict()['Eps.epsilon'].detach().cpu().numpy())
    del y, y_label, t_label, treatO, tprc, eps, y2, y2label, treatmentlabel2, treatment2, data, datatrain, conf, model, optim, output_model_file, best_pre, LossC,
    print("\n\n\n\n\n")