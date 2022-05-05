import random
import numpy as np
import torch,torchvision
import torch.optim
import torch.utils.data
import os
import datetime
import time
import opts_baseline as opts
from log import Log
import logging
import shutil
from torchvision import transforms
from Dataset import *
from sklearn.metrics import accuracy_score

# 跑300 round 数据集
def trainAndValidate(train_loader,val_loader, model, optimizer, lossFunc,schedule,opt):
    max_iter = opt['train_epochs']
    best_cvpr_sbir_acc_multi = 0
    iter = 0
    best_perf_array = 0
    for idx in range(0, max_iter):
        model.train()
        epoch_crossen_loss = []
        
        for datas,targets,img_names in train_loader:
            datas = datas.cuda()
            targets = targets.cuda()

            pred_label = model(datas).cuda()

            if opt['debug']:
                import pdb
                pdb.set_trace()

            cross_entropy_loss = lossFunc(pred_label, targets)
            loss = cross_entropy_loss
            epoch_crossen_loss.append(cross_entropy_loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter = iter + 1

        log.info('cross entropy loss: %f' % ((np.asarray(epoch_crossen_loss).sum()) / len(epoch_crossen_loss)))

        model.eval()
        schedule.step()

        acc = 0
        if opt['test_type']=="single":
            # center crop test
            with torch.no_grad():
                pred_labels = None
                gt_labels = None
                for datas, targets, _ in val_loader:
                    centercrop_pred = model(datas.cuda())
                    centercrop_label = torch.argmax(centercrop_pred, dim=1).cpu()
                    if pred_labels is None:
                        pred_labels = centercrop_label
                        gt_labels = targets
                    else:
                        pred_labels = torch.cat((pred_labels, centercrop_label))
                        gt_labels = torch.cat((gt_labels, targets))
                acc = accuracy_score(gt_labels.numpy(), pred_labels.numpy())
        else:
            # multi crop test
            with torch.no_grad():
                pred_labels = None
                gt_labels = None
                for datas, targets, _ in val_loader:
                    b, c, w, h = datas.shape
                    multicrop_datas = do_multiview_crop(datas)  # 160 x 3 x 224 x 224
                    multicrop_labels = model(multicrop_datas.cuda()).squeeze()
                    avg_label = torch.argmax(multicrop_labels.view(b, 10, -1).mean(dim=1), dim=1).cpu()
                    if pred_labels is None:
                        pred_labels = avg_label
                        gt_labels = targets
                    else:
                        pred_labels = torch.cat((pred_labels, avg_label))
                        gt_labels = torch.cat((gt_labels, targets))

                acc = accuracy_score(gt_labels.numpy(), pred_labels.numpy())

        theTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if acc > best_perf_array:
            best_perf_array = acc
            torch.save(model.state_dict(), os.path.join(opt['checkpoint_path'], 'best.pth.tar'))
        log.info("Time: [%s], Epoch [%d], acc: %.4f, best acc: %.4f\n" % (
                theTime, idx, acc, best_perf_array))

    return  best_perf_array

# # 115 x 3 x 256 x256 -> 1150 x 3 x 224 x 224
def do_multiview_crop(data):
    crop_img_size = 224
    orig_img_size = 256

    n = data.shape[0]
    xs = [0, 0, orig_img_size - crop_img_size, orig_img_size - crop_img_size]  # 0, 0, 32, 32
    ys = [0, orig_img_size - crop_img_size, 0, orig_img_size - crop_img_size]  # 0, 32, 0, 32

    new_data = torch.zeros(n * 10, 3, crop_img_size, crop_img_size)
    y_cen = int((orig_img_size - crop_img_size) * 0.5)  # 16
    x_cen = int((orig_img_size - crop_img_size) * 0.5)  # 16

    for i in range(n):  # 115
        for (k, (x, y)) in enumerate(zip(xs, ys)):
            new_data[i * 10 + k, :, :, :] = data[i, :, y:y + crop_img_size, x:x + crop_img_size]  # 4个角
        new_data[i * 10 + 4, :, :, :] = data[i, :, y_cen:y_cen + crop_img_size, x_cen:x_cen + crop_img_size]  # 中心

        for k in range(5):  # 翻转
            new_data[i * 10 + k + 5, :, :, :] = flip(new_data[i * 10 + k, :, :, :], -1)

    return new_data


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, device=x.device)
    return x[tuple(indices)]

def backup(opt):
    global log
    current_time = time.strftime("%Y-%m-%d-%H-%M")
    
    
    if opt['debug']:
        opt['checkpoint_path'] ="debug_log"
    else:
        opt['checkpoint_path'] = "_".join([opt['checkpoint_path'],opt['backbone'],current_time])
    log_file_name =  os.path.join(opt['checkpoint_path'],current_time+'.log')
    os.makedirs(opt['checkpoint_path'],exist_ok=True)
    log = Log(loggername=__name__, loglevel=logging.DEBUG,file_path=log_file_name).getlog()
    log.info(opt)
    ##copy file
    shutil.copyfile(__file__,os.path.join(opt['checkpoint_path'],os.path.basename(__file__)))

def pretrain_model_func(model,pre_resnet,skip_key=['']):
    state_dict = model.state_dict()
   
    for key in state_dict.keys():
        if key in pre_resnet.keys():
            if key in skip_key:
                log.info('skip key: %s' % key)
            else:
                state_dict[key] = pre_resnet[key]
        else:
            log.info('missing pretrain key: %s'%key)

    model.load_state_dict(state_dict)
    return model


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpus']
    backup(opt)
    if opt['loss_func']=='crossentropy' or opt['loss_func']=='Crossentropy'or opt['loss_func']=='Cross_entropy' or opt['loss_func']=='Cross_Entropy':
        lossFunc = torch.nn.CrossEntropyLoss()
    else:
        print('Could not find %s loss function'%(opt['loss_func']))
        exit()

    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop((224,224)),  # 随机裁剪224
        transforms.RandomHorizontalFlip(),  # 50%几率翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if 'TUBerLin' in opt['trainset_path']:
        train_dataset = TB_ImageSet(opt['trainset_path'],os.path.join(opt['trainset_path'],'../train.txt'), transform=train_transform)
        val_dataset = TB_ImageSet(opt['testset_path'],os.path.join(opt['testset_path'],'../test.txt'), transform=test_transform)
    else:
        train_dataset = ImageSet(opt['trainset_path'],transform=train_transform)
        val_dataset = ImageSet(opt['testset_path'],transform=test_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = opt['batch_size'],
        shuffle=True,
        num_workers =8
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = opt['batch_size'],
        shuffle=False,
        num_workers =8
    )

    if opt['backbone']=='resnet50' or opt['backbone']=='Resnet50':
        model = torchvision.models.resnet50(num_classes=opt['num_classes']).cuda()

        if os.path.exists(opt['pretrain']):
            pretrain_model = torch.load(opt['pretrain'])
            if 'network' in pretrain_model.keys():
                pretrain_model = pretrain_model['network']
            elif 'swav' in opt['pretrain'] or 'deepcluster' in opt['pretrain']:
                model = torch.nn.DataParallel(model)
            model=pretrain_model_func(model,pretrain_model,skip_key=['fc.weight', 'fc.bias'])
        else:
            log.info("could not find pretrain file %s"%(opt['pretrain']))
    elif opt['backbone']=='googlenet':
        model = torchvision.models.googlenet(num_classes = opt['num_classes']).cuda()
        pretrain_model = torch.load(opt['pretrain'])
        model=pretrain_model_func(model,pretrain_model,skip_key=['fc.weight', 'fc.bias'])
    else:
        print('Could not find backbone %s ' % (opt['backbone']))
        exit()
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.SGD([{'params': model.parameters()}], opt['training_lr'], momentum=0.9)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt['step_size'], gamma=opt['step_gamma'], last_epoch=-1)

    best_perf_array = trainAndValidate(train_loader,val_loader, model, optimizer, lossFunc,schedule,opt)


    log_line='best result {:.4f}'.format(best_perf_array)
    print(log_line)
    log.info(log_line + '\n')
    log.info('-------------------------------------------------------- \n')

if __name__ == "__main__":
    opt = opts.parse_opt()
    opt = vars(opt)
    main(opt)
