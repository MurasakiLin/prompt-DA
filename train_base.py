import torch.utils.data as dataloader
from dataloader import H5Dataset
import torch.optim as optim
from common import *
from network.model import *
from loss.losses import SoftmaxLoss
from torch.utils.tensorboard import SummaryWriter

random_seed = 304  #304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# --------------------------CUDA check-----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------init Seg---------------
model_S = UNet3D(2,4,True,False).to(device)
# --------------Loss---------------------------
criterion_S = SoftmaxLoss(tau=0.1).cuda()
# setup optimizer
optimizer_S = optim.Adam(model_S.parameters(), lr=lr_S, weight_decay=6e-4, betas=(0.97, 0.999))
scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=step_size_S, gamma=0.1)
# --------------Start Training and Validation ---------------------------
if __name__ == '__main__':
    print(random_seed)
    ckpt_dir = ''
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'logs'))

    #-----------------------Training--------------------------------------
    mri_data_train = H5Dataset("", mode='train')
    trainloader = dataloader.DataLoader(mri_data_train, batch_size=6, shuffle=True)
    mri_data_val = H5Dataset("", mode='val')
    valloader = dataloader.DataLoader(mri_data_val, batch_size=1, shuffle=False)
    print('Rate     | epoch  | Loss seg| DSC_val')
    iter_num = 0
    for epoch in range (num_epoch):
        scheduler_S.step(epoch)
        # zero the parameter gradients
        model_S.train()
        for i, data in enumerate(trainloader):
            images, targets = data
            # Set mode cuda if it is enable, otherwise mode CPU
            images = images.to(device)
            targets = targets.to(device)           
            optimizer_S.zero_grad()
            outputs,a,b = model_S(images)
            targets_ce = torch.argmax(targets, dim=1)
            loss_seg = criterion_S(outputs,targets_ce) #Crossentropy loss for Seg
            loss_seg.backward()
            optimizer_S.step()

            iter_num = iter_num + 1
            writer.add_scalar('label/seg_loss', loss_seg, iter_num)
            writer.add_scalar('lr/seg', optimizer_S.param_groups[0]['lr'], iter_num)

            if iter_num % 50 == 0:
                for a in range(images.shape[0]):
                    if (images[a, 0:1, :, :,:].max() - images[a, 0:1, :, :,:].min()) != 0:
                        break
                image = images[a, 0, :, :,:]
                image = (image - image.min()) / (image.max() - image.min())
                image = image[31:32,:,:]

                writer.add_image('6m/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                #                outputs = torch.argmax(outputs, dim=1, keepdim=True)
                writer.add_image('6m/Prediction', outputs[a,0,31:32, ...] * 50, iter_num)
                labs = torch.argmax(targets, dim=1)[a,31:32, ...] * 50
                writer.add_image('6m/GroundTruth', labs, iter_num)
        # -----------------------Validation------------------------------------
        # no update parameter gradients during validation
        with torch.no_grad():
            for data_val in valloader:
                images_val, targets_val = data_val
                model_S.eval()
                images_val = images_val.to(device)
                targets_val = targets_val.to(device)
                targets_val = torch.argmax(targets_val, dim=1)
                outputs_val,a,b = model_S(images_val)
                _, predicted = torch.max(outputs_val.data, 1)
                # ----------Compute dice-----------
                predicted_val = predicted.data.cpu().numpy()
                targets_val = targets_val.data.cpu().numpy()
                
                dsc = []
                for i in range(1, num_classes):  # ignore Background 0
                    dsc_i = dice(predicted_val, targets_val, i)
                    dsc.append(dsc_i)
                dsc = np.mean(dsc)

                # outputs_val = model_S(images_val)
                # _, predicted = torch.max(outputs_val.data, 1)
                # # ----------Compute dice-----------
                # predicted = predicted.squeeze()
                # targets_val = targets_val.data[0].cpu().numpy()
                # dsc = []
                # for i in range(1, num_classes):  # ignore Background 0
                #     if (np.sum(targets_val[targets_val==i])>0):
                #         dsc_i = dice(predicted, targets_val, i)
                #         dsc.append(dsc_i)
                # dsc = np.mean(dsc)

        #-------------------Debug-------------------------
        for param_group in optimizer_S.param_groups:
            print('%0.6f | %6d | %0.5f | %0.5f ' % (\
                    param_group['lr'], epoch,
                    # loss_seg,
                    loss_seg.data.cpu().numpy(),
                    #dsc for center path
                    dsc))

        #Save checkpoint
        if (epoch % step_size_S) == 0 or epoch == (num_epoch - 1) or (epoch % 1000) == 0:
            torch.save(model_S.state_dict(), ckpt_dir + '%s_%s.pth' % (str(epoch).zfill(5), checkpoint_name))

