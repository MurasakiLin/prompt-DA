import torch.utils.data as dataloader
from dataloader import H5Dataset,unlabel_H5Dataset
import torch.optim as optim
from common import *
from network.model import *
from loss.losses import SoftmaxLoss
from torch.utils.tensorboard import SummaryWriter

random_seed = 304  # 304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# --------------------------CUDA check-----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------init Seg---------------
#model_S = prompt_UNet3D_conv(2, 4, True, False).to(device)
#model_S = prompt_UNet3D(2, 4, True, False, False).to(device)
model_S = UNet3D(2, 4, True, False).to(device)
discriminator = Discriminator().to(device)
# --------------Loss---------------------------
criterion_S = SoftmaxLoss(tau=0.1).cuda()
criterion_d = torch.nn.BCELoss()
criterion_c = torch.nn.CrossEntropyLoss()
# setup optimizer
optimizer_S = optim.Adam(model_S.parameters(), lr=lr_S, weight_decay=6e-4, betas=(0.97, 0.999))
optimizer_g = optim.Adam(model_S.down_3.parameters(), lr=lr_S, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(),lr=lr_S, betas=(0.5, 0.999))
scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=step_size_S, gamma=0.1)
scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=step_size_S, gamma=0.1) #step_size_S=5000
scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=step_size_S, gamma=0.1) #step_size_S=5000

# --------------Start Training and Validation ---------------------------
if __name__ == '__main__':
    print(random_seed)
    ckpt_dir = ''
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'logs'))

    # -----------------------Training--------------------------------------
    mri6_data_train = H5Dataset("", mode='train')
    train6loader = dataloader.DataLoader(mri6_data_train, batch_size=6, shuffle=True)
    mri3_data_train = unlabel_H5Dataset("", mode='train')
    train3loader = dataloader.DataLoader(mri3_data_train, batch_size=3, shuffle=True)
    mri12_data_train = unlabel_H5Dataset("", mode='train')
    train12loader = dataloader.DataLoader(mri12_data_train, batch_size=3, shuffle=True)

    mri_data_val = H5Dataset("", mode='val')
    valloader = dataloader.DataLoader(mri_data_val, batch_size=1, shuffle=False)
    print('Rate     | epoch  | Loss seg| DSC_val')
    iter_num = 0
    c_t = torch.tensor([1,1,1,1,1,1,0,0,0,2,2,2]).long().to(device)
    d_t = torch.tensor([1,1,1,1,1,1,0,0,0,0,0,0]).float().to(device)
    g_t = torch.tensor([1, 1, 1, 1, 1, 1]).float().to(device)
    for epoch in range(num_epoch):
        scheduler_S.step(epoch)
        scheduler_g.step(epoch)
        scheduler_d.step(epoch)
        # zero the parameter gradients
        model_S.train()
        discriminator.train()

        for data6,data3,data12 in zip(enumerate(train6loader),enumerate(train3loader),enumerate(train12loader),):
            torch.cuda.empty_cache()
            i,data = data6
            _,image3 = data3
            _,image12 = data12
            images,targets = data
            # Set mode cuda if it is enable, otherwise mode CPU
            images = images.to(device)
            targets = targets.to(device)
            image3 = image3.to(device)
            image12 = image12.to(device)
            image = torch.concat([images,image3, image12], dim=0)
            #g

            if iter_num % 5 == 0:
                outputs, cls, d_out = model_S(image)
                optimizer_g.zero_grad()
                d_cp = discriminator(d_out)
                g_cp = d_cp[6:12, :]
                g_loss = 5*criterion_d(g_cp,g_t.unsqueeze(1))
                g_loss.backward()
                optimizer_g.step()
            #d
            optimizer_d.zero_grad()
            outputs, cls, d_out = model_S(image)
            d_cp = discriminator(d_out)
            d_loss = criterion_d(d_cp, d_t.unsqueeze(1))
            d_loss.backward()
            optimizer_d.step()
            #seg,cls
            optimizer_S.zero_grad()
            torch.cuda.empty_cache()
            outputs, cls, d_out = model_S(image)
            targets_ce = torch.argmax(targets, dim=1)
            outputs1 = outputs[0:6,...]
            loss_seg = criterion_S(outputs1, targets_ce)  # Crossentropy loss for Seg
            loss_seg.backward()
#            optimizer_g.zero_grad()
            optimizer_S.step()


            iter_num = iter_num + 1
            writer.add_scalar('label/seg_loss', loss_seg, iter_num)
            writer.add_scalar('GAN/d_loss', d_loss, iter_num)
            if iter_num % 5==0:
                writer.add_scalar('GAN/g_loss', g_loss, iter_num)
            writer.add_scalar('lr/seg', optimizer_S.param_groups[0]['lr'], iter_num)
            writer.add_scalar('lr/d', optimizer_d.param_groups[0]['lr'], iter_num)
            writer.add_scalar('lr/g', optimizer_g.param_groups[0]['lr'], iter_num)

            if iter_num % 50 == 0:
                for a in range(images.shape[0]):
                    if (images[a, 0:1, :, :, :].max() - images[a, 0:1, :, :, :].min()) != 0:
                        break
                image = images[a, 0, :, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                image = image[31:32, :, :]

                writer.add_image('6m/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                #                outputs = torch.argmax(outputs, dim=1, keepdim=True)
                writer.add_image('6m/Prediction', outputs[a, 0, 31:32, ...] * 50, iter_num)
                labs = torch.argmax(targets, dim=1)[a, 31:32, ...] * 50
                writer.add_image('6m/GroundTruth', labs, iter_num)

                for a in range(image12.shape[0]):
                    if (image12[a, 0:1, :, :,:].max() - image12[a, 0:1, :, :,:].min()) != 0:
                        break
                image = image12[a, 0, :, :,:]
                image = (image - image.min()) / (image.max() - image.min())
                image = image[31:32, :, :]
                writer.add_image('12m/Image12', image, iter_num)
                writer.add_image('12m/Prediction12', outputs[a+9,0,31:32, ...]*50, iter_num)

                for a in range(image3.shape[0]):
                    if (image3[a, 0:1, :, :,:].max() - image3[a, 0:1, :, :,:].min()) != 0:
                        break
                image = image3[a, 0, :, :,:]
                image = (image - image.min()) / (image.max() - image.min())
                image = image[31:32, :, :]
                writer.add_image('3m/Image', image, iter_num)
                writer.add_image('3m/Prediction', outputs[a+6,0,31:32, ...]*50, iter_num)
        # -----------------------Validation------------------------------------
        # no update parameter gradients during validation
        with torch.no_grad():
            for data_val in valloader:
                images_val, targets_val = data_val
                model_S.eval()
                images_val = images_val.to(device)
                targets_val = targets_val.to(device)
                targets_val = torch.argmax(targets_val, dim=1)
                outputs_val, a, b = model_S(images_val)
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

        # -------------------Debug-------------------------
        for param_group in optimizer_S.param_groups:
            print('%0.6f | %6d | %0.5f | %0.5f ' % (
                param_group['lr'], epoch,
                # loss_seg,
                loss_seg.data.cpu().numpy(),
                # dsc for center path
                dsc))

        # Save checkpoint
        if (epoch % step_size_S) == 0 or epoch == (num_epoch - 1) or (epoch % 1000) == 0:
            torch.save(model_S.state_dict(), ckpt_dir + '%s_%s.pth' % (str(epoch).zfill(5), checkpoint_name))

