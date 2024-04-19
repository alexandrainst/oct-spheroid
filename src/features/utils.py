import random

import mat73
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import torch
import torch.nn as nn
import torchvision
from sklearn import metrics
from torch.utils.data import DataLoader

# from torch_trandsforms import Compose, RandomRotate
# from torchvision import ops, transforms
from tqdm import tqdm

from src.models.generate_model import main

random.seed(45)


def normalise_to_positive(vol):
    vol_norm = vol.astype("float32")
    # calculate global mean and standard deviation
    mean, std = vol_norm.mean(), vol_norm.std()
    # print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # global standardization of pixels
    vol_norm = (vol_norm - mean) / std
    # clip pixel values to [-1,1]
    vol_norm = np.clip(vol_norm, -1.0, 1.0)
    # shift from [-1,1] to [0,1] with 0.5 mean
    vol_norm = (vol_norm + 1.0) / 2.0
    # confirm it had the desired effect
    mean, std = vol_norm.mean(), vol_norm.std()
    # print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # print('Min: %.3f, Max: %.3f' % (vol_norm.min(), vol_norm.max()))
    return vol_norm


class SphereHandler:
    def __init__(self, path, thresh, norm=False, vis=False, debug=False):
        self.path = path
        self.thresh = thresh
        self.norm = norm
        self.vis = vis
        self.debug = debug
        self.filename = self.path.split("/")[-1].split(".")[0]

    def remove_specles(self, img, plot=False, count=0):
        test_thresh = (img > self.thresh).astype(float)
        # test_fil = cv.morphologyEx(test_thresh,
        #                               cv.MORPH_TOPHAT,
        #                               kernel)
        test_fil = skimage.measure.label(test_thresh, connectivity=2)
        uni, counts = np.unique(test_fil, return_counts=True)
        idx = [i for i in range(len(counts)) if counts[i] > 10000]
        good_labels = [uni[i] for i in idx]
        bad_labels = [i for i in uni if i not in good_labels]
        for i in bad_labels:
            test_fil[test_fil == i] = 0

        test_thresh_fil = test_thresh.copy()
        test_thresh_fil[test_fil == 0] = 0

        if plot:
            # plt.imshow(test_thresh)

            plt.clf()
            plt.figure(figsize=(30, 30))
            plt.subplot(411)
            plt.imshow(img, cmap="gray")
            plt.subplot(412)
            plt.imshow(test_thresh, cmap="gray")
            plt.subplot(413)
            plt.imshow(test_fil)
            plt.subplot(414)
            plt.imshow(test_thresh_fil, cmap="gray")
            # plt.show()
            plt.savefig(f"thresh_test_{count}_{self.filename}.png")
        return test_thresh_fil

    def find_width(self, img, plot=False, debug=False, count=0):
        if debug:
            test_thresh = self.remove_specles(img, True, count)
        else:
            test_thresh = self.remove_specles(img)
        rays = 0
        row = 0
        end_f = 0
        begin_f = 0
        for i in range(test_thresh.shape[1]):
            begin = 0
            end = 0
            x = test_thresh[i, :]
            for j in range(len(x)):
                if x[j] == 1:
                    begin = j
            for k in range(len(x) - 1, -1, -1):
                if x[k] == 1:
                    end = k
            length = end - begin
            #         rays.append(-length)
            if -length > rays:
                rays = -length
                row = i
                end_f = end
                begin_f = begin
        #     print(rays)
        if plot:
            plt.figure(figsize=(10, 10))
            plt.subplot(121)
            plt.imshow(img, cmap="gray")
            plt.plot(
                [
                    i
                    for i in range(
                        end_f,
                        begin_f,
                    )
                ],
                [
                    row
                    for i in range(
                        end_f,
                        begin_f,
                    )
                ],
            )
            plt.title("Original Image")
            plt.axis("off")
            plt.subplot(122)
            plt.imshow(test_thresh, cmap="gray")
            plt.plot(
                [
                    i
                    for i in range(
                        end_f,
                        begin_f,
                    )
                ],
                [
                    row
                    for i in range(
                        end_f,
                        begin_f,
                    )
                ],
            )
            plt.title("Filtered Image")
            plt.axis("off")
            plt.show()
        return rays

    def plot_sphere(self, volume):
        all_widths = []
        for i in tqdm(volume, position=0, leave=True):
            width = self.find_width(i, self.thresh)
            all_widths.append(width)
        b, e, s1 = self.find_sphere_on_graph(all_widths)
        plt.imshow(volume[:, 500, :], cmap="gray")
        plt.plot([i for i in range(1000)], [int(b) for i in range(1000)])
        plt.plot([i for i in range(1000)], [int(e) for i in range(1000)])
        plt.show()

        return

    def find_sphere_on_graph(self, width_list, plot=False, debug=True):
        #     b = []
        #     e = []
        b = 0
        e = 0
        for i in range(len(width_list)):
            if i < len(width_list) - 150:
                if width_list[i] == 0:
                    b = i
            else:
                if np.mean(width_list[i : i + 10]) > width_list[i] + 200:
                    # print('end sample')
                    #                     l = [-3,-2,-1,0,1,2,3]
                    idx = np.argmin(width_list[i : i + 5])
                    e = i + idx
                    break

        if debug:
            plt.figure(figsize=(10, 5))
            plt.plot(width_list)
            plt.scatter(b, width_list[b], color="red")
            plt.scatter(e, width_list[e], color="green")
            plt.savefig(f"widths_{self.filename}.png")

        if plot:
            plt.figure(figsize=(30, 20))
            # plt.subplot(311); plt.plot(width_list); plt.scatter( begin, width_list[begin] , color='red'); plt.scatter( end, width_list[end], color='green'); plt.title('Width across volume 1 smoothed')
            plt.subplot(311)
            plt.plot(
                width_list
            )  # ; plt.scatter( begin, width_list[begin] , color='red'); plt.scatter( end, width_list[end], color='green'); plt.title('Width across volume 1 smoothed')
            plt.subplot(312)
            plt.plot(width_list[b:e])
        return b, e, width_list[b:e]

    def find_sphere_on_graph_no_histogel(self, width_list, plot=False, debug=True):
        b = []
        e = []
        for i in range(len(width_list[30:])):
            if i == len(width_list[30:]) - 4:
                break
            else:
                if width_list[30:][i] - width_list[30:][i + 1] > 200:
                    b.append(i + 1)
                elif np.mean(width_list[30:][i : i + 5]) > width_list[30:][i] + 100:
                    e.append(i)

        print(b)
        print(e)

        test_b = [i + 30 for i in b]
        test_e = [i + 30 for i in e]
        test_c = []
        for i in range(len(test_e)):
            if i == len(test_e) - 1:
                test_c.append(test_e[i])
                break
            elif test_e[i + 1] - test_e[i] > 1:
                test_c.append(test_e[i])
        #     test_e = [test_e[i] + 30 for i in range(len(test_e)) if i > 0 and i < len(test_e) and test_e[i+1] - test_e[i] > 1 ]

        #     test_e = [i for i in test_e if (i+1) - i > 1]
        print(test_c)
        diff = 1000000
        begin = [i for i in test_b if i < len(width_list) / 2]
        begin = max(begin)
        for i in test_c:
            if i - begin < 0:
                continue
            elif i - begin < diff:
                diff = i - begin
                end = i
        if debug:
            plt.figure(figsize=(10, 5))
            plt.plot(width_list)
            plt.scatter(test_b, [width_list[i] for i in test_b], color="red")
            plt.scatter(test_c, [width_list[i] for i in test_c], color="green")

        if plot:
            plt.figure(figsize=(30, 20))
            # plt.subplot(311); plt.plot(width_list); plt.scatter( begin, width_list[begin] , color='red'); plt.scatter( end, width_list[end], color='green'); plt.title('Width across volume 1 smoothed')
            plt.subplot(311)
            plt.plot(
                width_list
            )  # ; plt.scatter( begin, width_list[begin] , color='red'); plt.scatter( end, width_list[end], color='green'); plt.title('Width across volume 1 smoothed')
            plt.subplot(312)
            plt.plot(width_list[begin:end])
        return begin, end, width_list[begin:end]

    def isolate_sphere(self, vis=True):
        v = self.load_matfile()
        if self.norm:
            v = normalise_to_positive(v)
        # m = np.mean([np.mean(j) for j in v])
        # if m < 40:
        #     t = 50
        # else:
        #     t = 60
        all_widths = []
        counter = 0
        for i in tqdm(v, position=0, leave=True):
            if counter % 50 == 0:
                width = self.find_width(i, False, True, counter)
            else:
                width = self.find_width(i)
            all_widths.append(width)
            counter += 1

        b, e, s = self.find_sphere_on_graph(all_widths, debug=vis)
        if vis:
            # plt.show()
            plt.clf()
            plt.imshow(v[:, 500, :], cmap="gray")
            plt.plot([i for i in range(1000)], [int(b) for i in range(1000)])
            plt.plot([i for i in range(1000)], [int(e) for i in range(1000)])
            plt.savefig(f"sphere_isolated_{self.filename}.png")

        return v[b:e, :, :]

    def load_matfile(self):
        try:
            img = scipy.io.loadmat(self.path)
            return img["tdData"]
        except NotImplementedError:
            try:
                img = mat73.loadmat(self.path)
                return img["tdData"]
            except NotImplementedError:
                print("Cannot load file!!")


# class CustomDataset(Dataset):
#     def __init__(self,df,transform = None):
#         self.labels_dict = {'CTRL':0,'25':2,'100':1}
#         # self.df = df
#         self.df = df[(df['labels'] == 'CTRL') | (df['labels'] == '100')]

#         # self.folder = folder
#         # self.files = glob.glob(os.path.join(folder,'*/*'))
#         self.transform = transform
#         # self.samples = [(i,labels_dict[str(i.split('/')[-2])]) for i in self.files]
#     def __getitem__(self,index):
#         # row = self.df.iloc[index,0]
#         img_name, label = self.df.iloc[index,0], self.df.iloc[index,1]
#         label = self.labels_dict[label]
#         # path, target = self.samples[index]
#         # img_name = img_name.replace('mat','tif')
#         # print(f'Image name:{img_name}')
#         sample = self.load_matfile(img_name)
#         # sample = io.imread(img_name)
#         sample = normalise_to_positive(sample)
#         # cv2.imshow('before',sample[0,:,:])
#         # plt.imshow(sample[150,:,:])
#         # plt.savefig('before.pdf')
#         # plt.show()
#         # sample = resize(sample,(50,100,100))

#         # plt.imshow(sample[150,:,:])
#         # plt.savefig('after.pdf')

#         # cv2.imshow('after',sample[0,:,:])
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#         # plt.imshow(sample[0,:,:])
#         # plt.show()
#         # sample = sample[:300,:650,:650]
#         if self.transform is not None:
#             #data = {'image':sample}
#             sample = self.transform(**{'image':sample})
#             sample = sample['image']
#         transform_backbone = transforms.Compose([
#         transforms.ToTensor(),
#         ops.Permute([1,2,0]),
#     ])
#         sample = transform_backbone(sample)
#         sample = torch.unsqueeze(sample,0)

#         return sample, label
#     def load_matfile(self,path):
#         try:
#             img = scipy.io.loadmat(path)
#             # print('Loaded with scipy')
#         except NotImplementedError:
#             try:
#                 img = mat73.loadmat(path)
#                 # print('Loaded with mat73')
#             except:
#                 print('Cannot load file!!')

#         return img['tdData']
#     def __len__(self):
#         return len(self.df)


# class CustomDataset2D(Dataset):
#     def __init__(self,df,transform = None):
#         self.labels_dict = {'CTRL':0,'25':2,'100':1}
#         # self.df = df
#         # self.df = df[(df['labels'] == 'CTRL') | (df['labels'] == '100')]
#         self.df = df[(df['labels'] == '25')]

#         self.transform = Compose([
#             Rotate((-45, 45), (0, 0), (0, 0), p=1),
#         ], p=1.0)
#         self.images = []
#         self.labels = []
#         for i in tqdm(range(len(self.df)),desc='Preparing dataset',position=0,leave=True,total=len(self.df)):
#             self.images.append(self.get_slices(self.df.iloc[i,0]))
#             self.labels.append(self.df.iloc[i,1])
#         self.images = [resize(i,(4,250,1000)) for i in self.images]
#         self.labels = [self.labels_dict[i] for i in self.labels]
#         for i in tqdm(range(len(self.df)),desc='Saving images',position=0,leave=True,total=len(self.df)):
#             io.imsave(self.df.iloc[i,0].replace('.mat','_4slices.tif'),self.images[i])


#     def __getitem__(self,index):
#         # row = self.df.iloc[index,0]
#         sample = self.images[index]
#         label = self.labels[index]

#         transform_backbone = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[9.713811],std=[10.071295]),
#         ops.Permute([1,2,0])
#     ])
#         sample = transform_backbone(sample)

#         return sample, label
#     def load_matfile(self,path):
#         try:
#             img = scipy.io.loadmat(path)
#             # print('Loaded with scipy')
#         except NotImplementedError:
#             try:
#                 img = mat73.loadmat(path)
#                 # print('Loaded with mat73')
#             except:
#                 print('Cannot load file!!')

#         return img['tdData']
#     def get_slices(self,path):
#         vol = self.load_matfile(path)
#         # vol = normalise_to_positive(vol)
#         vol_thresh = (vol > 0.9).astype(float)
#         median_pos = [int(np.median(i)) for i in np.where(vol_thresh == 1)]
#         sample = self.transform(**{'image':vol})
#         sample = sample['image']
#         sliced_input = np.zeros((4,vol.shape[0],vol.shape[2]))
#         sliced_input[0] = vol[:,median_pos[1],:]
#         sliced_input[1] = vol[:,:,median_pos[2]]
#         sliced_input[2] = sample[:,median_pos[1],:]
#         sliced_input[3] = sample[:,:,median_pos[2]]
#         return sliced_input
#     def __len__(self):
#         return len(self.df)

# class CustomDataset2Dslices(Dataset):
#     def __init__(self,df,n_slices,transform = None):
#         self.n_slices = str(n_slices)
#         self.labels_dict = {'CTRL':0,'25':2,'100':1}
#         # self.df = df
#         self.df = df[(df['labels'] == 'CTRL') | (df['labels'] == '100')]
#         self.transform = Compose([
#             Rotate((-45, 45), (0, 0), (0, 0), p=1),
#         ], p=1.0)

#         ending = '_'+self.n_slices+'slices.tif'
#         self.images = []
#         self.labels = []
#         for i in tqdm(range(len(self.df)),desc='Preparing dataset',position=0,leave=True,total=len(self.df)):
#             filename = self.df.iloc[i,0].replace('.mat',ending)
#             self.images.append(io.imread(filename))
#             self.labels.append(self.df.iloc[i,1])
#         self.labels = [self.labels_dict[i] for i in self.labels]


#     def __getitem__(self,index):
#         # row = self.df.iloc[index,0]
#         sample = self.images[index]
#         label = self.labels[index]

#         transform_backbone = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[9.713811],std=[10.071295]),
#         # ops.Permute([1,2,0])
#     ])
#         sample = transform_backbone(sample)

#         return sample, label

#     def load_matfile(self,path):
#         try:
#             img = scipy.io.loadmat(path)
#             # print('Loaded with scipy')
#         except NotImplementedError:
#             try:
#                 img = mat73.loadmat(path)
#                 # print('Loaded with mat73')
#             except:
#                 print('Cannot load file!!')

#         return img['tdData']
#     def get_slices(self,path):
#         vol = self.load_matfile(path)
#         # vol = normalise_to_positive(vol)
#         vol_thresh = (vol > 0.9).astype(float)
#         median_pos = [int(np.median(i)) for i in np.where(vol_thresh == 1)]
#         sample = self.transform(**{'image':vol})
#         sample = sample['image']
#         sliced_input = np.zeros((4,vol.shape[0],vol.shape[2]))
#         sliced_input[0] = vol[:,median_pos[1],:]
#         sliced_input[1] = vol[:,:,median_pos[2]]
#         sliced_input[2] = sample[:,median_pos[1],:]
#         sliced_input[3] = sample[:,:,median_pos[2]]
#         return sliced_input
#     def __len__(self):
#         return len(self.df)


# def create_splits(data_folder,rs=42):
#     n = glob.glob(os.path.join(data_folder,'*','*','*'))
#     labels = [i.split('/')[-2] for i in n]
#     dictionary = {'file name':n,'labels':labels}
#     df = pd.DataFrame.from_dict(dictionary)

#     target_column = 'labels'

#     train_ratio = 0.7
#     validation_ratio = 0.2
#     test_ratio = 0.1

#     # Split into training, validation, and test sets
#     train_df, temp_df = train_test_split(df, test_size=(validation_ratio + test_ratio),stratify=df[target_column], random_state=rs)
#     validation_df, test_df = train_test_split(temp_df, test_size=test_ratio/(validation_ratio + test_ratio),stratify=temp_df[target_column], random_state=rs)

#     train_df = train_df.sort_values(by='labels')
#     validation_df = validation_df.sort_values(by='labels')
#     test_df = test_df.sort_values(by='labels')


#     train_df.to_csv('./data/train_split.csv', index=False)
#     validation_df.to_csv('./data/val_split.csv', index=False)
#     test_df.to_csv('./data/test_split.csv', index=False)

# # class multiview_resnet(torchvision.models.resnet50()):
# #     def __init__(self,n_views):
# #         super().__init__()
# #         self.n_views = n_views
# #         self.new_first_layer = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# #     def _forward_impl(self):
# #         x = [self.new_first_layer(x[:,i] for i in range(self.n_views))]
# #         x = torch.cat(x,dim=1)
# #         x = self.bn1(x)
# #         x = self.relu(x)
# #         x = self.maxpool(x)

# #         x = self.layer1(x)
# #         x = self.layer2(x)
# #         x = self.layer3(x)
# #         x = self.layer4(x)

# #         x = self.avgpool(x)
# #         x = torch.flatten(x, 1)
# #         x = self.fc(x)


class SeparateChannelsResNet18(nn.Module):
    def __init__(self, num_channels=1, num_classes=2, num_views=4):
        super(SeparateChannelsResNet18, self).__init__()

        # Load the ResNet model without its final fully connected layer
        resnet = torchvision.models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.num_views = num_views

        # Modify the first layer to have three separate layers for each channel
        if self.num_views == 8:
            self.features[0] = nn.ModuleList(
                [
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                ]
            )
        else:
            self.features[0] = nn.ModuleList(
                [
                    nn.Conv2d(num_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
                ]
            )
        # Add a new fully connected layer for the modified input size
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Separate input channels
        if self.num_views == 8:
            channel1, channel2, channel3, channel4, channel5, channel6, channel7, channel8 = torch.chunk(x, 8, dim=1)

            # Process each channel through the modified first layers
            features_channel1 = self.features[0][0](channel1)
            features_channel2 = self.features[0][1](channel2)
            features_channel3 = self.features[0][2](channel3)
            features_channel4 = self.features[0][3](channel4)
            features_channel5 = self.features[0][4](channel5)
            features_channel6 = self.features[0][5](channel6)
            features_channel7 = self.features[0][6](channel7)
            features_channel8 = self.features[0][7](channel8)

            # Concatenate the features
            combined_features = torch.cat(
                [
                    features_channel1,
                    features_channel2,
                    features_channel3,
                    features_channel4,
                    features_channel5,
                    features_channel6,
                    features_channel7,
                    features_channel8,
                ],
                dim=1,
            )
        else:
            channel1, channel2, channel3, channel4 = torch.chunk(x, 4, dim=1)

            # Process each channel through the modified first layers
            features_channel1 = self.features[0][0](channel1)
            features_channel2 = self.features[0][1](channel2)
            features_channel3 = self.features[0][2](channel3)
            features_channel4 = self.features[0][3](channel4)

            # Concatenate the features
            combined_features = torch.cat([features_channel1, features_channel2, features_channel3, features_channel4], dim=1)

        # Continue with the rest of the ResNet model
        x = self.features[1:](combined_features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# def train_multiview(traindataload,valdataload,model_name,epochs,patience,learning_rate):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     device = torch.device('cpu')
#     # net = main('resnet',10,2,1,128)    # net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     # Modify the last layer of the model to match the number of texture classes in DTD

#     # net = torchvision.models.resnet50(
#     #         weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
#     #     )
#     # num_texture_classes = 2
#     # num_features = net.fc.in_features
#     # net.fc = nn.Sequential(
#     #         nn.Linear(num_features, 1024),
#     #         nn.ReLU(),
#     #         nn.Linear(1024, num_texture_classes),
#     #     )

#     # fl = new_first_layer(4)
#     # net.conv1 = fl

#     net = SeparateChannelsResNet18()
#     net = net.float()


#     torch.cuda.empty_cache()
#     criterion = torch.nn.CrossEntropyLoss()
#     # criterion =  nn.SmoothL1Loss()
#     optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate)
#     net = net.to(device)
#     # net = net.float()
#     # epochs = 1000
#     accum_iter = 6
#     best_loss = 0
#     counter = 0
#     train_loss_final = []
#     train_acc_final = []
#     val_loss_final = []
#     val_acc_final = []
#     with open('./results/models/training_log_'+str(model_name)+'.txt','w') as w:
#         for epoch in range(epochs):
#             start_time_epoch = time.time()
#             train_loss = []
#             train_acc = []
#             val_loss = []
#             val_acc = []
#             for step,(img,label) in enumerate(traindataload):
#                 start_time_step = time.time()
#                 net.train()
#     #         sub_batch = 0
#     #         for img in batch:
#                 img = img.to(device, dtype=torch.float)
#                 # img = img.to(device)
#                 label = label.to(device)
#                 optimizer.zero_grad()
#                 pred = net.forward(img.float())
#                 #print(pred,label)
#     #             print(pred.shape,label.shape)
#                 loss = criterion(pred,label)
#                 loss = loss / accum_iter
#                 acc = accuracy(pred=pred,label=label)
#                 train_loss.append(loss.item())
#                 train_acc.append(acc)
#                 loss.backward()
#                 # if ((step + 1) % accum_iter == 0) or (step + 1 == len(traindataload)):
#                 #     optimizer.step()
#                 #     optimizer.zero_grad()
#                 optimizer.step()
#                 end_time_step = time.time()
#     #             sub_batch += 1
#                 print('Epoch [{}/{}], Step [{}/{}], training loss:{:.4f}, time passed:{}'.format(epoch +1,epochs,step+1,len(traindataload),loss.item(),end_time_step - start_time_step))
#                 w.write('Epoch [{}/{}], Step [{}/{}], training loss:{:.4f}, time passed:{} \n'.format(epoch +1,epochs,step+1,len(traindataload),loss.item(),end_time_step - start_time_step))
#             with torch.no_grad():
#                 for step,(img,label) in enumerate(valdataload):
#     #             sub_batch = 0
#     #             for img in batch:
#                     start_time_step = time.time()
#                     net.eval()
#                     img = img.to(device, dtype=torch.float)
#                     # img = img.to(device)
#                     label = label.to(device)
#                     pred = net.forward(img.float())
#                     loss = criterion(pred,label)
#                     acc = accuracy(pred=pred,label=label)
#                     val_loss.append(loss.item())
#                     val_acc.append(acc)
#                     end_time_step = time.time()
#     #                 sub_batch += 1
#                     print('Epoch [{}/{}], Step [{}/{}], val loss:{:.4f}'.format(epoch +1,epochs,step+1,len(valdataload),loss.item(),end_time_step - start_time_step))
#                     w.write('Epoch [{}/{}], Step [{}/{}], val loss:{:.4f}, time passed:{} \n'.format(epoch +1,epochs,step+1,len(valdataload),loss.item(),end_time_step - start_time_step))

#             end_time_epoch = time.time()
#             # print('After Epoch [{}/{}], average training loss:{:.4f}, average val loss:{:.4f}'.format(epoch+1,epochs,np.mean(train_loss),np.mean(val_loss)))
#             print('After Epoch [{}/{}], average training loss:{:.4f}, average training accuracy {:.4f}, average val loss:{:.4f}, average val {:.4f}, time passed:{}'.format(epoch+1,epochs,np.mean(train_loss),np.mean(train_acc),np.mean(val_loss),np.mean(val_acc),end_time_epoch - start_time_epoch))

#             # w.write('After Epoch [{}/{}], average training loss:{:.4f}, average val loss:{:.4f} \n'.format(epoch+1,epochs,np.mean(train_loss),np.mean(val_loss)))
#             w.write(' After Epoch [{}/{}], average training loss:{:.4f}, average training accuracy {:.4f}, average val loss:{:.4f}, average val {:.4f}, time passed:{} \n'.format(epoch+1,epochs,np.mean(train_loss),np.mean(train_acc),np.mean(val_loss),np.mean(val_acc),end_time_epoch - start_time_epoch))

#             current_loss = np.mean(val_loss)
#             if best_loss == 0:
#                 best_loss = current_loss
#                 print('FIRST EPOCH!!')
#                 w.write('FIRST EPOCH!! \n' )
#                 torch.save(net.state_dict(),'./results/models/model_'+str(model_name)+'.pt')
#             elif current_loss < best_loss:
#                 print('GOOD EPOCH!!')
#                 w.write('GOOD EPOCH!! \n')
#                 best_loss = current_loss
#                 torch.save(net.state_dict(),'./results/models/model_'+str(model_name)+'.pt')
#                 counter = 0
#             else:
#                 print('BAD EPOCH!!!')
#                 w.write('BAD EPOCH!! \n')
#                 counter += 1
#                 if counter >= patience:
#                     print('MODEL STARTS TO OVERFIT!!!')
#                     w.write('MODEL STARTS TO OVERFIT!!! \n')
#                     break
#             train_loss_final.append(np.mean(train_loss))
#             train_acc_final.append(np.mean(train_acc))
#             val_loss_final.append(np.mean(val_loss))
#             val_acc_final.append(np.mean(val_acc))

#         fig, (ax1,ax2) = plt.subplots(1,2,figsize=(30,20))
#         ax1.plot(train_loss_final,label = 'Training loss')
#         ax1.plot(val_loss_final,label='Validation loss')
#         ax1.legend(loc = 'upper right')
#         ax2.plot(train_acc_final,label = 'Training accuracy')
#         ax2.plot(val_acc_final,label='Validation accuracy')
#         ax2.legend(loc = 'lower right')


#         # plt.figure(figsize=(30,20))
#         # plt.plot(train_loss_final,label = 'Training loss')
#         # plt.plot(val_loss_final,label='Validation loss')
#         # plt.legend(loc = 'upper right')
#         plt.savefig('./results/models/learning_curves_'+str(model_name)+'.png')

# def train(traindataload,valdataload,model_name,epochs,patience,learning_rate):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     device = torch.device('cpu')
#     # net = main('resnet',10,2,1,128)    # net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     # Modify the last layer of the model to match the number of texture classes in DTD

#     net = torchvision.models.resnet50(
#             weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
#         )
#     num_texture_classes = 2
#     num_features = net.fc.in_features
#     net.fc = nn.Sequential(
#             nn.Linear(num_features, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, num_texture_classes),
#         )
#     new_first_layer = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

# # Replace the original first layer with the new one
#     net.conv1 = new_first_layer
#     #net.fc = nn.Sequential(
#     #nn.Linear(num_features, 1024),
#     #nn.ReLU(),
#     #nn.Linear(1024, 3)
#     #)

#     torch.cuda.empty_cache()
#     criterion = torch.nn.CrossEntropyLoss()
#     # criterion =  nn.SmoothL1Loss()
#     optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate)
#     net = net.to(device)
#     # net = net.float()
#     # epochs = 1000
#     accum_iter = 6
#     best_loss = 0
#     counter = 0
#     train_loss_final = []
#     train_acc_final = []
#     val_loss_final = []
#     val_acc_final = []
#     with open('./results/models/training_log_'+str(model_name)+'.txt','w') as w:
#         for epoch in range(epochs):
#             start_time_epoch = time.time()
#             train_loss = []
#             train_acc = []
#             val_loss = []
#             val_acc = []
#             for step,(img,label) in enumerate(traindataload):
#                 start_time_step = time.time()
#                 net.train()
#     #         sub_batch = 0
#     #         for img in batch:
#                 img = img.to(device, dtype=torch.float)
#                 # img = img.to(device)
#                 label = label.to(device)
#                 optimizer.zero_grad()
#                 pred = net.forward(img)
#                 #print(pred,label)
#     #             print(pred.shape,label.shape)
#                 loss = criterion(pred,label)
#                 loss = loss / accum_iter
#                 acc = accuracy(pred=pred,label=label)
#                 train_loss.append(loss.item())
#                 train_acc.append(acc)
#                 loss.backward()
#                 # if ((step + 1) % accum_iter == 0) or (step + 1 == len(traindataload)):
#                 #     optimizer.step()
#                 #     optimizer.zero_grad()
#                 optimizer.step()
#                 end_time_step = time.time()
#     #             sub_batch += 1
#                 print('Epoch [{}/{}], Step [{}/{}], training loss:{:.4f}, time passed:{}'.format(epoch +1,epochs,step+1,len(traindataload),loss.item(),end_time_step - start_time_step))
#                 w.write('Epoch [{}/{}], Step [{}/{}], training loss:{:.4f}, time passed:{} \n'.format(epoch +1,epochs,step+1,len(traindataload),loss.item(),end_time_step - start_time_step))
#             with torch.no_grad():
#                 for step,(img,label) in enumerate(valdataload):
#     #             sub_batch = 0
#     #             for img in batch:
#                     start_time_step = time.time()
#                     net.eval()
#                     img = img.to(device, dtype=torch.float)
#                     # img = img.to(device)
#                     label = label.to(device)
#                     pred = net.forward(img)
#                     loss = criterion(pred,label)
#                     acc = accuracy(pred=pred,label=label)
#                     val_loss.append(loss.item())
#                     val_acc.append(acc)
#                     end_time_step = time.time()
#     #                 sub_batch += 1
#                     print('Epoch [{}/{}], Step [{}/{}], val loss:{:.4f}'.format(epoch +1,epochs,step+1,len(valdataload),loss.item(),end_time_step - start_time_step))
#                     w.write('Epoch [{}/{}], Step [{}/{}], val loss:{:.4f}, time passed:{} \n'.format(epoch +1,epochs,step+1,len(valdataload),loss.item(),end_time_step - start_time_step))

#             end_time_epoch = time.time()
#             # print('After Epoch [{}/{}], average training loss:{:.4f}, average val loss:{:.4f}'.format(epoch+1,epochs,np.mean(train_loss),np.mean(val_loss)))
#             print('After Epoch [{}/{}], average training loss:{:.4f}, average training accuracy {:.4f}, average val loss:{:.4f}, average val {:.4f}, time passed:{}'.format(epoch+1,epochs,np.mean(train_loss),np.mean(train_acc),np.mean(val_loss),np.mean(val_acc),end_time_epoch - start_time_epoch))

#             # w.write('After Epoch [{}/{}], average training loss:{:.4f}, average val loss:{:.4f} \n'.format(epoch+1,epochs,np.mean(train_loss),np.mean(val_loss)))
#             w.write(' After Epoch [{}/{}], average training loss:{:.4f}, average training accuracy {:.4f}, average val loss:{:.4f}, average val {:.4f}, time passed:{} \n'.format(epoch+1,epochs,np.mean(train_loss),np.mean(train_acc),np.mean(val_loss),np.mean(val_acc),end_time_epoch - start_time_epoch))

#             current_loss = np.mean(val_loss)
#             if best_loss == 0:
#                 best_loss = current_loss
#                 print('FIRST EPOCH!!')
#                 w.write('FIRST EPOCH!! \n' )
#                 torch.save(net.state_dict(),'./results/models/model_'+str(model_name)+'.pt')
#             elif current_loss < best_loss:
#                 print('GOOD EPOCH!!')
#                 w.write('GOOD EPOCH!! \n')
#                 best_loss = current_loss
#                 torch.save(net.state_dict(),'./results/models/model_'+str(model_name)+'.pt')
#                 counter = 0
#             else:
#                 print('BAD EPOCH!!!')
#                 w.write('BAD EPOCH!! \n')
#                 counter += 1
#                 if counter >= patience:
#                     print('MODEL STARTS TO OVERFIT!!!')
#                     w.write('MODEL STARTS TO OVERFIT!!! \n')
#                     break
#             train_loss_final.append(np.mean(train_loss))
#             train_acc_final.append(np.mean(train_acc))
#             val_loss_final.append(np.mean(val_loss))
#             val_acc_final.append(np.mean(val_acc))

#         fig, (ax1,ax2) = plt.subplots(1,2,figsize=(30,20))
#         ax1.plot(train_loss_final,label = 'Training loss')
#         ax1.plot(val_loss_final,label='Validation loss')
#         ax1.legend(loc = 'upper right')
#         ax2.plot(train_acc_final,label = 'Training accuracy')
#         ax2.plot(val_acc_final,label='Validation accuracy')
#         ax2.legend(loc = 'lower right')


#         # plt.figure(figsize=(30,20))
#         # plt.plot(train_loss_final,label = 'Training loss')
#         # plt.plot(val_loss_final,label='Validation loss')
#         # plt.legend(loc = 'upper right')
#         plt.savefig('./results/models/learning_curves_'+str(model_name)+'.png')


# def accuracy(pred,label):
#     # print(pred.shape,label.shape)
#     pred = pred.cpu()
#     label = label.cpu()
#     corr = 0
#     for i in range(pred.shape[0]):
#         _ , p = torch.max(pred,1)
#         if p[i].item() == label[i].item():
#             corr = corr + 1
#     score = (corr/pred.shape[0])*100
#     return score


# def evaluate(testDataLoader,weights):
#     acc = []
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     device = torch.device('cpu')
#     net = main('resnet',10,3,1,128)    # net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     # Modify the last layer of the model to match the number of texture classes in DTD
#     num_texture_classes = 2  # Number of texture classes in DTD
#     num_features = net.fc.in_features
#     net.fc = torch.nn.Linear(num_features, num_texture_classes)
#     net.load_state_dict(torch.load(weights))
#     net = net.to(device)
#     with open('./results/models/evaluation_log_'+str(weights.split('/')[-1].split('.')[0])+'.txt','w') as w:
#         with torch.no_grad():
#             for step,(img,label) in tqdm(enumerate(testDataLoader),desc='Test evaluation',leave=True,position=0,total=len(testDataLoader)):
#                 net.train()
#             #         sub_batch = 0
#             #         for img in batch:
#                 img = img.to(device)
#                 pred = net.forward(img)
#                 single_acc = accuracy(pred=pred,label=label)
#                 w.write(f'True label for this volume: {label.item()}, prediction for this volume: {torch.max(pred,1)[1].item()}. \n')
#                 # print(f'True label for this volume: {label.item()}, prediction for this volume: {torch.max(pred,1)[1].item()}')
#                 acc.append(single_acc)
#         w.write(f'Mean accuracy across test dataset:{np.mean(acc)} \n')
#         print(f'Mean accuracy across test dataset:{np.mean(acc)}')
def accuracy(pred, label):
    # print(pred.shape,label.shape)
    pred = pred.cpu()
    label = label.cpu()
    corr = 0
    for i in range(pred.shape[0]):
        _, p = torch.max(pred, 1)
        if p[i].item() == label[i].item():
            corr = corr + 1
    score = (corr / pred.shape[0]) * 100
    return score


def evaluate2D(testdataset, weights):
    testDataLoader = DataLoader(testdataset, batch_size=1)
    acc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     device = torch.device('cpu')
    #     net = torchvision.models.resnet50(
    #             weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    #         )
    #     num_texture_classes = 2
    #     num_features = net.fc.in_features
    #     net.fc = nn.Sequential(
    #             nn.Linear(num_features, 1024),
    #             nn.ReLU(),
    #             nn.Linear(1024, num_texture_classes),
    #         )
    #     new_first_layer = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # # Replace the original first layer with the new one
    #     net.conv1 = new_first_layer

    net = SeparateChannelsResNet18(num_classes=3)
    # net = net.float()
    net.load_state_dict(torch.load(weights))
    net = net.to(device)
    labels = []
    preds = []
    with open("./results/models/evaluation_log_" + str(weights.split("/")[-1].split(".")[0]) + ".txt", "w") as w:
        with torch.no_grad():
            for step, (img, label) in tqdm(enumerate(testDataLoader), desc="Test evaluation", leave=True, position=0, total=len(testDataLoader)):
                filename = testdataset.df.iloc[step, 0].split("/")[-1]
                net.train()
                #         sub_batch = 0
                #         for img in batch:
                img = img.to(device, dtype=torch.float)
                pred = net.forward(img)
                single_acc = accuracy(pred=pred, label=label)
                preds.append(torch.max(pred, 1)[1].item())
                labels.append(label.item())
                w.write(f"True label for this volume: {label.item()}, prediction for this volume: {torch.max(pred,1)[1].item()} for a volume {filename}. \n")
                # print(f'True label for this volume: {label.item()}, prediction for this volume: {torch.max(pred,1)[1].item()}')
                acc.append(single_acc)
        w.write(f"Mean accuracy across test dataset:{np.mean(acc)} \n")
        print(f"Mean accuracy across test dataset:{np.mean(acc)}")

    confusion_matrix = metrics.confusion_matrix(labels, preds)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    plt.figure(figsize=(20, 20))
    cm_display.plot()
    plt.show()
    plt.savefig("evaluation_confusion_matrix_" + str(weights.split("/")[-1].split(".")[0]) + ".png")


def evaluate(testdataset, weights):
    testDataLoader = DataLoader(testdataset, batch_size=1, shuffle=True)
    acc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     device = torch.device('cpu')
    net = main("resnet", 10, 2, 1, 128)  # net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Modify the last layer of the model to match the number of texture classes in DTD
    num_texture_classes = 2
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, num_texture_classes)

    # Replace the original first layer with the new one
    net.load_state_dict(torch.load(weights))
    net = net.to(device)
    labels = []
    preds = []
    with open("./results/models/evaluation_log_" + str(weights.split("/")[-1].split(".")[0]) + ".txt", "w") as w:
        with torch.no_grad():
            for step, (img, label) in tqdm(enumerate(testDataLoader), desc="Test evaluation", leave=True, position=0, total=len(testDataLoader)):
                filename = testdataset.df.iloc[step, 0].split("/")[-1]
                net.train()
                #         sub_batch = 0
                #         for img in batch:
                img = img.to(device, dtype=torch.float)
                pred = net.forward(img)
                single_acc = accuracy(pred=pred, label=label)
                preds.append(torch.max(pred, 1)[1].item())
                labels.append(label.item())
                w.write(f"True label for this volume: {label.item()}, prediction for this volume: {torch.max(pred,1)[1].item()} for a volume {filename}. \n")
                # print(f'True label for this volume: {label.item()}, prediction for this volume: {torch.max(pred,1)[1].item()}')
                acc.append(single_acc)
        w.write(f"Mean accuracy across test dataset:{np.mean(acc)} \n")
        print(f"Mean accuracy across test dataset:{np.mean(acc)}")

    confusion_matrix = metrics.confusion_matrix(labels, preds)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    plt.figure(figsize=(20, 20))
    cm_display.plot()
    plt.show()
    plt.savefig("evaluation_confusion_matrix_" + str(weights.split("/")[-1].split(".")[0]) + ".png")
