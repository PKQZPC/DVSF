import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

BATCH_SIZE = 256  # batch size
EPOCH = 300  # number of epoch
FOLD = 5  # = NUM_SAMPLE / number of val samples
Num_class = 2
result_path = './Voip/DVSF/'# output file
Num_layers = 2
LAMDA = 0.05 # temperature 
LR = 0.002 # learning rate
BN_DIM = 300 # batch normalization dimension (qim 300 / pms 400 / qimpms 700)

#best_acc_test = 0

def get_file_list(folder):
    file_list = []
    for file in os.listdir(folder):
        file_list.append(os.path.join(folder, file))
    return file_list


def parse_sample(base_path, file_name):
    file_path = os.path.join(base_path, file_name)
    if not os.path.exists(file_path):
        print(f"The file path does not exist：{file_path}")
        return None
    file = open(file_path, 'r')
    lines = file.readlines()
    sample = []
    for line in lines:
        line = [int(l) for l in line.split()]
        sample.append(line)
    return sample


def save_variable(file_name, variable):
    # Save variables to a file
    file_object = open(file_name, "wb")
    pickle.dump(variable, file_object)
    file_object.close()


def get_alter_loaders():

    File_Embed = "./g729a_Steg_QIM_feat" # stego file path
    File_NoEmbed = "./g729a_0_QIM_feat" # cover file path
    pklfilex = './ch_qim_p_1000ms_all.pkl'



    if not os.path.exists(pklfilex):
        df = pd.read_csv('data/data_SFFN/data_SFFN_train/data_SFFN_7_dim/train_lable.csv', header=None)
        print('################ Embed data for processing ##################')
        file_list_embed = os.listdir(File_Embed)
        file_classes_embed = {}  # Store file name and category information for embedded files
        print('Start traversing the list of embedded files')
        for file in file_list_embed:
            file_name = file.split('_')[0]  # Extract file name
            matching_row = df.loc[df[0] == file_name]
            if not matching_row.empty:
                class_name = matching_row.iloc[0, 2]  # Retrieve the category information corresponding to the file
                if file not in file_classes_embed:  # Check if the file is already in the dictionary
                    file_classes_embed[file] = class_name  # Add file name and category information to the dictionary

        # List of files classified by category
        class_files_embed = {}
        for file, class_name in file_classes_embed.items():
            if class_name not in class_files_embed:
                class_files_embed[class_name] = []
            class_files_embed[class_name].append(file)

        # The total number of initialization files is 0
        total_files = 0

        # Output the number of files for each classname
        for class_name, files in class_files_embed.items():
            num_files = len(files)
            print(f"The number of files in category {class_name} is: {num_files}")
            total_files += num_files
        print(f"The total number of files is: {total_files}")

        print('################ Process Noembed data ##################')
        file_list_noembed = os.listdir(File_NoEmbed)
        file_classes_noembed = {}  
        print('Start traversing the list of non embedded files')
        for file in file_list_noembed:
            class_name = -1
            file_classes_noembed[file] = class_name  
        # List of files classified by category
        class_files_noembed = {}
        for file, class_name in file_classes_noembed.items():
            if class_name not in class_files_noembed:
                class_files_noembed[class_name] = []
            class_files_noembed[class_name].append(file)

        total_files = 0


        for class_name, files in class_files_noembed.items():
            num_files = len(files)
            print(f"The number of files in category {class_name} is: {num_files}")
            total_files += num_files
        print(f"The total number of files is: {total_files}")

        print('################ Data statistics completed, start building dataset ##################')

        total_files = 0
        for class_name, files in class_files_embed.items():
            num_files = len(files)
            print(f"The number of files in category {class_name} is: {num_files}")
            total_files += num_files

        for class_name, files in class_files_noembed.items():
            num_files = len(files)
            print(f"The number of files in category {class_name} is: {num_files}")
            total_files += num_files

        print(f"The total number of files is: {total_files}")
        print('Traverse completed, start dividing dataset ')

        train, val = [], []
        print('################ Building the Embed dataset ##################')
        for class_name, files in class_files_embed.items():
            random.shuffle(files)  # Randomly shuffle file order
            print('class_name', class_name)
            print(f"The total number of category {class_name} files is: {len (files)}")
            val_size = int(len(files) / FOLD) 
            val_files = files[:val_size]  
            train_files = files[val_size:]  

            print(f"The number of test set files for category {class_name} is: {len(val_files)}")
            print(f"The number of train set files for category {class_name} is: {len(val_files)}")

            for file in val_files:
                val.append([parse_sample(File_Embed, file), class_name, 1])
            for file in train_files:
                train.append([parse_sample(File_Embed, file), class_name, 1])

        print('################ Build Noembed dataset ##################')
        for class_name, files in class_files_noembed.items():
            print('Total number of Noembed files:', len(files))
            val_size = int(len(files) / FOLD)  
            print('Number of Noembed test files:', val_size)

            # For non embedded files, they are divided into training and testing sets in an 8:2 ratio
            val_files = files[:val_size]  
            train_files = files[val_size:]  

            for file in val_files:
                val.append([parse_sample(File_NoEmbed, file), 0, 0])

            for file in train_files:
                train.append([parse_sample(File_NoEmbed, file), 0, 0])

        print('################ Construction of dataset completed ##################')

        print("Number of training samples:", len(train))
        print("Number of test samples:", len(val))

        print('################ Shuffle and split data and labels ##################')
        random.shuffle(train)
        random.shuffle(val)

        x_train, y_train, x_val, y_val = [], [], [], []

        for sample in train:
            x_train.append(sample[0]) 
            y_train.append(sample[1:])  

        for sample in val:
            x_val.append(sample[0])  
            y_val.append(sample[1:]) 


        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)

        print("x_train.shape:", x_train.shape) 
        print("y_train.shape", y_train.shape)  
        print("x_val.shape", x_val.shape)  
        print("y_val:", y_val.shape)  

        print('Save the processed data in a pkl file for future reading.')
        with open(pklfilex, 'wb') as f:
            pickle.dump((x_train, y_train, x_val, y_val), f)

    else:
        print("Read the dataset from the saved pkl file")
        with open(pklfilex, 'rb') as f:
            x_train, y_train, x_val, y_val = pickle.load(f)
        print("x_train.shape:", x_train.shape)  
        print("y_train.shape", y_train.shape)  
        print("x_val.shape", x_val.shape)  
        print("y_val:", y_val.shape) 
    return x_train, y_train, x_val, y_val


def convert_to_loader_CL(x_train, y_train, x_val, y_val, batch_size):
    x_train_tensor = torch.Tensor(x_train)
    y_train_tensor = torch.Tensor(y_train)
    x_val_tensor = torch.Tensor(x_val)
    y_val_tensor = torch.Tensor(y_val)


    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    steg_indices = [i for i, label in enumerate(y_train) if label[1] == 1]

    cover_indices = [i for i, label in enumerate(y_train) if label[1] == 0]


    train_steg_dataset = Subset(train_dataset, steg_indices)
    train_cover_dataset = Subset(train_dataset, cover_indices)

    train_steg_loader = DataLoader(train_steg_dataset, batch_size=batch_size, shuffle=True)
    train_cover_loader = DataLoader(train_cover_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    return train_steg_loader, train_cover_loader, val_loader


class HAM(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(HAM, self).__init__()
        
        # Multi head self attention mechanism
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) 

        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Convolutional layer for local feature extraction
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv_combined = nn.Conv1d(2 * d_model, d_model, kernel_size=3, padding=1)

        #self.bn = nn.BatchNorm1d(num_features=BN_DIM)
        
        # Add regularization term for weight decay
        self.weight_decay = 1e-5


    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        

        src = src.transpose(0, 1).transpose(1, 2)  # (S, B, E) -> (B, E, S)
        src2 = self.conv1(src)
        src2 = F.gelu(src2) 
        src2 = self.conv2(src2)
        src_cnn = src + self.dropout2(src2)  
        src_cnn = src_cnn.transpose(1, 2).transpose(0, 1)  # (B, E, S) -> (S, B, E)
        src = src.transpose(1, 2).transpose(0, 1)  # (B, E, S) -> (S, B, E)


        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        concatenated = torch.cat((src_cnn, self.dropout2(src2)), dim=2)
        concatenated = concatenated.transpose(0, 1).transpose(1, 2)

        output = self.conv_combined(concatenated)
        output = output.transpose(1, 2).transpose(0, 1)

        src = src + self.dropout2(output)
        src = self.norm2(src)  
        
        return src

# A regularizer adapted to weak sample learning
def apply_regularization(model, weight_decay):
    for param in model.parameters():
        param.data = param.data - weight_decay * param.data


class HAM_muti(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(HAM_muti, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            apply_regularization(self.layers[i], self.layers[i].weight_decay)

        return output



################  MODEL ################
# Define the PyTorch model
class Model_RUN(nn.Module):
    def __init__(self, num_layers=6):
        super(Model_RUN, self).__init__()
        self.embedding = nn.Embedding(256, 64)
        self.position_embedding = PositionalEncoding(64)
        self.WE_encoder_layer = HAM(d_model=64, nhead=8)
        self.WE_encoder = HAM_muti(self.WE_encoder_layer, num_layers=num_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.long()
        emb_x = self.embedding(x)
        
        emb_x += self.position_embedding(emb_x)

        emb_x = emb_x.view(emb_x.size(0), -1, emb_x.size(3))
        
        emb_x = emb_x.permute(1, 0, 2)  
        outputs = self.WE_encoder(emb_x)
        outputs = self.pooling(outputs.permute(1, 2, 0)).squeeze(2)
        return outputs


class Classifier_CL(nn.Module):
    def __init__(self, num_layers, num_class=Num_class):
        super(Classifier_CL, self).__init__()
        self.model = Model_RUN(num_layers)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_class)

    def forward(self, x):

        x_unsup = self.model(x)
        x_sup_1 = torch.zeros(x_unsup.size(0) // 3, x_unsup.size(1)).to(device)
        x_sup_2 = torch.zeros(x_unsup.size(0) // 3, x_unsup.size(1)).to(device)
        for i in range(x_sup_1.size(0)):
            x_sup_1[i] = x_unsup[3 * i]
            x_sup_2[i] = x_unsup[3 * i + 2]
        x_feats=x_sup_1
        x_sup_1 = self.dropout(x_sup_1)
        x_sup_1 = self.fc(x_sup_1)
        x_sup_1 = F.softmax(x_sup_1, dim=1)
        x_sup_2 = self.dropout(x_sup_2)
        x_sup_2 = self.fc(x_sup_2)
        x_sup_2 = F.softmax(x_sup_2, dim=1)

        return x_unsup, x_sup_1, x_sup_2,x_feats


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return x


####################### cutmix ############################

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    #cut_rat = np.sqrt(1. - lam)
    cut_rat = 1. - lam
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
 
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
 
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2



def compute_CL_loss(y_pred,lamda=LAMDA):
    row = torch.arange(0,y_pred.shape[0],3,device='cuda')
    col = torch.arange(y_pred.shape[0], device='cuda')
    col = torch.where(col % 3 != 0)[0].cuda()
    y_true = torch.arange(0,len(col),2,device='cuda')
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    
    similarities = torch.index_select(similarities,0,row)
    similarities = torch.index_select(similarities,1,col)
    
    # Shielding diagonal matrices, i.e. their own equal losses
    similarities = similarities / lamda
    
    # Divide by temperature hyperparameter
    loss = F.cross_entropy(similarities,y_true)
    return torch.mean(loss)


def train_val_Model_RUN(model, train_steg_loader, train_cover_loader, val_loader, optimizer, loss_fun_sup, num_epochs=EPOCH):

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_num = 0
        for (inputs_1, labels_1), (inputs_0, labels_0) in zip(train_steg_loader, train_cover_loader):
            inputs_1, labels_1, inputs_0, labels_0 = inputs_1.to(device), labels_1.to(device), inputs_0.to(device), labels_0.to(device)

            labels_1 = labels_1[:, 1]
            labels_0 = labels_0[:, 1]
            inputs_size = min(inputs_1.size(0), inputs_0.size(0))


            r = np.random.rand(1) 
            if 0.6 > 0 and r < 0.7:
                # generate mixed sample
                lam = np.random.beta(0.6, 0.6)
                rand_index = torch.randperm(inputs_1.size()[0]).cuda()
                target_a1 = labels_1
                target_b1 = labels_1[rand_index]
                target_a1 = torch.eye(2).to(device)[target_a1.unsqueeze(1).long()].squeeze().to(device)
                target_b1 = torch.eye(2).to(device)[target_b1.unsqueeze(1).long()].squeeze().to(device)
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs_1.size(), lam)
                bby1=0
                bby2=3
                inputs_1[:,  bbx1:bbx2, bby1:bby2] = inputs_1[rand_index,  bbx1:bbx2, bby1:bby2]
                lam1 = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs_1.size()[-1] * inputs_1.size()[-2]))
                

                lam = np.random.beta(0.4, 0.4)
                rand_index = torch.randperm(inputs_0.size()[0]).cuda()
                target_a0 = labels_0
                target_b0 = labels_0[rand_index]
                target_a0 = torch.eye(2).to(device)[target_a0.unsqueeze(1).long()].squeeze().to(device)
                target_b0 = torch.eye(2).to(device)[target_b0.unsqueeze(1).long()].squeeze().to(device)
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs_0.size(), lam)
                bby1=0
                bby2=3
                inputs_0[:,  bbx1:bbx2, bby1:bby2] = inputs_0[rand_index,  bbx1:bbx2, bby1:bby2]
                lam0 = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs_0.size()[-1] * inputs_0.size()[-2]))


                inputs_1 = inputs_1[:inputs_size]
                inputs_0 = inputs_0[:inputs_size]
                labels_1 = labels_1[:inputs_size]
                labels_0 = labels_0[:inputs_size]
                target_a1 =target_a1[:inputs_size]
                target_b1 =target_b1[:inputs_size]
                target_a0 = target_a0[:inputs_size]
                target_b0 = target_b0[:inputs_size]

                input_final = torch.zeros(inputs_size*3, inputs_1.size(1), inputs_1.size(2)).to(device)
                for i in range(inputs_size):
                    input_final[3*i] = inputs_0[i % inputs_size]
                    input_final[3*i+1] = inputs_0[(i+1)  % inputs_size]
                    input_final[3*i+2] = inputs_1[i  % inputs_size]

                optimizer.zero_grad()
                outputs_unsup, outputs_sup_1, outputs_sup_2,_ = model(input_final)

                loss_sup_1 = loss_fun_sup(outputs_sup_1, target_a0) * lam0 + loss_fun_sup(outputs_sup_1, target_b0) * (1. - lam0)
                loss_sup_2 = loss_fun_sup(outputs_sup_2, target_a1) * lam1 + loss_fun_sup(outputs_sup_2, target_b1) * (1. - lam1)


                loss_sup = (loss_sup_1 + loss_sup_2) / 2
                loss_unsup = compute_CL_loss(outputs_unsup)

            else:
                # compute output
                labels_1 = torch.eye(2).to(device)[labels_1.unsqueeze(1).long()].squeeze().to(device)
                labels_0 = torch.eye(2).to(device)[labels_0.unsqueeze(1).long()].squeeze().to(device)
                inputs_size = min(inputs_1.size(0), inputs_0.size(0))
                inputs_1 = inputs_1[:inputs_size]
                inputs_0 = inputs_0[:inputs_size]
                labels_1 = labels_1[:inputs_size]
                labels_0 = labels_0[:inputs_size]
                input_final = torch.zeros(inputs_size*3, inputs_1.size(1), inputs_1.size(2)).to(device)
                for i in range(inputs_size):
                    input_final[3*i] = inputs_0[i % inputs_size]
                    input_final[3*i+1] = inputs_0[(i+1)  % inputs_size]
                    input_final[3*i+2] = inputs_1[i  % inputs_size]

                optimizer.zero_grad()
                outputs_unsup, outputs_sup_1, outputs_sup_2,_ = model(input_final)


                loss_sup_1 = loss_fun_sup(outputs_sup_1, labels_0)
                loss_sup_2 = loss_fun_sup(outputs_sup_2, labels_1)
                loss_sup = (loss_sup_1 + loss_sup_2) / 2
                loss_unsup = compute_CL_loss(outputs_unsup)


            
            loss = loss_sup + loss_unsup
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs_size * 2
            total_num += inputs_size * 2

        epoch_loss = running_loss / total_num
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        model.eval()
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                input_final = torch.zeros(inputs.size(0) * 3, inputs.size(1), inputs.size(2)).to(device)
                for i in range(inputs.size(0)):
                    input_final[3 * i] = inputs[i]
                    input_final[3 * i + 1] = inputs[i]
                    input_final[3 * i + 2] = inputs[i]
                _, outputs_sup_1, outputs_sup_2,_ = model(input_final)
                _, predicted_1 = torch.max(outputs_sup_1, 1)
                _, predicted_2 = torch.max(outputs_sup_2, 1)
                labels = labels.squeeze()
                total_preds += labels.size(0) * 2
                correct_preds += (predicted_1 == labels).sum().item()
                correct_preds += (predicted_2 == labels).sum().item()

        accuracy = correct_preds / total_preds
        print(f"Validation Accuracy: {accuracy:.4f}")

        is_best = accuracy > best_acc
        best_acc = max(accuracy, best_acc)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_acc': best_acc,
            }, is_best, prefix=result_path + '/')

            f = open(os.path.join(result_path, "result.txt"), 'a')
            f.write("loaded best_checkpoint (epoch %d, best_acc %.4f)\n" % (epoch, best_acc))
            f.close()
            testQIM()


def save_checkpoint(state, is_best, prefix):
    if is_best:
        directory = os.path.dirname(prefix)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # save model
        torch.save(state, prefix + 'model_best.pth.tar')
        print('save beat check :' + prefix + 'model_best.pth.tar')


def parse_sample_test(file_path):

    file = open(file_path, 'r')
    lines = file.readlines()
    sample = []
    for line in lines:

        line = [int(l) for l in line.split()]
        sample.append(line)
    return sample




# def visualize_with_tsne(features, labels, title="t-SNE Visualization for Steganalysis", save_path=None):
#     tsne = TSNE(n_components=3, random_state=42)
#     feats_3d = tsne.fit_transform(features)
    
#     # Create graphics and 3D coordinate axes
#     fig = plt.figure(figsize=(12, 9), dpi=300)  
#     ax = fig.add_subplot(111, projection='3d')

#     # Determine the unique label and set the corresponding color (directly specify the color of Steg and Cover)
#     unique_labels = np.unique(labels)
#     cmap = plt.cm.tab10  # 使用tab10颜色方案
#     colors = {0: cmap(0), 1: cmap(1)}  # Steg: 1, Cover: 0
#     scatter_points = []
#     for label, color in colors.items():
#         idx = labels == label
#         sc = ax.scatter(feats_3d[idx, 0], feats_3d[idx, 1], feats_3d[idx, 2], 
#                        c=color, label=("Steg" if label else "Cover"), alpha=0.6, edgecolors=color, linewidth=0.5)  # 边缘宽度适当增加
#         scatter_points.append(sc)

#     legend = ax.legend(handles=scatter_points, scatterpoints=1, loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=12) 

#     plt.title(title, fontsize=18, fontweight='bold')
#     # ax.set_xlabel('Component 1', fontsize=14)
#     # ax.set_ylabel('Component 2', fontsize=14)
#     # ax.set_zlabel('Component 3', fontsize=14)

#     plt.tight_layout()  
#     plt.show()

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')  
#         print(f"Figure saved at {save_path}")





def test_model_with_best_checkpoint(File_Embed, File_NoEmbed, emd_rate):

    # import model
    model = Classifier_CL(num_layers=Num_layers)
    model = model.to(device)

    best_checkpoint = torch.load(os.path.join(result_path, 'model_best.pth.tar'))
    print('load bestcheck from :', os.path.join(result_path, 'model_best.pth.tar'))
    model.load_state_dict(best_checkpoint['model'])

    x_test, y_test = get_alter_loaders_test(File_Embed, File_NoEmbed)
    x_test = x_test[:, :, 0:3] ##QIM PMS[:, :, 3:]

    test_loader = convert_to_loader_test(x_test, y_test, BATCH_SIZE)

    model.eval()
    correct_preds = 0
    total_preds = 0

    feats = []
    labelss = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            input_final = torch.zeros(inputs.size(0) * 3, inputs.size(1), inputs.size(2)).to(device)
            for i in range(inputs.size(0)):
                input_final[3 * i] = inputs[i]
                input_final[3 * i + 1] = inputs[i]
                input_final[3 * i + 2] = inputs[i]
            _, outputs_sup_1, outputs_sup_2,Mfeats = model(input_final)
            _, predicted_1 = torch.max(outputs_sup_1, 1)
            _, predicted_2 = torch.max(outputs_sup_2, 1)
            _, labels = torch.max(labels, 1)
            total_preds += labels.size(0) * 2
            correct_preds += (predicted_1 == labels).sum().item()
            correct_preds += (predicted_2 == labels).sum().item()
            feats.append(Mfeats.cpu().numpy())
            labelss.append(labels.cpu().numpy())
    feats = np.concatenate(feats)
    labelss = np.concatenate(labelss)


    accuracy = correct_preds / total_preds
    print(f"test Accuracy: {accuracy:.4f}")
    f = open(os.path.join(result_path, "result.txt"), 'a')
    f.write("test %d%%, acc %.4f\n" % (emd_rate, accuracy))
    f.close()


def convert_to_loader_test(x_test, y_test, batch_size):
    x_test_tensor = torch.Tensor(x_test)
    y_test_tensor = torch.Tensor(y_test)

    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


def get_alter_loaders_test(File_Embed, File_NoEmbed):

    FOLDERS = [
        {"class": 1, "folder": File_Embed},  
        {"class": 0, "folder": File_NoEmbed}  
    ]


    all_files = [(item, folder["class"]) for folder in FOLDERS for item in get_file_list(folder["folder"])]
    random.shuffle(all_files)


    all_samples_x = [(parse_sample_test(item[0])) for item in all_files]
    all_samples_y = [item[1] for item in all_files]  
    np_all_samples_x = np.asarray(all_samples_x)  
    np_all_samples_y = np.asarray(all_samples_y)  

    encoder = OneHotEncoder(categories='auto', sparse_output=False)


    x_test = np_all_samples_x
    y_test_ori = np_all_samples_y

 
    y_test = encoder.fit_transform(y_test_ori.reshape(-1, 1))

    return x_test, y_test


def testQIM():
    print('10%') # embedding rate
    test_model_with_best_checkpoint(
        './Steg',
        './Cover', 10) # test file paths




if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print('\nEPOCH = ', EPOCH)
    print('Num_layers = ', Num_layers)
    x_train, y_train, x_val, y_val = get_alter_loaders()
    x_train = x_train[:, :, 0:3]##QIM PMS[:, :, 3:]
    x_val = x_val[:, :, 0:3]##QIM PMS[:, :, 3:]
    print(x_train.shape)

    y_val = y_val[:, 1:]

    train_steg_loader, train_cover_loader, val_loader = convert_to_loader_CL(x_train, y_train, x_val, y_val, BATCH_SIZE)

    device = torch.device("cuda")

    model = Classifier_CL(num_layers=Num_layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_fun_sup = nn.CrossEntropyLoss()

    train_val_Model_RUN(model, train_steg_loader, train_cover_loader, val_loader, optimizer, loss_fun_sup, num_epochs=EPOCH)
