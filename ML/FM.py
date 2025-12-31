# library import
import numpy as np 
import pandas as pd 
import random
import time

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import os
import json
import copy
from io import StringIO
import shutil


def split_train_test(data, test_ratio):
#   np.random.seed() 
    shuffled_indices = np.random.permutation(len(data)) 
    
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    test_indices = np.sort(test_indices, axis=0)
    train_indices = np.sort(train_indices, axis=0) 
    return train_indices, test_indices

class FM_order_3(nn.Module):
    def __init__(self, n, k2, k3):
        super().__init__()
        self.lin = nn.Linear(n, 1) # first-order interaction
        self.K_second = nn.Parameter(torch.randn(n, k2),requires_grad=True) # second-order interaction
        self.K_third = nn.Parameter(torch.randn(n, k3),requires_grad=True) # third-order interaction

    def forward(self, x):
    # Second-order terms "Lemma3, Polynomial Networks and Factorization Machines" 
        #homogenius polynomial kernel H2(K_second, x)
        H_2_px = torch.matmul(x, self.K_second).pow(2).sum(1, keepdim=True) 
        #D2(K_second, x)
        D_2_px = torch.matmul(x.pow(2), self.K_second.pow(2)).sum(1, keepdim=True)        
        # (1/2)*(H2-D2)
        Anova_2nd = 0.5*(H_2_px - D_2_px)
    # Third-order terms "Lemma3, Polynomial Networks and Factorization Machines" 
        #homogenius polynomial kernel H3(K_second, x)
        H_3_px = torch.matmul(x, self.K_third).pow(3).sum(1, keepdim=True) 
        #D3(K_third, x)
        D_3_px = torch.matmul(x.pow(3), self.K_third.pow(3)).sum(1, keepdim=True)        
        #D2,1(K_third, x)
        D_21_px_D2 = torch.matmul(x.pow(2), self.K_third.pow(2))
        D_21_px_D1 = torch.matmul(x.pow(1), self.K_third.pow(1))
        D_21_px=torch.mul(D_21_px_D2,D_21_px_D1).sum(1, keepdim=True) 
        # (1/6)*(H3-3D21+2D3)
        Anova_3rd = (1/6)*(H_3_px - 3*D_21_px + 2*D_3_px)
    # First-order term         
        Anova_1st = self.lin(x)
        out = Anova_1st + Anova_2nd + Anova_3rd        
        return out

# for reproducibility
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

####layer = num of bits
###filename = dataset name
def train_valid_order3(sample_train, sample_valid, filename, iteration, model_class=None, model_params=None, num_epochs=1, 
                        early_stop=2000, early_stop_loss = 1, criterion=None, optimizer_class=None, opt_params=None, device=None, file_list=None):
    sample_x_train = sample_train #batch train
    sample_y_valid = sample_valid #batch valid
    
    #number_of_input = sample_train.shape[1]
    model = model_class(**model_params).to(device)
    optimizer_model = optimizer_class(model.parameters(), **opt_params) # torch.optim.Adam
    
    min_loss_val = float('inf')
    min_epoch = 0
    early_stop_count = 0 # if this count == 2000, stop training
    best_model = None
    validation_interval = len(sample_x_train)//1
    
    train_time_data = []
    evaluation_time_data = []
    train_loss_data = []
    valid_loss_data = []
    
    #start_train_time
    start_train_time = time.time()
    
    for epoch in range(num_epochs):
        min_epoch += 1
        model.train()
        for i,(batch_x, batch_y) in enumerate(sample_x_train):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            FOM_predict = model(batch_x)
            loss = criterion(FOM_predict, batch_y)
            model.zero_grad()
            loss.backward()
            optimizer_model.step()
            
        # Validate after the end of epochs
        if (epoch+1) % validation_interval == 0 or epoch == num_epochs-1:
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                total_val_sample = 0
                for j, (valid_batch_x, valid_batch_y) in enumerate(sample_y_valid):
                    valid_batch_x, valid_batch_y = valid_batch_x.to(device), valid_batch_y.to(device)
                    FOM_validate = model(valid_batch_x)
                    loss_validate = criterion(FOM_validate, valid_batch_y)
                    total_val_loss += loss_validate.item() * valid_batch_x.shape[0]
                    total_val_sample += valid_batch_x.shape[0]
                val_loss = total_val_loss / total_val_sample
                
            # Save the model if loss of validation is smaller than minimum
            if val_loss < min_loss_val:
                min_loss_val = val_loss
                early_stop_count = 0
                best_model = model.state_dict()
                #torch.save(best_model, filename+'_k'+str(0)+"_model.pt")
            else:
                if(val_loss < early_stop_loss):
                    # early_stop_loss is the absolute criterion of early stop
                    # to get a model which have a sufficiently small loss.
                    early_stop_count += 1
                
                if early_stop_count >= early_stop:
                    #print("End of epochs")
                    #print(f'epoch {epoch+1} train loss: {loss} valid loss {min_loss_val}')
                    break
    end_train_time = time.time()
    train_time_data.append(end_train_time - start_train_time)
    #end_train_time
    #add_list_data(file_list[0],train_time_data)
    #print("epoch : "+str(min_epoch))
    #print("validation_loss : "+str(min_loss_val))
    '''# After the end of the last epoch, save the model.
    if best_model is None:  
        torch.save(model.state_dict(), filename+'_k'+str(0)+"_model.pt")
        pass
    model.load_state_dict(torch.load(filename+'_k'+str(0)+"_model.pt",map_location=device))'''
    # save FM hyperparameters
    #print('save coefficients')
    bias = model.lin.bias.detach().cpu().numpy()
    linear = model.lin.weight.detach().cpu().numpy()
    linear = linear.reshape(-1,1)
    quadratic = model.K_second.detach().cpu().numpy()
    third_interaction = model.K_third.detach().cpu().numpy()
    
    bias = np.array(bias)
    linear = np.array(linear)
    quadratic = np.array(quadratic) 
    Q_layer=quadratic@np.transpose(quadratic)
    
    QUBO=np.zeros((linear.shape[0],linear.shape[0])) ## 32 x 32
    #print(linear.shape[0])
    
    QUBO[0:Q_layer.shape[0],0:Q_layer.shape[0]]=Q_layer

    for i in range(linear.shape[0]):
        for j in range(linear.shape[0]):
            if i>=j:
                QUBO[i,j]=0

    for i in range(linear.shape[0]):
        QUBO[i,i]=linear[i][0]
        
    #np.savetxt('bias' + '.txt', bias, fmt='%f')
    #np.savetxt('QUBO' + '.txt', QUBO, fmt='%f')
    #np.savetxt('k_third' + '.txt', third_interaction, fmt='%f')
    #np.savetxt('linear.txt', linear, fmt='%f')
    #np.savetxt('quadratic.txt', quadratic, fmt='%f')
    
    return QUBO, third_interaction

def mfi(name_file):
    text_file_buffer=open(name_file,'r')
    content_buffer=text_file_buffer.read()
    np_buffer=StringIO(content_buffer)
    data_array=np.loadtxt(np_buffer)
    return data_array

def save_batch_to_txt(filename, layer, subfolder_name, sub_subfolder_name, batch):
    # Create an empty list
    results = []

    for batch_x, batch_y in batch:
        # Convert input data to numpy array and cast data type to int
        batch_x = batch_x.numpy().astype(int)
        # Convert label data to numpy array and reshape into a column vector
        batch_y = batch_y.numpy().reshape(-1, 1)
        # Append to the results list
        results.append(np.concatenate((batch_y, batch_x), axis=1))

    # Create the overall result array
    result_np = np.concatenate(results, axis=0)

    #np.savetxt(f"tmp{filename}.txt", result_np, delimiter='\t')
    
    # Load the saved file
    #ar_r = np.loadtxt("tmp"+filename+'.txt')

    first_col_cv = ar_r[:, 0].reshape(-1, 1)
    rest_cols_cv = ar_r[:, 1:]

    # Concatenate first column and the rest columns
    result_cv = np.concatenate((first_col_cv, rest_cols_cv), axis=1)
    
    current_directory = os.getcwd()
    save_path = os.path.join(current_directory, subfolder_name, sub_subfolder_name) # Create subfolders if they don't exist
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)
    
    file_path = os.path.join(save_path, f"{filename}.txt")

    # Create format list
    fmt = ['%f'] + ['%d']*layer
    with open(file_path, 'a') as file:
        for line in result_cv:
            file.write(f"{line}\n")

    
# Load saved bias from a txt file.
def get_bias_from_txt(filename):
    file_path = filename + '.txt'
    try:
        with open(file_path, 'r') as file:
            # 파일 내용을 읽어와 변수에 저장합니다.
            bias = float(file.read())

    except FileNotFoundError:
        print(f"Can not find {file_path}")
    except Exception as e:
        print(f"Error occurred while reading the file: {e}")
    return bias

# Load saved QUBO from a txt file.
def get_matrix_from_txt(filename):
    file_path = filename + '.txt'
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        values = line.strip().split()  
        row = [float(value) for value in values]  
        data.append(row)

    QUBO = np.array(data)
    return QUBO

# Convert index into bits
def int_to_padded_float32_binary_array(num, num_bits):
    if num < 0:
        raise ValueError("We cannot process negative numbers.")
    
    binary_string = bin(num)[2:]  
    
    binary_string = binary_string.zfill(num_bits)
    
    binary_array = np.array(list(binary_string), dtype=np.float32)
    
    return binary_array


# calculate third interaction term
def cal_3_interaction(x, third_interactions):
    # x = binary bit. np.array
    # Third-order terms "Lemma3, Polynomial Networks and Factorization Machines" 
    #homogenius polynomial kernel H3(K_second, x)
    # Convert input x and third_interaction into tensors using PyTorch.
    x_2_dimension = x.reshape(1, -1)
    tensor_x = torch.from_numpy(x_2_dimension)
    tensor_k = torch.from_numpy(third_interactions)
    H_3_px = torch.matmul(tensor_x, tensor_k).pow(3).sum(1, keepdim=True) 
    #D3(K_third, x)
    D_3_px = torch.matmul(tensor_x.pow(3), tensor_k.pow(3)).sum(1, keepdim=True)        
    #D2,1(K_third, x)
    D_21_px_D2 = torch.matmul(tensor_x.pow(2), tensor_k.pow(2))
    D_21_px_D1 = torch.matmul(tensor_x.pow(1), tensor_k.pow(1))
    D_21_px=torch.mul(D_21_px_D2,D_21_px_D1).sum(1, keepdim=True) 
    # (1/6)*(H3-3D21+2D3)
    Anova_3rd = (1/6)*(H_3_px - 3*D_21_px + 2*D_3_px) 
    return Anova_3rd
  

# Function to determine if the obtained position exists in the dataset, and if so, to retrieve another random position.
def get_random_position(dataset,position):
    if any(np.all(dataset[i] == position) for i in range(dataset.shape[0])):
        #If the position already exists in the dataset, receive a random value
        random_value = random.randint(0,2**(dataset.shape[1])-1)
        position = int_to_padded_float32_binary_array(random_value, dataset.shape[1])
        return get_random_position(dataset,position)
    else:
        return position

# Add new data to a txt file.
def add_data(filename, value):
    if type(value) != str:
        value_str = str(value)
    else:
        value_str = value
    if os.path.isfile(filename):
        with open(filename +'.txt', 'r') as file:
            existing_data = file.readlines() 
        with open(filename +'.txt', 'w') as file:
            for line in existing_data:
                file.write(line)
                file.write(f"{value_str}\n")
    else:
        with open(filename +'.txt', 'a') as file:
            file.write(value_str + "\n")
            
# Add new FOM to a txt file.    
def add_FOM_data(filename, value_data, position_data):
    if type(value_data) != str:
        value_str = str(value_data)
        position_str = ' '.join(map(str,position_data))
    else:
        value_str = value_data
        position_str = ' '.join(map(str,position_data))
    
    if os.path.isfile(filename):
        with open(filename +'.txt', 'r') as file:
            existing_data = file.readlines()
        with open(filename +'.txt', 'w') as file:
            for line in existing_data:
                file.write(line)
            file.write(f"{value_str} {position_str}\n")
    else:
        with open(filename +'.txt', 'a') as file:
            file.write(f"{value_str} {position_str}\n")

def save_array_txt_int(file_name, subfolder_name, sub_subfolder_name, array):
    # Set up subfolder paths
    subfolder_path = os.path.join(os.getcwd(), subfolder_name)
    sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder_name)

    # Create subfolders if they don't exist
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Save data to a temporary file first
    if not os.path.exists(sub_subfolder_path):
        os.makedirs(sub_subfolder_path)

    # Open the file, remove decimals, and write to a new file
    temp_file = 'temp_data.txt'
    #np.savetxt(temp_file, array, delimiter=' ')

    # 파일 열어서 소수점 제거 후 새 파일에 쓰기
    with open(temp_file, 'r') as file:
        lines = file.readlines()

    # Remove decimals and write to the new file
    with open(os.path.join(sub_subfolder_path, file_name + '.txt'), 'w') as new_file:
        for line in lines:
            data = line.strip().split()  # Split each line by space
            integer_data = [str(int(float(value))) for value in data]  # Convert each value to float, then to integer, and store as a string
            new_line = ' '.join(integer_data)  # Join each value with space to create a new line
            new_file.write(new_line + '\n')  # Write the new line to the file

    # 임시 파일 삭제
    os.remove(temp_file)
            
def save_matrix_to_subfolder(subfolder_name, subsubfolder_name, filename):
    current_directory = os.getcwd()
    from_file_path = os.path.join(current_directory,filename)
    to_file_path = os.path.join(current_directory, subfolder_name, subsubfolder_name, filename)
    
    shutil.copy(from_file_path, to_file_path)

def save_data_to_subfolder(data, subfolder_name, subsubfolder_name, filename):
    current_directory = os.getcwd()
    subfolder_path = os.path.join(current_directory, subfolder_name, subsubfolder_name)

    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    file_path = os.path.join(subfolder_path, f"{filename}.txt") 

    with open(file_path, 'a') as file:
        file.write(str(data)) 
        
def rms_loss(y,y_star):
    diffy=y-y_star
    rms=np.sqrt(np.mean(diffy**2))
    return rms

# get predicted FOM from model
def model_output(sample_x, model_class, model_params, device, filename):
    
    #Convert to Torch
    if torch.cuda.is_available():   
        sample_x_ts=torch.cuda.FloatTensor(sample_x).to(device)
        
    else: 
        sample_x_ts=torch.FloatTensor(sample_x).to(device)
    y_predict_container=torch.FloatTensor(torch.zeros([sample_x_ts.shape[0],1])).to(device)
    
    
    model = model_class(**model_params).to(device)
    model.load_state_dict(torch.load(filename+'_k'+str(0)+"_model.pt",map_location=device))
    
    model.eval()
    y_predict_container[:,0]=model(sample_x_ts).reshape(1,-1)
    
    del model
    
    y_predict_avg=y_predict_container.mean(dim=1)
    return y_predict_avg


def save_fom_position_batch(data_values, binary_values, subfolder_name, subsubfolder_name, file_name):
    current_directory = os.getcwd()
    subfolder_path = os.path.join(current_directory, subfolder_name, subsubfolder_name)

    file_path = os.path.join(subfolder_path, f"{file_name}.txt")  # 파일 이름 설정
    
    with open(file_path, 'w') as file:
        for i in range(len(data_values)):
            line = f"{data_values[i]} {binary_values[i]}"
            file.write(f"{line}\n")

# QUBO_3rd
def cal_3_interaction(x, third_interactions):
    # x = binary bit. np.array
    # Third-order terms "Lemma3, Polynomial Networks and Factorization Machines" 
    #homogenius polynomial kernel H3(K_second, x)
    # input x와 third_interaction을 torch의 tensor로 바꿔주기.
    x_2_dimension = x.reshape(1, -1)
    f_array = third_interactions.astype(np.float64)
    tensor_x = torch.from_numpy(x_2_dimension)
    tensor_k = torch.from_numpy(f_array)
    H_3_px = torch.matmul(tensor_x, tensor_k).pow(3).sum(1, keepdim=True) 
    #D3(K_third, x)
    D_3_px = torch.matmul(tensor_x.pow(3), tensor_k.pow(3)).sum(1, keepdim=True)        
    #D2,1(K_third, x)
    D_21_px_D2 = torch.matmul(tensor_x.pow(2), tensor_k.pow(2))
    D_21_px_D1 = torch.matmul(tensor_x.pow(1), tensor_k.pow(1))
    D_21_px=torch.mul(D_21_px_D2,D_21_px_D1).sum(1, keepdim=True) 
    # (1/6)*(H3-3D21+2D3)
    Anova_3rd = (1/6)*(H_3_px - 3*D_21_px + 2*D_3_px) 
    return Anova_3rd
# Load saved bias from a txt file.
def get_bias_from_txt(filename):
    file_path = filename + '.txt'
    try:
        with open(file_path, 'r') as file:
            # 파일 내용을 읽어와 변수에 저장합니다.
            bias = float(file.read())

    except FileNotFoundError:
        print(f"Can not find {file_path}")
    except Exception as e:
        print(f"Error occurred while reading the file: {e}")
    return bias

# Load saved QUBO from a txt file.
def get_matrix_from_txt(filename):
    file_path = filename + '.txt'
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        values = line.strip().split()  
        row = [float(value) for value in values]  
        data.append(row)

    QUBO = np.array(data)
    return QUBO
# Convert index into bits
def int_to_padded_float64_binary_array(num, num_bits):
    if num < 0:
        raise ValueError("We cannot process negative numbers.")
    
    binary_string = bin(num)[2:]  
    
    binary_string = binary_string.zfill(num_bits)
    
    binary_array = np.array(list(binary_string), dtype=np.float64)
    
    return binary_array
### define function for making keys for 3rd order QUBO matrix
def generate_combinations(n):
    combinations = []
    for i in range(1, n):
        for j in range(i + 1, n):
            combinations.append((i, j))
    return combinations

def save_combinations_to_txt(combinations, filename):
    with open(filename, 'w') as file:
        for combination in combinations:
            file.write(f"{combination[0]} {combination[1]}\n")

### define function for reading keys from txt file
def read_combinations_from_txt(filename):
    combinations = []
    with open(filename, 'r') as file:
        for line in file:
            pair = line.strip().split()
            combinations.append([int(pair[0]), int(pair[1])])
    return combinations

def bit_extension_3rd(position,combinations):
    temp = np.zeros(position.shape[0] + combinations.shape[0], dtype=np.float64)
    temp[:position.shape[0]] = position
    for i in range(combinations.shape[0]):
        ind1 = combinations[i][0] - 1
        ind2 = combinations[i][1] - 1
        temp[position.shape[0]+i] = position[ind1] * position[ind2]
        
    return temp    

def FM_3rd(train):
    #######
    #main
    num_bits = train.shape[1]-1
    initial_data = train.shape[0]
    n = num_bits
    
    #------------------------ Hyper parameter -----------------------------#
    iterations = 1
    Model_class = FM_order_3
    Model_params={'n':n, 'k2' : 8, 'k3': 8}
    Num_epochs = 15000    
    Early_stop = 7500
    Early_stop_loss = 2.5
    Criterion = nn.MSELoss()
    Optimizer_class = torch.optim.Adam
    Opt_params={'lr': 0.1, 'weight_decay' : 0}
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    option1 = 2 # The number of substance. In this problems, SiO2 and TiO2
    option2 = n # Exponent. If option2 == n (bit), only 1 junk for brute force. If n-1, two junk.
    
    sampling_size = 200 # If the dataset exceeds 200 entries, increment the data gradually.
    #if you don't want to sample the dataset, use next line instead of above line
    #sampling_size = iteration + X_input_.shape[0] + 1
    slope = 2 # The gradient of increaseing data. 1/2, 1/4 ...
    
    ###
    
    batch_size_train = 2**8  
    batch_size_validation = 2**6  # Because validation set is 1/4 of Train set 

    All_X_input = train[:,1:]
    
    #sampling
    NOS = 0
    if train.shape[0] > sampling_size:
        NOS = sampling_size + (train.shape[0] - sampling_size)//slope
    else:
        NOS = train.shape[0]
    random_indices = np.random.choice(train.shape[0], size=NOS, replace=False)
    sampled_dataset = train[random_indices]
    
    X_input = sampled_dataset[:,1:] ## Binary bit.
    train_target = sampled_dataset[:,0] ## fom
    
    train_indices, test_indices = split_train_test(X_input, 0.2) ## test set이 0.2, train set이 0.8
    
    X_train_num = int(train_indices.shape[0]) 
    X_test_num = int(test_indices.shape[0]) 
    
    X_train = X_input[train_indices] # Convert to bits at the specified index.
    X_test = X_input[test_indices]
    
    target = train_target[train_indices] # FOM of train dataset
    real_y = train_target[test_indices] # FOM of validation dataset
    
    y = target.astype(np.float32)
    y = y.reshape(-1,1)
    
    val_y = real_y.astype(np.float32)
    val_y = val_y.reshape(-1,1)
    
    n = X_train.shape[1] # number of total binary vector
    
    sample_binary_train_ts = torch.from_numpy(X_train).float()
    sample_fom_train_ts = torch.from_numpy(y) # FOM of train dataset
    sample_binary_test_ts = torch.from_numpy(X_test).float()
    sample_fom_test_ts = torch.from_numpy(val_y) # FOM of validation dataset
    
    train_dataset = torch.utils.data.TensorDataset(sample_binary_train_ts, sample_fom_train_ts)
    val_dataset = torch.utils.data.TensorDataset(sample_binary_test_ts, sample_fom_test_ts)
    
    if len(train_dataset)<= batch_size_train :
        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=False)     # modify: shuffle=True
        val_loader = DataLoader(val_dataset, batch_size=batch_size_validation, shuffle=False, drop_last=False)
    else :
        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=True)
        
        if len(val_dataset) <=  batch_size_validation:
            val_loader = DataLoader(val_dataset, batch_size=batch_size_validation, shuffle=False, drop_last=False)
        else:    
            val_loader = DataLoader(val_dataset, batch_size=batch_size_validation, shuffle=False, drop_last=True)
    
    #save_batch_to_txt(train_set_data, n, 'dataset', str(i), train_loader) #train_filename.txt file
    #save_batch_to_txt(valid_set_data, n, 'dataset', str(i), val_loader) #cv_filename.txt file
      
    QUBO, k_third = train_valid_order3(sample_train = train_loader,
                                       sample_valid = val_loader,
                                       filename = None,
                                       iteration = None,
                                       model_class = Model_class, 
                                       model_params=Model_params,
                                       num_epochs = Num_epochs,
                                       early_stop = Early_stop,
                                       early_stop_loss = Early_stop_loss,
                                       criterion = Criterion,
                                       optimizer_class=Optimizer_class,
                                       opt_params=Opt_params,
                                       device=DEVICE,
                                       file_list = None)


    return QUBO, k_third

