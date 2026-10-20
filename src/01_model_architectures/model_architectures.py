import torch
import torch.nn as nn
import torch.nn.functional as F

##################################
#Collection of model architectures
##################################

def outputLen_MaxPool(input_len, kernel_size, padding, stride, dilation):
    """
    Helper function for the use in the DeepSTARR and MultiNet architectures.
    """
    out = torch.floor(((input_len + (2*padding) - (dilation*(kernel_size - 1)) -1)/float(stride)) + 1)
    return(out)

class MultiNetToy(torch.nn.Module):
    """
    A toy architecture to try out models with multiple input modalities.
    """
    def __init__(self):

        super(MultiNetToy, self).__init__()

        self.conv1 = nn.Conv1d(4, 256, kernel_size=7, stride=1, padding = "same")
        self.conv1_bn = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.layers_cnn = nn.ModuleList()
        self.layers_fc = nn.ModuleList()

        self.layers_cnn.append(nn.Conv1d(256,120, kernel_size=3, stride=1, padding = "same"))
        self.layers_cnn.append(nn.BatchNorm1d(120))
        self.layers_cnn.append(nn.ReLU())
        self.layers_cnn.append(nn.MaxPool1d(kernel_size=2))

        self.layers_cnn.append(nn.Conv1d(120, 60, kernel_size=3, stride=1, padding = "same"))
        self.layers_cnn.append(nn.BatchNorm1d(60))
        self.layers_cnn.append(nn.ReLU())
        self.layers_cnn.append(nn.MaxPool1d(kernel_size=2))

        self.flatten = nn.Flatten()

        self.layers_fc.append(nn.Linear(60*125,64))
        self.layers_fc.append(nn.BatchNorm1d(64))
        self.layers_fc.append(nn.ReLU())
        self.layers_fc.append(nn.Linear(64,30))
        self.layers_fc.append(nn.BatchNorm1d(30))
        self.layers_fc.append(nn.ReLU())

        self.dense_output = nn.Linear(30,1)


    def forward(self,seq, gex):
        
        x = self.conv1(seq.float())

        x = x*gex.unsqueeze(2)

        x = self.conv1_bn(x)
        x = self.relu1(x)
        x = self.pool1(x)

        for layer in self.layers_cnn:
            x = layer(x)
        
        x = self.flatten(x)

        for layer in self.layers_fc:
            x = layer(x)
        
        return self.dense_output(x)

class MultiNet_v1(torch.nn.Module):
    """
    The first version of the MultiNet architecture.
    Scales response maps of first layer kernels with TF expression values, otherwise identical with DeepSTARR architecture.
    Used to produce the results of the 2025-03-17 moday meeting report.
    """
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, input_length, input_channel, output_channel1, output_channel2, output_channel3, output_channel4,
                 kernel_size1, kernel_size2, kernel_size3, kernel_size4, stride, kernel_pool,
                 padding, nlayers_cnn, nlayers_fc, output_fc1, output_fc2, prob):
        super(MultiNet_v1, self).__init__()
        self.input_length = torch.tensor([input_length])
        self.input_channel = 4
        self.output_channel1 = output_channel1
        self.output_channel2 = output_channel2
        self.output_channel3 = output_channel3
        self.output_channel4 = output_channel4
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4
        self.nlayers_cnn = nlayers_cnn
        self.nlayers_fc = nlayers_fc
        self.stride = 1
        self.padding = 'same'
        self.kernel_pool = 2 # also default stride
        self.prob = prob
                
        # Conv layer 1
        self.conv1 = torch.nn.Conv1d(self.input_channel, self.output_channel1, 
            kernel_size=self.kernel_size1, stride=self.stride, padding=self.padding)

        x_len_out = self.input_length
            
        # Batch Normalization 1
        self.conv1_bn = nn.BatchNorm1d(self.output_channel1)
        self.relu1 = nn.ReLU()
        
        # Max pooling 1
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
        x_len_out = outputLen_MaxPool(x_len_out, self.pool.kernel_size, self.pool.padding, self.pool.stride, self.pool.dilation)
        
        current_dim = self.output_channel1
        self.layers_cnn = nn.ModuleList()
        self.layers_fc = nn.ModuleList()
        hidden_dim = [output_channel1, output_channel2, output_channel3, output_channel4]
        kernel_sizes = [kernel_size1, kernel_size2, kernel_size3, kernel_size4]
        outputs_fc = [output_fc1, output_fc2]
        for i in range(1, nlayers_cnn):
            self.layers_cnn.append(torch.nn.Conv1d(current_dim, hidden_dim[i], 
                kernel_size=kernel_sizes[i], stride=self.stride, padding=self.padding))

            self.layers_cnn.append(nn.BatchNorm1d(hidden_dim[i]))
            self.layers_cnn.append(nn.ReLU())
            self.layers_cnn.append(nn.MaxPool1d(kernel_size=self.kernel_pool))
            
            x_len_out = outputLen_MaxPool(x_len_out, self.layers_cnn[-1].kernel_size, self.layers_cnn[-1].padding, self.layers_cnn[-1].stride, self.layers_cnn[-1].dilation)
            current_dim = hidden_dim[i]
            
        self.flatten = nn.Flatten()
        
        input_final_fc = current_dim*int(x_len_out.item())

        for i in range(0, nlayers_fc):
            self.layers_fc.append(nn.Linear(input_final_fc, outputs_fc[i]))
            self.layers_fc.append(nn.BatchNorm1d(outputs_fc[i]))
            self.layers_fc.append(nn.ReLU())
            self.layers_fc.append(nn.Dropout(self.prob))
            input_final_fc = outputs_fc[i]
        
        self.dense_output = nn.Linear(input_final_fc, 1)

        
    def forward(self, seq, gex):

        gex=gex[:,:self.output_channel1] #TO DO: implement selection of best TFs
        x = self.conv1(seq.float())
        x = x*gex.unsqueeze(2)
        #return x
        x = self.relu1(self.conv1_bn(x))
        x = self.pool(x)

        for layer in self.layers_cnn:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.layers_fc:
            x = layer(x)
        return self.dense_output(x)
    
class MultiNet_v2(torch.nn.Module):
    """
    MultiNet architecture.
    The version used for training the pbmc and blastoid dynamic models.
    Scales response maps of first layer kernels with TF expression values, otherwise identical with DeepSTARR architecture.
    Response maps are scaled by r --> (r*g+r)/2
    """
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, input_length, input_channel, output_channel1, output_channel2, output_channel3, output_channel4,
                 kernel_size1, kernel_size2, kernel_size3, kernel_size4, stride, kernel_pool,
                 padding, nlayers_cnn, nlayers_fc, output_fc1, output_fc2, prob):
        super(MultiNet_v2, self).__init__()
        self.input_length = torch.tensor([input_length])
        self.input_channel = 4
        self.output_channel1 = output_channel1
        self.output_channel2 = output_channel2
        self.output_channel3 = output_channel3
        self.output_channel4 = output_channel4
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4
        self.nlayers_cnn = nlayers_cnn
        self.nlayers_fc = nlayers_fc
        self.stride = 1
        self.padding = 'same'
        self.kernel_pool = 2 # also default stride
        self.prob = prob
                
        # Conv layer 1
        self.conv1 = torch.nn.Conv1d(self.input_channel, self.output_channel1, 
            kernel_size=self.kernel_size1, stride=self.stride, padding=self.padding)

        x_len_out = self.input_length
            
        # Batch Normalization 1
        self.conv1_bn = nn.BatchNorm1d(self.output_channel1)
        self.relu1 = nn.ReLU()
        
        # Max pooling 1
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
        x_len_out = outputLen_MaxPool(x_len_out, self.pool.kernel_size, self.pool.padding, self.pool.stride, self.pool.dilation)
        
        current_dim = self.output_channel1
        self.layers_cnn = nn.ModuleList()
        self.layers_fc = nn.ModuleList()
        hidden_dim = [output_channel1, output_channel2, output_channel3, output_channel4]
        kernel_sizes = [kernel_size1, kernel_size2, kernel_size3, kernel_size4]
        outputs_fc = [output_fc1, output_fc2]
        for i in range(1, nlayers_cnn):
            self.layers_cnn.append(torch.nn.Conv1d(current_dim, hidden_dim[i], 
                kernel_size=kernel_sizes[i], stride=self.stride, padding=self.padding))

            self.layers_cnn.append(nn.BatchNorm1d(hidden_dim[i]))
            self.layers_cnn.append(nn.ReLU())
            self.layers_cnn.append(nn.MaxPool1d(kernel_size=self.kernel_pool))
            
            x_len_out = outputLen_MaxPool(x_len_out, self.layers_cnn[-1].kernel_size, self.layers_cnn[-1].padding, self.layers_cnn[-1].stride, self.layers_cnn[-1].dilation)
            current_dim = hidden_dim[i]
            
        self.flatten = nn.Flatten()
        
        input_final_fc = current_dim*int(x_len_out.item())

        for i in range(0, nlayers_fc):
            self.layers_fc.append(nn.Linear(input_final_fc, outputs_fc[i]))
            self.layers_fc.append(nn.BatchNorm1d(outputs_fc[i]))
            self.layers_fc.append(nn.ReLU())
            self.layers_fc.append(nn.Dropout(self.prob))
            input_final_fc = outputs_fc[i]
        
        self.dense_output = nn.Linear(input_final_fc, 1)

        
    def forward(self, seq, gex):
        x = self.conv1(seq.float())
        x = x*gex.unsqueeze(2)
        x = self.conv1_bn(x)
        x = self.relu1(x)
        x = self.pool(x)

        for layer in self.layers_cnn:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.layers_fc:
            x = layer(x)

        return self.dense_output(x)

class MultiNet_v3(torch.nn.Module):
    """
    MultiNet architecture.
    Version where the GEX scaling is moved after the batch norm.
    Scales response maps of first layer kernels with TF expression values, otherwise identical with DeepSTARR architecture.
    Response maps are scaled by r --> (r*g+r)/2
    """
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, input_length, input_channel, output_channel1, output_channel2, output_channel3, output_channel4,
                 kernel_size1, kernel_size2, kernel_size3, kernel_size4, stride, kernel_pool,
                 padding, nlayers_cnn, nlayers_fc, output_fc1, output_fc2, prob):
        super(MultiNet_v3, self).__init__()
        self.input_length = torch.tensor([input_length])
        self.input_channel = 4
        self.output_channel1 = output_channel1
        self.output_channel2 = output_channel2
        self.output_channel3 = output_channel3
        self.output_channel4 = output_channel4
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4
        self.nlayers_cnn = nlayers_cnn
        self.nlayers_fc = nlayers_fc
        self.stride = 1
        self.padding = 'same'
        self.kernel_pool = 2 # also default stride
        self.prob = prob
                
        # Conv layer 1
        self.conv1 = torch.nn.Conv1d(self.input_channel, self.output_channel1, 
            kernel_size=self.kernel_size1, stride=self.stride, padding=self.padding)

        x_len_out = self.input_length
            
        # Batch Normalization 1
        self.conv1_bn = nn.BatchNorm1d(self.output_channel1)
        self.relu1 = nn.ReLU()
        
        # Max pooling 1
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
        x_len_out = outputLen_MaxPool(x_len_out, self.pool.kernel_size, self.pool.padding, self.pool.stride, self.pool.dilation)
        
        current_dim = self.output_channel1
        self.layers_cnn = nn.ModuleList()
        self.layers_fc = nn.ModuleList()
        hidden_dim = [output_channel1, output_channel2, output_channel3, output_channel4]
        kernel_sizes = [kernel_size1, kernel_size2, kernel_size3, kernel_size4]
        outputs_fc = [output_fc1, output_fc2]
        for i in range(1, nlayers_cnn):
            self.layers_cnn.append(torch.nn.Conv1d(current_dim, hidden_dim[i], 
                kernel_size=kernel_sizes[i], stride=self.stride, padding=self.padding))

            self.layers_cnn.append(nn.BatchNorm1d(hidden_dim[i]))
            self.layers_cnn.append(nn.ReLU())
            self.layers_cnn.append(nn.MaxPool1d(kernel_size=self.kernel_pool))
            
            x_len_out = outputLen_MaxPool(x_len_out, self.layers_cnn[-1].kernel_size, self.layers_cnn[-1].padding, self.layers_cnn[-1].stride, self.layers_cnn[-1].dilation)
            current_dim = hidden_dim[i]
            
        self.flatten = nn.Flatten()
        
        input_final_fc = current_dim*int(x_len_out.item())

        for i in range(0, nlayers_fc):
            self.layers_fc.append(nn.Linear(input_final_fc, outputs_fc[i]))
            self.layers_fc.append(nn.BatchNorm1d(outputs_fc[i]))
            self.layers_fc.append(nn.ReLU())
            self.layers_fc.append(nn.Dropout(self.prob))
            input_final_fc = outputs_fc[i]
        
        self.dense_output = nn.Linear(input_final_fc, 1)

        
    def forward(self, seq, gex):
        x = self.conv1(seq.float())
        x = self.conv1_bn(x)
        x = self.relu1(x)
        x = x*gex.unsqueeze(2)
        x = self.pool(x)

        for layer in self.layers_cnn:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.layers_fc:
            x = layer(x)

        return self.dense_output(x)

class MultiNet_v3_01(torch.nn.Module):
    """
    MultiNet architecture.
    Version where the GEX scaling is moved after the batch norm.
    Same as MultiNet_v3 but with additional output of the scaled resopnse maps for training with sparcity regularisation.
    Scales response maps of first layer kernels with TF expression values, otherwise identical with DeepSTARR architecture.
    Response maps are scaled by r --> (r*g+r)/2
    """
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, input_length, input_channel, output_channel1, output_channel2, output_channel3, output_channel4,
                 kernel_size1, kernel_size2, kernel_size3, kernel_size4, stride, kernel_pool,
                 padding, nlayers_cnn, nlayers_fc, output_fc1, output_fc2, prob):
        super(MultiNet_v3_01, self).__init__()
        self.input_length = torch.tensor([input_length])
        self.input_channel = 4
        self.output_channel1 = output_channel1
        self.output_channel2 = output_channel2
        self.output_channel3 = output_channel3
        self.output_channel4 = output_channel4
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4
        self.nlayers_cnn = nlayers_cnn
        self.nlayers_fc = nlayers_fc
        self.stride = 1
        self.padding = 'same'
        self.kernel_pool = 2 # also default stride
        self.prob = prob
                
        # Conv layer 1
        self.conv1 = torch.nn.Conv1d(self.input_channel, self.output_channel1, 
            kernel_size=self.kernel_size1, stride=self.stride, padding=self.padding)

        x_len_out = self.input_length
            
        # Batch Normalization 1
        self.conv1_bn = nn.BatchNorm1d(self.output_channel1)
        self.relu1 = nn.ReLU()
        
        # Max pooling 1
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
        x_len_out = outputLen_MaxPool(x_len_out, self.pool.kernel_size, self.pool.padding, self.pool.stride, self.pool.dilation)
        
        current_dim = self.output_channel1
        self.layers_cnn = nn.ModuleList()
        self.layers_fc = nn.ModuleList()
        hidden_dim = [output_channel1, output_channel2, output_channel3, output_channel4]
        kernel_sizes = [kernel_size1, kernel_size2, kernel_size3, kernel_size4]
        outputs_fc = [output_fc1, output_fc2]
        for i in range(1, nlayers_cnn):
            self.layers_cnn.append(torch.nn.Conv1d(current_dim, hidden_dim[i], 
                kernel_size=kernel_sizes[i], stride=self.stride, padding=self.padding))

            self.layers_cnn.append(nn.BatchNorm1d(hidden_dim[i]))
            self.layers_cnn.append(nn.ReLU())
            self.layers_cnn.append(nn.MaxPool1d(kernel_size=self.kernel_pool))
            
            x_len_out = outputLen_MaxPool(x_len_out, self.layers_cnn[-1].kernel_size, self.layers_cnn[-1].padding, self.layers_cnn[-1].stride, self.layers_cnn[-1].dilation)
            current_dim = hidden_dim[i]
            
        self.flatten = nn.Flatten()
        
        input_final_fc = current_dim*int(x_len_out.item())

        for i in range(0, nlayers_fc):
            self.layers_fc.append(nn.Linear(input_final_fc, outputs_fc[i]))
            self.layers_fc.append(nn.BatchNorm1d(outputs_fc[i]))
            self.layers_fc.append(nn.ReLU())
            self.layers_fc.append(nn.Dropout(self.prob))
            input_final_fc = outputs_fc[i]
        
        self.dense_output = nn.Linear(input_final_fc, 1)

        
    def forward(self, seq, gex):
        x = self.conv1(seq.float())
        x = self.conv1_bn(x)
        x = self.relu1(x)
        x = x*gex.unsqueeze(2)
        conv1_response_maps_scaled = x.clone()
        x = self.pool(x)

        for layer in self.layers_cnn:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.layers_fc:
            x = layer(x)

        return self.dense_output(x), conv1_response_maps_scaled

class MultiNet_v3_02(torch.nn.Module):
    """
    MultiNet architecture.
    Version where the GEX scaling is moved after the batch norm.
    Same as MultiNet_v3 but with additional output of the scaled resopnse maps for training with sparcity regularisation and it has a trainable mask to cancel out "noncausal" kernels (hopefully)
    Scales response maps of first layer kernels with TF expression values, otherwise identical with DeepSTARR architecture.
    Response maps are scaled by r --> (r*g+r)/2
    """
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, input_length, input_channel, output_channel1, output_channel2, output_channel3, output_channel4,
                 kernel_size1, kernel_size2, kernel_size3, kernel_size4, stride, kernel_pool,
                 padding, nlayers_cnn, nlayers_fc, output_fc1, output_fc2, prob):
        super(MultiNet_v3_02, self).__init__()
        self.input_length = torch.tensor([input_length])
        self.input_channel = 4
        self.output_channel1 = output_channel1
        self.output_channel2 = output_channel2
        self.output_channel3 = output_channel3
        self.output_channel4 = output_channel4
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4
        self.nlayers_cnn = nlayers_cnn
        self.nlayers_fc = nlayers_fc
        self.stride = 1
        self.padding = 'same'
        self.kernel_pool = 2 # also default stride
        self.prob = prob

        self.conv1_trainable_mask = nn.Parameter(
            torch.ones(1,output_channel1,1)
        )

                
        # Conv layer 1
        self.conv1 = torch.nn.Conv1d(self.input_channel, self.output_channel1, 
            kernel_size=self.kernel_size1, stride=self.stride, padding=self.padding)

        x_len_out = self.input_length
            
        # Batch Normalization 1
        self.conv1_bn = nn.BatchNorm1d(self.output_channel1)
        self.relu1 = nn.ReLU()
        
        # Max pooling 1
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
        x_len_out = outputLen_MaxPool(x_len_out, self.pool.kernel_size, self.pool.padding, self.pool.stride, self.pool.dilation)
        
        current_dim = self.output_channel1
        self.layers_cnn = nn.ModuleList()
        self.layers_fc = nn.ModuleList()
        hidden_dim = [output_channel1, output_channel2, output_channel3, output_channel4]
        kernel_sizes = [kernel_size1, kernel_size2, kernel_size3, kernel_size4]
        outputs_fc = [output_fc1, output_fc2]
        for i in range(1, nlayers_cnn):
            self.layers_cnn.append(torch.nn.Conv1d(current_dim, hidden_dim[i], 
                kernel_size=kernel_sizes[i], stride=self.stride, padding=self.padding))

            self.layers_cnn.append(nn.BatchNorm1d(hidden_dim[i]))
            self.layers_cnn.append(nn.ReLU())
            self.layers_cnn.append(nn.MaxPool1d(kernel_size=self.kernel_pool))
            
            x_len_out = outputLen_MaxPool(x_len_out, self.layers_cnn[-1].kernel_size, self.layers_cnn[-1].padding, self.layers_cnn[-1].stride, self.layers_cnn[-1].dilation)
            current_dim = hidden_dim[i]
            
        self.flatten = nn.Flatten()
        
        input_final_fc = current_dim*int(x_len_out.item())

        for i in range(0, nlayers_fc):
            self.layers_fc.append(nn.Linear(input_final_fc, outputs_fc[i]))
            self.layers_fc.append(nn.BatchNorm1d(outputs_fc[i]))
            self.layers_fc.append(nn.ReLU())
            self.layers_fc.append(nn.Dropout(self.prob))
            input_final_fc = outputs_fc[i]
        
        self.dense_output = nn.Linear(input_final_fc, 1)

        
    def forward(self, seq, gex):
        x = self.conv1(seq.float())
        x = self.conv1_bn(x)
        #apply mask to conv1 outputs
        x = x * self.conv1_trainable_mask
        x = self.relu1(x)
        x = x*gex.unsqueeze(2)
        conv1_response_maps_scaled = x.clone()
        x = self.pool(x)

        for layer in self.layers_cnn:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.layers_fc:
            x = layer(x)

        return self.dense_output(x), conv1_response_maps_scaled


class MultiNet_v4(torch.nn.Module):
    """
    MultiNet architecture.
    All norm layers are removed.
    GEX scaling after the batch norm.
    Scales response maps of first layer kernels with TF expression values, otherwise identical with DeepSTARR architecture.
    Response maps are scaled by r --> (r*g+r)/2
    """
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, input_length, input_channel, output_channel1, output_channel2, output_channel3, output_channel4,
                 kernel_size1, kernel_size2, kernel_size3, kernel_size4, stride, kernel_pool,
                 padding, nlayers_cnn, nlayers_fc, output_fc1, output_fc2, prob):
        super(MultiNet_v4, self).__init__()
        self.input_length = torch.tensor([input_length])
        self.input_channel = 4
        self.output_channel1 = output_channel1
        self.output_channel2 = output_channel2
        self.output_channel3 = output_channel3
        self.output_channel4 = output_channel4
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4
        self.nlayers_cnn = nlayers_cnn
        self.nlayers_fc = nlayers_fc
        self.stride = 1
        self.padding = 'same'
        self.kernel_pool = 2 # also default stride
        self.prob = prob
                
        # Conv layer 1
        self.conv1 = torch.nn.Conv1d(self.input_channel, self.output_channel1, 
            kernel_size=self.kernel_size1, stride=self.stride, padding=self.padding)

        x_len_out = self.input_length
            
        # Batch Normalization 1
        #self.conv1_bn = nn.BatchNorm1d(self.output_channel1)
        self.relu1 = nn.ReLU()
        
        # Max pooling 1
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
        x_len_out = outputLen_MaxPool(x_len_out, self.pool.kernel_size, self.pool.padding, self.pool.stride, self.pool.dilation)
        
        current_dim = self.output_channel1
        self.layers_cnn = nn.ModuleList()
        self.layers_fc = nn.ModuleList()
        hidden_dim = [output_channel1, output_channel2, output_channel3, output_channel4]
        kernel_sizes = [kernel_size1, kernel_size2, kernel_size3, kernel_size4]
        outputs_fc = [output_fc1, output_fc2]
        for i in range(1, nlayers_cnn):
            self.layers_cnn.append(torch.nn.Conv1d(current_dim, hidden_dim[i], 
                kernel_size=kernel_sizes[i], stride=self.stride, padding=self.padding))

            #self.layers_cnn.append(nn.BatchNorm1d(hidden_dim[i]))
            self.layers_cnn.append(nn.ReLU())
            self.layers_cnn.append(nn.MaxPool1d(kernel_size=self.kernel_pool))
            
            x_len_out = outputLen_MaxPool(x_len_out, self.layers_cnn[-1].kernel_size, self.layers_cnn[-1].padding, self.layers_cnn[-1].stride, self.layers_cnn[-1].dilation)
            current_dim = hidden_dim[i]
            
        self.flatten = nn.Flatten()
        
        input_final_fc = current_dim*int(x_len_out.item())

        for i in range(0, nlayers_fc):
            self.layers_fc.append(nn.Linear(input_final_fc, outputs_fc[i]))
            #self.layers_fc.append(nn.BatchNorm1d(outputs_fc[i]))
            self.layers_fc.append(nn.ReLU())
            #self.layers_fc.append(nn.Dropout(self.prob))
            input_final_fc = outputs_fc[i]
        
        self.dense_output = nn.Linear(input_final_fc, 1)

        
    def forward(self, seq, gex):
        x = self.conv1(seq.float())
        #x = self.conv1_bn(x)
        x = self.relu1(x)
        x = x*gex.unsqueeze(2)
        x = self.pool(x)

        for layer in self.layers_cnn:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.layers_fc:
            x = layer(x)

        return self.dense_output(x)

class MultiNet_v5(torch.nn.Module):
    """
    MultiNet architecture.
    Norm layers after the GEX scaling are removed.
    GEX scaling after the batch norm.
    Scales response maps of first layer kernels with TF expression values, otherwise identical with DeepSTARR architecture.
    Response maps are scaled by r --> (r*g+r)/2
    """
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, input_length, input_channel, output_channel1, output_channel2, output_channel3, output_channel4,
                 kernel_size1, kernel_size2, kernel_size3, kernel_size4, stride, kernel_pool,
                 padding, nlayers_cnn, nlayers_fc, output_fc1, output_fc2, prob):
        super(MultiNet_v5, self).__init__()
        self.input_length = torch.tensor([input_length])
        self.input_channel = 4
        self.output_channel1 = output_channel1
        self.output_channel2 = output_channel2
        self.output_channel3 = output_channel3
        self.output_channel4 = output_channel4
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4
        self.nlayers_cnn = nlayers_cnn
        self.nlayers_fc = nlayers_fc
        self.stride = 1
        self.padding = 'same'
        self.kernel_pool = 2 # also default stride
        self.prob = prob
                
        # Conv layer 1
        self.conv1 = torch.nn.Conv1d(self.input_channel, self.output_channel1, 
            kernel_size=self.kernel_size1, stride=self.stride, padding=self.padding)

        x_len_out = self.input_length
            
        # Batch Normalization 1
        self.conv1_bn = nn.BatchNorm1d(self.output_channel1)
        self.relu1 = nn.ReLU()
        
        # Max pooling 1
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
        x_len_out = outputLen_MaxPool(x_len_out, self.pool.kernel_size, self.pool.padding, self.pool.stride, self.pool.dilation)
        
        current_dim = self.output_channel1
        self.layers_cnn = nn.ModuleList()
        self.layers_fc = nn.ModuleList()
        hidden_dim = [output_channel1, output_channel2, output_channel3, output_channel4]
        kernel_sizes = [kernel_size1, kernel_size2, kernel_size3, kernel_size4]
        outputs_fc = [output_fc1, output_fc2]
        for i in range(1, nlayers_cnn):
            self.layers_cnn.append(torch.nn.Conv1d(current_dim, hidden_dim[i], 
                kernel_size=kernel_sizes[i], stride=self.stride, padding=self.padding))

            #self.layers_cnn.append(nn.BatchNorm1d(hidden_dim[i]))
            self.layers_cnn.append(nn.ReLU())
            self.layers_cnn.append(nn.MaxPool1d(kernel_size=self.kernel_pool))
            
            x_len_out = outputLen_MaxPool(x_len_out, self.layers_cnn[-1].kernel_size, self.layers_cnn[-1].padding, self.layers_cnn[-1].stride, self.layers_cnn[-1].dilation)
            current_dim = hidden_dim[i]
            
        self.flatten = nn.Flatten()
        
        input_final_fc = current_dim*int(x_len_out.item())

        for i in range(0, nlayers_fc):
            self.layers_fc.append(nn.Linear(input_final_fc, outputs_fc[i]))
            #self.layers_fc.append(nn.BatchNorm1d(outputs_fc[i]))
            self.layers_fc.append(nn.ReLU())
            #self.layers_fc.append(nn.Dropout(self.prob))
            input_final_fc = outputs_fc[i]
        
        self.dense_output = nn.Linear(input_final_fc, 1)

        
    def forward(self, seq, gex):
        x = self.conv1(seq.float())
        x = self.conv1_bn(x)
        x = self.relu1(x)
        x = x*gex.unsqueeze(2)
        x = self.pool(x)

        for layer in self.layers_cnn:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.layers_fc:
            x = layer(x)

        return self.dense_output(x)

class DeepSTARR(torch.nn.Module):

    """
    Pytorch implementation of the DeepSTARR model.
    Written by Monika.
    """
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, input_length, input_channel, output_channel1, output_channel2, output_channel3, output_channel4,
                 kernel_size1, kernel_size2, kernel_size3, kernel_size4, stride, kernel_pool,
                 padding, nlayers_cnn, nlayers_fc, output_fc1, output_fc2, prob):
        super(DeepSTARR, self).__init__()
        self.input_length = torch.tensor([input_length])
        self.input_channel = 4
        self.output_channel1 = output_channel1
        self.output_channel2 = output_channel2
        self.output_channel3 = output_channel3
        self.output_channel4 = output_channel4
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4
        self.nlayers_cnn = nlayers_cnn
        self.nlayers_fc = nlayers_fc
        self.stride = 1
        self.padding = 'same'
        self.kernel_pool = 2 # also default stride
        self.prob = prob
                
        # Conv layer 1
        self.conv1 = torch.nn.Conv1d(self.input_channel, self.output_channel1, 
            kernel_size=self.kernel_size1, stride=self.stride, padding=self.padding)

        x_len_out = self.input_length
            
        # Batch Normalization 1
        self.conv1_bn = nn.BatchNorm1d(self.output_channel1)
        self.relu1 = nn.ReLU()
        
        # Max pooling 1
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
        x_len_out = outputLen_MaxPool(x_len_out, self.pool.kernel_size, self.pool.padding, self.pool.stride, self.pool.dilation)
        
        current_dim = self.output_channel1
        self.layers_cnn = nn.ModuleList()
        self.layers_fc = nn.ModuleList()
        hidden_dim = [output_channel1, output_channel2, output_channel3, output_channel4]
        kernel_sizes = [kernel_size1, kernel_size2, kernel_size3, kernel_size4]
        outputs_fc = [output_fc1, output_fc2]
        for i in range(1, nlayers_cnn):
            self.layers_cnn.append(torch.nn.Conv1d(current_dim, hidden_dim[i], 
                kernel_size=kernel_sizes[i], stride=self.stride, padding=self.padding))

            self.layers_cnn.append(nn.BatchNorm1d(hidden_dim[i]))
            self.layers_cnn.append(nn.ReLU())
            self.layers_cnn.append(nn.MaxPool1d(kernel_size=self.kernel_pool))
            
            x_len_out = outputLen_MaxPool(x_len_out, self.layers_cnn[-1].kernel_size, self.layers_cnn[-1].padding, self.layers_cnn[-1].stride, self.layers_cnn[-1].dilation)
            current_dim = hidden_dim[i]
            
        self.flatten = nn.Flatten()
        
        input_final_fc = current_dim*int(x_len_out.item())

        for i in range(0, nlayers_fc):
            self.layers_fc.append(nn.Linear(input_final_fc, outputs_fc[i]))
            self.layers_fc.append(nn.BatchNorm1d(outputs_fc[i]))
            self.layers_fc.append(nn.ReLU())
            self.layers_fc.append(nn.Dropout(self.prob))
            input_final_fc = outputs_fc[i]
        
        self.dense_output = nn.Linear(input_final_fc, 1)
        
    def forward(self, x):
        x = self.relu1(self.conv1_bn(self.conv1(x.float())))
        x = self.pool(x)

        for layer in self.layers_cnn:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.layers_fc:
            x = layer(x)

        return self.dense_output(x)

class DeepSTARR_delta(torch.nn.Module):

    """
    Pytorch implementation of the DeepSTARR model.
    Adapted from DeepSTARR implementation above which has been written by Monika.
    """
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, input_length, input_channel, output_channel1, output_channel2, output_channel3, output_channel4,
                 kernel_size1, kernel_size2, kernel_size3, kernel_size4, stride, kernel_pool,
                 padding, nlayers_cnn, nlayers_fc, output_fc1, output_fc2, prob):
        super(DeepSTARR_delta, self).__init__()
        self.input_length = torch.tensor([input_length])
        self.input_channel = 4
        self.output_channel1 = output_channel1
        self.output_channel2 = output_channel2
        self.output_channel3 = output_channel3
        self.output_channel4 = output_channel4
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4
        self.nlayers_cnn = nlayers_cnn
        self.nlayers_fc = nlayers_fc
        self.stride = 1
        self.padding = 'same'
        self.kernel_pool = 2 # also default stride
        self.prob = prob
                
        # Conv layer 1
        self.conv1 = torch.nn.Conv1d(self.input_channel, self.output_channel1, 
            kernel_size=self.kernel_size1, stride=self.stride, padding=self.padding)

        x_len_out = self.input_length
            
        # Batch Normalization 1
        self.conv1_bn = nn.BatchNorm1d(self.output_channel1)
        self.relu1 = nn.ReLU()
        
        # Max pooling 1
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
        x_len_out = outputLen_MaxPool(x_len_out, self.pool.kernel_size, self.pool.padding, self.pool.stride, self.pool.dilation)
        
        current_dim = self.output_channel1
        self.layers_cnn = nn.ModuleList()
        self.layers_fc = nn.ModuleList()
        hidden_dim = [output_channel1, output_channel2, output_channel3, output_channel4]
        kernel_sizes = [kernel_size1, kernel_size2, kernel_size3, kernel_size4]
        outputs_fc = [output_fc1, output_fc2]
        for i in range(1, nlayers_cnn):
            self.layers_cnn.append(torch.nn.Conv1d(current_dim, hidden_dim[i], 
                kernel_size=kernel_sizes[i], stride=self.stride, padding=self.padding))

            self.layers_cnn.append(nn.BatchNorm1d(hidden_dim[i]))
            self.layers_cnn.append(nn.ReLU())
            self.layers_cnn.append(nn.MaxPool1d(kernel_size=self.kernel_pool))
            
            x_len_out = outputLen_MaxPool(x_len_out, self.layers_cnn[-1].kernel_size, self.layers_cnn[-1].padding, self.layers_cnn[-1].stride, self.layers_cnn[-1].dilation)
            current_dim = hidden_dim[i]
            
        self.flatten = nn.Flatten()
        
        input_final_fc = current_dim*int(x_len_out.item())

        for i in range(0, nlayers_fc):
            self.layers_fc.append(nn.Linear(input_final_fc, outputs_fc[i]))
            self.layers_fc.append(nn.BatchNorm1d(outputs_fc[i]))
            self.layers_fc.append(nn.ReLU())
            self.layers_fc.append(nn.Dropout(self.prob))
            input_final_fc = outputs_fc[i]
        
        self.dense_output_count = nn.Linear(input_final_fc, 1)
        self.dense_output_delta = nn.Linear(input_final_fc, 1)
        
    def forward(self, x):
        x = self.relu1(self.conv1_bn(self.conv1(x.float())))
        x = self.pool(x)

        for layer in self.layers_cnn:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.layers_fc:
            x = layer(x)
        
        pred_count = self.dense_output_count(x)
        pred_delta = self.dense_output_delta(x)

        return pred_count, pred_delta

class DeepSTARR_deltaonly(torch.nn.Module):

    """
    Pytorch implementation of the DeepSTARR model.
    Adapted from DeepSTARR implementation above which has been written by Monika.
    """
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, input_length, input_channel, output_channel1, output_channel2, output_channel3, output_channel4,
                 kernel_size1, kernel_size2, kernel_size3, kernel_size4, stride, kernel_pool,
                 padding, nlayers_cnn, nlayers_fc, output_fc1, output_fc2, prob):
        super(DeepSTARR_deltaonly, self).__init__()
        self.input_length = torch.tensor([input_length])
        self.input_channel = 4
        self.output_channel1 = output_channel1
        self.output_channel2 = output_channel2
        self.output_channel3 = output_channel3
        self.output_channel4 = output_channel4
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4
        self.nlayers_cnn = nlayers_cnn
        self.nlayers_fc = nlayers_fc
        self.stride = 1
        self.padding = 'same'
        self.kernel_pool = 2 # also default stride
        self.prob = prob
                
        # Conv layer 1
        self.conv1 = torch.nn.Conv1d(self.input_channel, self.output_channel1, 
            kernel_size=self.kernel_size1, stride=self.stride, padding=self.padding)

        x_len_out = self.input_length
            
        # Batch Normalization 1
        self.conv1_bn = nn.BatchNorm1d(self.output_channel1)
        self.relu1 = nn.ReLU()
        
        # Max pooling 1
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
        x_len_out = outputLen_MaxPool(x_len_out, self.pool.kernel_size, self.pool.padding, self.pool.stride, self.pool.dilation)
        
        current_dim = self.output_channel1
        self.layers_cnn = nn.ModuleList()
        self.layers_fc = nn.ModuleList()
        hidden_dim = [output_channel1, output_channel2, output_channel3, output_channel4]
        kernel_sizes = [kernel_size1, kernel_size2, kernel_size3, kernel_size4]
        outputs_fc = [output_fc1, output_fc2]
        for i in range(1, nlayers_cnn):
            self.layers_cnn.append(torch.nn.Conv1d(current_dim, hidden_dim[i], 
                kernel_size=kernel_sizes[i], stride=self.stride, padding=self.padding))

            self.layers_cnn.append(nn.BatchNorm1d(hidden_dim[i]))
            self.layers_cnn.append(nn.ReLU())
            self.layers_cnn.append(nn.MaxPool1d(kernel_size=self.kernel_pool))
            
            x_len_out = outputLen_MaxPool(x_len_out, self.layers_cnn[-1].kernel_size, self.layers_cnn[-1].padding, self.layers_cnn[-1].stride, self.layers_cnn[-1].dilation)
            current_dim = hidden_dim[i]
            
        self.flatten = nn.Flatten()
        
        input_final_fc = current_dim*int(x_len_out.item())

        for i in range(0, nlayers_fc):
            self.layers_fc.append(nn.Linear(input_final_fc, outputs_fc[i]))
            self.layers_fc.append(nn.BatchNorm1d(outputs_fc[i]))
            self.layers_fc.append(nn.ReLU())
            self.layers_fc.append(nn.Dropout(self.prob))
            input_final_fc = outputs_fc[i]
        
        self.dense_output_delta = nn.Linear(input_final_fc, 1)
        
    def forward(self, x):
        x = self.relu1(self.conv1_bn(self.conv1(x.float())))
        x = self.pool(x)

        for layer in self.layers_cnn:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.layers_fc:
            x = layer(x)
        
        pred_delta = self.dense_output_delta(x)

        return pred_delta


class DeepSTARR_delta_2count(torch.nn.Module):

    """
    Pytorch implementation of the DeepSTARR model.
    Adapted from DeepSTARR implementation above which has been written by Monika.
    """
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, input_length, input_channel, output_channel1, output_channel2, output_channel3, output_channel4,
                 kernel_size1, kernel_size2, kernel_size3, kernel_size4, stride, kernel_pool,
                 padding, nlayers_cnn, nlayers_fc, output_fc1, output_fc2, prob):
        super(DeepSTARR_delta_2count, self).__init__()
        self.input_length = torch.tensor([input_length])
        self.input_channel = 4
        self.output_channel1 = output_channel1
        self.output_channel2 = output_channel2
        self.output_channel3 = output_channel3
        self.output_channel4 = output_channel4
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4
        self.nlayers_cnn = nlayers_cnn
        self.nlayers_fc = nlayers_fc
        self.stride = 1
        self.padding = 'same'
        self.kernel_pool = 2 # also default stride
        self.prob = prob
                
        # Conv layer 1
        self.conv1 = torch.nn.Conv1d(self.input_channel, self.output_channel1, 
            kernel_size=self.kernel_size1, stride=self.stride, padding=self.padding)

        x_len_out = self.input_length
            
        # Batch Normalization 1
        self.conv1_bn = nn.BatchNorm1d(self.output_channel1)
        self.relu1 = nn.ReLU()
        
        # Max pooling 1
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
        x_len_out = outputLen_MaxPool(x_len_out, self.pool.kernel_size, self.pool.padding, self.pool.stride, self.pool.dilation)
        
        current_dim = self.output_channel1
        self.layers_cnn = nn.ModuleList()
        self.layers_fc = nn.ModuleList()
        hidden_dim = [output_channel1, output_channel2, output_channel3, output_channel4]
        kernel_sizes = [kernel_size1, kernel_size2, kernel_size3, kernel_size4]
        outputs_fc = [output_fc1, output_fc2]
        for i in range(1, nlayers_cnn):
            self.layers_cnn.append(torch.nn.Conv1d(current_dim, hidden_dim[i], 
                kernel_size=kernel_sizes[i], stride=self.stride, padding=self.padding))

            self.layers_cnn.append(nn.BatchNorm1d(hidden_dim[i]))
            self.layers_cnn.append(nn.ReLU())
            self.layers_cnn.append(nn.MaxPool1d(kernel_size=self.kernel_pool))
            
            x_len_out = outputLen_MaxPool(x_len_out, self.layers_cnn[-1].kernel_size, self.layers_cnn[-1].padding, self.layers_cnn[-1].stride, self.layers_cnn[-1].dilation)
            current_dim = hidden_dim[i]
            
        self.flatten = nn.Flatten()
        
        input_final_fc = current_dim*int(x_len_out.item())

        for i in range(0, nlayers_fc):
            self.layers_fc.append(nn.Linear(input_final_fc, outputs_fc[i]))
            self.layers_fc.append(nn.BatchNorm1d(outputs_fc[i]))
            self.layers_fc.append(nn.ReLU())
            self.layers_fc.append(nn.Dropout(self.prob))
            input_final_fc = outputs_fc[i]
        
        self.dense_output_count1 = nn.Linear(input_final_fc, 1)
        self.dense_output_count2 = nn.Linear(input_final_fc, 1)
        #self.dense_output_delta = nn.Linear(input_final_fc, 1)
        
    def forward(self, x):
        x = self.relu1(self.conv1_bn(self.conv1(x.float())))
        x = self.pool(x)

        for layer in self.layers_cnn:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.layers_fc:
            x = layer(x)
        
        pred_count1 = self.dense_output_count1(x)
        pred_count2 = self.dense_output_count2(x)
        #pred_delta = self.dense_output_delta(x)

        return pred_count1, pred_count2#, pred_delta

class DeepSTARR_delta_2count_v2(torch.nn.Module):

    """
    Pytorch implementation of the DeepSTARR model.
    Adapted from DeepSTARR implementation above which has been written by Monika.
    """
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, input_length, input_channel, output_channel1, output_channel2, output_channel3, output_channel4,
                 kernel_size1, kernel_size2, kernel_size3, kernel_size4, stride, kernel_pool,
                 padding, nlayers_cnn, nlayers_fc, output_fc1, output_fc2, prob):
        super(DeepSTARR_delta_2count_v2, self).__init__()
        self.input_length = torch.tensor([input_length])
        self.input_channel = 4
        self.output_channel1 = output_channel1
        self.output_channel2 = output_channel2
        self.output_channel3 = output_channel3
        self.output_channel4 = output_channel4
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4
        self.nlayers_cnn = nlayers_cnn
        self.nlayers_fc = nlayers_fc
        self.stride = 1
        self.padding = 'same'
        self.kernel_pool = 2 # also default stride
        self.prob = prob
                
        # Conv layer 1
        self.conv1 = torch.nn.Conv1d(self.input_channel, self.output_channel1, 
            kernel_size=self.kernel_size1, stride=self.stride, padding=self.padding)

        x_len_out = self.input_length
            
        # Batch Normalization 1
        self.conv1_bn = nn.BatchNorm1d(self.output_channel1)
        self.relu1 = nn.ReLU()
        
        # Max pooling 1
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
        x_len_out = outputLen_MaxPool(x_len_out, self.pool.kernel_size, self.pool.padding, self.pool.stride, self.pool.dilation)
        
        current_dim = self.output_channel1
        self.layers_cnn = nn.ModuleList()
        self.layers_fc = nn.ModuleList()
        hidden_dim = [output_channel1, output_channel2, output_channel3, output_channel4]
        kernel_sizes = [kernel_size1, kernel_size2, kernel_size3, kernel_size4]
        outputs_fc = [output_fc1, output_fc2]
        for i in range(1, nlayers_cnn):
            self.layers_cnn.append(torch.nn.Conv1d(current_dim, hidden_dim[i], 
                kernel_size=kernel_sizes[i], stride=self.stride, padding=self.padding))

            self.layers_cnn.append(nn.BatchNorm1d(hidden_dim[i]))
            self.layers_cnn.append(nn.ReLU())
            self.layers_cnn.append(nn.MaxPool1d(kernel_size=self.kernel_pool))
            
            x_len_out = outputLen_MaxPool(x_len_out, self.layers_cnn[-1].kernel_size, self.layers_cnn[-1].padding, self.layers_cnn[-1].stride, self.layers_cnn[-1].dilation)
            current_dim = hidden_dim[i]
            
        self.flatten = nn.Flatten()
        
        input_final_fc = current_dim*int(x_len_out.item())

        for i in range(0, nlayers_fc):
            self.layers_fc.append(nn.Linear(input_final_fc, outputs_fc[i]))
            self.layers_fc.append(nn.BatchNorm1d(outputs_fc[i]))
            self.layers_fc.append(nn.ReLU())
            self.layers_fc.append(nn.Dropout(self.prob))
            input_final_fc = outputs_fc[i]
        
        self.dense_output_count1 = nn.Linear(input_final_fc, 1)
        self.dense_output_count2 = nn.Linear(input_final_fc, 1)
        self.dense_output_delta = nn.Linear(input_final_fc, 1)
        
    def forward(self, x):
        x = self.relu1(self.conv1_bn(self.conv1(x.float())))
        x = self.pool(x)

        for layer in self.layers_cnn:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.layers_fc:
            x = layer(x)
        
        pred_count1 = self.dense_output_count1(x)
        pred_count2 = self.dense_output_count2(x)
        pred_delta = self.dense_output_delta(x)

        return pred_count1, pred_count2, pred_delta


deepstarr_default_params = {
    "input_length":1001,
    "input_channel":4, 
    "output_channel1":256,
    "output_channel2":120,
    "output_channel3":60, 
    "output_channel4":60, 
    "kernel_size1":7, 
    "kernel_size2":3, 
    "kernel_size3":3, 
    "kernel_size4":3, 
    "stride":1, 
    "kernel_pool":2,
    "padding":"same", 
    "nlayers_cnn":4, 
    "nlayers_fc":2,
    "output_fc1":64, 
    "output_fc2":256, 
    "prob":0.4
}


########################
#explaiNN architectures
########################

class ExpActivation(nn.Module):
    """
    Exponential activation function from Koo & Ploenzke, 2021 (PMID: 34322657)
    """
    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, x):
        return torch.exp(x)

class Unsqueeze(torch.nn.Module):
    """
    Unsqueeze for sequential models
    """
    def forward(self, x):
        return x.unsqueeze(-1)

class ExplaiNN(nn.Module):
    """
    The ExplaiNN model (PMID: 37370113)
    """
    def __init__(self, num_cnns, input_length, num_classes, filter_size=19, num_fc=2, pool_size=7, pool_stride=7,
                 weight_path=None):
        """
        :param num_cnns: int, number of independent cnn units
        :param input_length: int, input sequence length
        :param num_classes: int, number of outputs
        :param filter_size: int, size of the unit's filter, default=19
        :param num_fc: int, number of FC layers in the unit, default=2
        :param pool_size: int, size of the unit's maxpooling layer, default=7
        :param pool_stride: int, stride of the unit's maxpooling layer, default=7
        :param weight_path: string, path to the file with model weights
        """
        super(ExplaiNN, self).__init__()

        self._options = {
            "num_cnns": num_cnns,
            "input_length": input_length,
            "num_classes": num_classes,
            "filter_size": filter_size,
            "num_fc": num_fc,
            "pool_size": pool_size,
            "pool_stride": pool_stride,
            "weight_path": weight_path
        }

        if num_fc == 0:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(input_length - (filter_size-1)),
                nn.Flatten())
        elif num_fc == 1:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(((input_length - (filter_size-1)) - (pool_size-1)-1)/pool_stride + 1) * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Flatten())
        elif num_fc == 2:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(((input_length - (filter_size-1)) - (pool_size-1)-1)/pool_stride + 1) * num_cnns,
                          out_channels=100 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(in_channels=100 * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Flatten())
        else:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(((input_length - (filter_size-1)) - (pool_size-1)-1)/pool_stride + 1) * num_cnns,
                          out_channels=100 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU())

            self.linears_bg = nn.ModuleList([nn.Sequential(nn.Dropout(0.3),
                                                           nn.Conv1d(in_channels=100 * num_cnns,
                                                                     out_channels=100 * num_cnns, kernel_size=1,
                                                                     groups=num_cnns),
                                                           nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                                                           nn.ReLU()) for i in range(num_fc - 2)])

            self.last_linear = nn.Sequential(nn.Dropout(0.3),
                                             nn.Conv1d(in_channels=100 * num_cnns, out_channels=1 * num_cnns,
                                                       kernel_size=1,
                                                       groups=num_cnns),
                                             nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                                             nn.ReLU(),
                                             nn.Flatten())

        self.final = nn.Linear(num_cnns, num_classes)

        if weight_path:
            self.load_state_dict(torch.load(weight_path))

    def forward(self, x):
        x = x.repeat(1, self._options["num_cnns"], 1)
        if self._options["num_fc"] <= 2:
            outs = self.linears(x)
        else:
            outs = self.linears(x)
            for i in range(len(self.linears_bg)):
                outs = self.linears_bg[i](outs)
            outs = self.last_linear(outs)
        out = self.final(outs)
        return out

class ExplaiNN_dynamic_v1(nn.Module):
    """
    The ExplaiNN model (PMID: 37370113)
    """
    def __init__(self, num_cnns, input_length, num_classes, filter_size=19, num_fc=2, pool_size=7, pool_stride=7,
                 weight_path=None):
        """
        :param num_cnns: int, number of independent cnn units
        :param input_length: int, input sequence length
        :param num_classes: int, number of outputs
        :param filter_size: int, size of the unit's filter, default=19
        :param num_fc: int, number of FC layers in the unit, default=2
        :param pool_size: int, size of the unit's maxpooling layer, default=7
        :param pool_stride: int, stride of the unit's maxpooling layer, default=7
        :param weight_path: string, path to the file with model weights
        """
        super(ExplaiNN_dynamic_v1, self).__init__()

        self._options = {
            "num_cnns": num_cnns,
            "input_length": input_length,
            "num_classes": num_classes,
            "filter_size": filter_size,
            "num_fc": num_fc,
            "pool_size": pool_size,
            "pool_stride": pool_stride,
            "weight_path": weight_path
        }


        self.conv1 = nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size, groups=num_cnns)
        self.bn1 = nn.BatchNorm1d(num_cnns)
        self.exp1 = ExpActivation()
        self.pool1 = nn.MaxPool1d(pool_size, pool_stride)
        self.flatten1 = nn.Flatten()
        self.unsqueeze1 = Unsqueeze()

        self.dense1 = nn.Conv1d(in_channels=int(((input_length - (filter_size-1)) - (pool_size-1)-1)/pool_stride + 1) * num_cnns,
                          out_channels=100 * num_cnns, kernel_size=1,
                          groups=num_cnns)
        self.bn2 = nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.dense2 = nn.Conv1d(in_channels=100 * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns)
        
        self.bn3 = nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True)
        self.relu2 = nn.ReLU()
        self.flatten2 = nn.Flatten()
        self.final = nn.Linear(num_cnns, num_classes)


    def forward(self, seq, gex):
        x = seq.repeat(1, self._options["num_cnns"], 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.exp1(x)
        x = x*gex.unsqueeze(2)
        x = self.pool1(x)
        x = self.flatten1(x)
        x = self.unsqueeze1(x)
        x = self.dense1(x)
        x = self.bn2(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.bn3(x)
        x = self.relu2(x)
        x = self.flatten2(x)
        out = self.final(x)
        return out
