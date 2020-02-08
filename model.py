import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    # Followed suggestions in Udacity's knowledge base and PyTorch tutorials.
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_size
        self.caption_embeddings = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True) 
        self.linear1 = nn.Linear(hidden_size,vocab_size)
        
    
    def forward(self, features, captions):
        #pass
        embeds = self.caption_embeddings(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1),embeds),1)
        
        lstm_out,hidden = self.lstm(inputs)
        #hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
        out = self.linear1(lstm_out)
        
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        retList = []                       # List of word indices to return
        lstm_out,states = self.lstm(inputs)     # Pass image feature through decoder to initialize
        #print(lstm_out.shape)
        out = self.linear1(lstm_out)
        #print(linOut[:10])
        
        for i in range(max_len):           # Pass output of decoder as input in the next step. Collec int in retList
            
            out_word = out.argmax().item()
            retList.append(out_word)
            new_word = out_word
            
            
            caption = torch.Tensor([new_word]).long().to("cuda")
            #caption = caption.to("cuda")
            
            embed = self.caption_embeddings(caption.unsqueeze(1))
            lstm_out,states = self.lstm(embed,states)
            out = self.linear1(lstm_out)
            
        return retList
        