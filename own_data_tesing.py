from Solubilitylib import *
import sys

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        inputs = inputs.unsqueeze(2).float()
        embeddings =  torch.cat((
            inputs, inputs), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        return encoding

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        return encoding

class bert_cla(nn.Module):
  def __init__(self,cnn_net, rnn_net, n_classes=2):
    super(bert_cla, self).__init__()
    self.bert = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
    self.drop1 = nn.Dropout(p=0.2)
    self.drop2 = nn.Dropout(p=0.5)
    self.drop3 = nn.Dropout(p=0.5)
    self.drop4 = nn.Dropout(p=0.2)
    self.cnn = cnn_net
    self.rnn = rnn_net
    self.outbio = nn.Linear(123, 512)
    self.relu = nn.ReLU()
    self.out = nn.Linear((800+256+512), n_classes)
    self.softmax = nn.Softmax(dim=1)
  def forward(self, input_ids):
    bioinfo = input_ids.pop("bioinfo")
    outputs = self.bert(**input_ids)
    pooled_out = outputs.pooler_output
    output = self.drop1(pooled_out)
    cnnout =  self.cnn(output)
    rnn_out = self.rnn(input_ids['input_ids'])
    out1 = self.out(torch.cat([self.drop2(cnnout), self.drop3(rnn_out), self.drop4(self.relu(self.outbio(bioinfo)))], dim=1))
    return out1

def get_rnn(vocab_size = 20, embed_size=64, num_hiddens=64, num_layers=2):
  net = BiRNN(vocab_size, embed_size, num_hiddens, num_layers)
  def init_weights(m):
      if type(m) == nn.Linear:
          nn.init.xavier_uniform_(m.weight)
      if type(m) == nn.LSTM:
          for param in m._flat_weights_names:
              if "weight" in param:
                  nn.init.xavier_uniform_(m._parameters[param])
  net.apply(init_weights);
  return net

def evaluate_accuracy_gpu(net, data_iter, device=None):
    net.cuda()
    net.eval()
    if isinstance(net, nn.Module):
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    i = 1
    with torch.no_grad():
        for data in data_iter:
            x_input_ids, x_token_type_ids, x_attention_mask = data['input_ids'], data['token_type_ids'], data['attention_mask']
            x_bioinfo = data['bioinfo']
            X = {}
            X['input_ids'] = x_input_ids.cuda()
            X['token_type_ids'] = x_token_type_ids.cuda()
            X['attention_mask'] = x_attention_mask.cuda()
            X['bioinfo'] = x_bioinfo.cuda()
            if i == 1:
              zz0 =  torch.softmax(net(X),dim=1)
              zzlabel0 = data['labels']
            else:
              zz =  torch.softmax(net(X),dim=1)
              zz0 = torch.cat((zz0,zz),0)
              zzlabel = data['labels']
              zzlabel0 = torch.cat((zzlabel0,zzlabel),0)
            i += 1
    return zz0,zzlabel0
  
class assesmentDataset(Dataset):
    def __init__(self, tokenizer, own_file_path, split = 'test', max_length=1200):
        self.valFolderPath = own_file_path
        self.valFilePath = os.path.join(self.valFolderPath, 'own_data.csv')
        self.valbio = os.path.join(self.valFolderPath, 'own_data.fasta')
        self.seqs, self.labels, self.bioinfo = self.load_dataset(self.valFilePath, self.valbio)
        self.max_length = max_length
        self.tokenizer = tokenizer

    def load_dataset(self, path, biopath):
        df = pd.read_csv(path, header = 0)
        seq = list(df['seq'])
        label = list(df['labels'])
        protein = iFeatureOmegaCLI.iProtein(biopath)
        protein.get_descriptor("PAAC")
        bio = protein.encodings
        bioinfo1 = ((bio - bio.mean())/(bio.std())).values
        protein.get_descriptor("CKSAAGP type 1")
        bio = protein.encodings
        bioinfo2 = ((bio - bio.mean())/(bio.std())).values
        bioinfo = np.concatenate((bioinfo1,bioinfo2),axis=1) 
        
        assert len(seq) == len(label)
        return seq, label, bioinfo

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = " ".join("".join(self.seqs[idx].split()))
        seq = re.sub(r"[UZOB]", "X", seq)
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])
        sample['bioinfo'] = torch.tensor(self.bioinfo[idx], dtype=torch.float32)

        return sample
    
own_data_path = sys.argv[1]
if not os.path.exists(own_data_path):
    print("own_data.csv and own_data.fasta files are not exist")
    sys.exit()

print("the best model loading...")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd" )
rnn_net = get_rnn(vocab_size = tokenizer.vocab_size, embed_size=64, num_hiddens=64, num_layers=2)
embed_size, kernel_sizes, nums_channels = 1, [1, 2, 3, 4, 5, 6, 7, 8], [100, 100, 100, 100, 100, 100, 100, 100]
cnn_net = TextCNN(tokenizer.vocab_size, embed_size, kernel_sizes, nums_channels)
modeltest = bert_cla(cnn_net=cnn_net, rnn_net=rnn_net, n_classes=2)
     
net = modeltest
path_checkpoint = "/home/bli/logbacktrain/ProteinSolubilityPrediction/ProtSol/best_model/bestmodel.pkl"
checkpoint = torch.load(path_checkpoint)
net.load_state_dict(checkpoint['net'], False)
net.eval()
devices = try_all_gpus()

print("loading your own data ...")
test_ass = assesmentDataset(tokenizer = tokenizer, own_file_path = own_data_path, split = 'val', max_length=1200)
test_asses = DataLoader(test_ass, batch_size = 32, shuffle = False)
zzz = evaluate_accuracy_gpu(net, test_asses, device=devices)
y_hat = zzz[0].argmax(1).cpu().numpy()

csv_path = own_data_path + "own_data.csv"
df = pd.read_csv(csv_path, header = 0)
df = df.assign(y_hat=y_hat)
df.to_csv(csv_path,index=False)

print("Prediction completed, plz check the ((y_hat_own_data.csv)) file in (own_dataset) folder")