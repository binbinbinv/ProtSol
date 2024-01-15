from Solubilitylib import *
import random



bert_model_path = "./model"
bestmodel_path = "./best_model"

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

class kmersCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(kmersCNN, self).__init__(**kwargs)
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
    self.bert = BertModel.from_pretrained(bert_model_path)
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

def assement_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度

    Defined in :numref:`sec_lenet`"""
    net.cuda()
    net.eval()  # 设置为评估模式
    if isinstance(net, nn.Module):
        if not device:
            device = next(iter(net.parameters())).device
    i = 1
    with torch.no_grad():
        for data in tqdm(data_iter):
            x_input_ids, x_token_type_ids, x_attention_mask = data['input_ids'], data['token_type_ids'], data['attention_mask']
            x_bioinfo = data['bioinfo']
            X = {}
            X['input_ids'] = x_input_ids.cuda()
            X['token_type_ids'] = x_token_type_ids.cuda()
            X['attention_mask'] = x_attention_mask.cuda()
            X['bioinfo'] = x_bioinfo.cuda()
            if i == 1:
              zz0 =  torch.softmax(net(X),dim=1)
            else:
              zz =  torch.softmax(net(X),dim=1)
              zz0 = torch.cat((zz0,zz),0)
            i = i + 1
    return zz0

def read_fasta(fasta_file):
  """
  读取fasta文件并生成dataframe

  Args:
    fasta_file: fasta文件路径

  Returns:
    dataframe
  """

  with open(fasta_file, "r") as f:
    data = []
    for line in f:
      if line.startswith(">"):
        id = line.strip()[1:]
      else:
        sequence = line.strip()
        data.append((id, sequence))
  return pd.DataFrame(data, columns=["id", "sequence"])

def print_box(message):
    box_width = 40
    message = f" {message} "
    padding = (box_width - len(message)) // 2
    border = '*' * box_width
    padding_str = '*' + ' ' * padding

    print(border)
    print(padding_str + message + ' ' * (box_width - len(padding_str) - len(message)) + '*')
    print(border)
                
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    rnn_net = get_rnn(vocab_size = tokenizer.vocab_size, embed_size=64, num_hiddens=64, num_layers=2)
    embed_size, kernel_sizes, nums_channels = 1, [1, 2, 3, 4, 5, 6, 7, 8], [100, 100, 100, 100, 100, 100, 100, 100]
    cnn_net = kmersCNN(tokenizer.vocab_size, embed_size, kernel_sizes, nums_channels)
    print_box("bert model loading...")
    modeltest = bert_cla(cnn_net=cnn_net, rnn_net=rnn_net, n_classes=2)
    print_box("bert model loading completed...")
    print_box("data loading...")

    predict_path = "./Predict/NEED_TO_PREPARE/own_data.fasta"
    df = read_fasta(predict_path)

    predict_dataset = SolubilityDatasetBioPrediction(predict_path, tokenizer = tokenizer,  max_length=1024)

    batch_size = 4
    predict_iter = DataLoader(predict_dataset, batch_size, shuffle = False)

    print_box("Dataset for prediction loading completed!")

    net = modeltest
    devices = try_all_gpus()
    path_bestmodel = bestmodel_path + "/bestmodel.pkl"
    checkpoint = torch.load(path_bestmodel)
    net.load_state_dict(checkpoint['net'])
    net.cuda().eval()
    print_box("Prediciton BEGIN!!!")

    y_scores = assement_accuracy_gpu(net, predict_iter, device=devices)

    y_hat = list(y_scores.argmax(1).cpu().numpy())
    
    # 保存修改后的 DataFrame 到 CSV 文件
    df['solubility'] = y_hat
    df.to_csv("./Predict/Output.csv", index=False)

    print_box("Prediction Completed and Check the Output.csv")
    
    


