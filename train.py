from Solubilitylib import *

bert_model_path = "model"
checkpoint_path = "checkpoint"

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

def main(net, train_iter, test_iter, val_iter, val1_iter, loss, trainer, num_epochs, start_epoch, devices = try_all_gpus()):
    logger = log_init()
    timer, num_batches = Timer(), len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    best_acc = 0
    for epoch in range(start_epoch, num_epochs):
        metric = Accumulator(4)
        print(f'epoch {epoch + 1} is processing')
        for i, data in enumerate(train_iter):
            timer.start()
            labels = data.pop('labels').cuda()
            features = data
            l, acc = train_batch(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        val_acc = evaluate_accuracy_gpu(net, val_iter)
        val1_acc = evaluate_accuracy_gpu(net, val1_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        logger.info(f'epoch {epoch + 1}/{num_epochs}:'\
            f' loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}, val acc {val_acc:.3f}, val1 acc {val1_acc:.3f}'\
            f' | {metric[2]*num_epochs/timer.sum():.1f} examples/sec on {str(devices)}') 

        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)

        if val1_acc >= best_acc:
            best_acc = val1_acc
            checkpoint = {
              "net": net.module.state_dict(),
              'optimizer':trainer.state_dict(),
              "epoch": epoch
              }
            torch.save(checkpoint, checkpoint_path + '/bestmodel.pkl')
            
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    rnn_net = get_rnn(vocab_size = tokenizer.vocab_size, embed_size=64, num_hiddens=64, num_layers=2)
    embed_size, kernel_sizes, nums_channels = 1, [1, 2, 3, 4, 5, 6, 7, 8], [100, 100, 100, 100, 100, 100, 100, 100]
    cnn_net = kmersCNN(tokenizer.vocab_size, embed_size, kernel_sizes, nums_channels)
    print("bert model loading...")
    modeltest = bert_cla(cnn_net=cnn_net, rnn_net=rnn_net, n_classes=2)
    print("bert model loading completed...")
    print("data loading...")
    train_dataset = SolubilityDatasetBio(tokenizer = tokenizer, split="train", max_length=1200)
    test_dataset = SolubilityDatasetBio(tokenizer = tokenizer, split="test", max_length=1200)
    val_dataset = SolubilityDatasetBio(tokenizer = tokenizer, split="val", max_length=1200)
    val1_dataset = SolubilityDatasetBio(tokenizer = tokenizer, split="val1", max_length=1200)

    batch_size = 2
    train_iter = DataLoader(train_dataset, batch_size, shuffle = True)
    test_iter = DataLoader(test_dataset, batch_size, shuffle = True)
    val_iter = DataLoader(val_dataset, batch_size, shuffle = True)
    val1_iter = DataLoader(val1_dataset, batch_size, shuffle = True)

    print("dataset load completed!")
    print("TRIANING BEGIN!!!")

    num_epochs = 40
    net = modeltest
    net.out.apply(init_weights);

    trainer = torch.optim.Adam([{"params": net.bert.parameters(), "lr": 1e-7},
                                {"params": net.cnn.parameters(),"lr": 1e-5},
                                {"params": net.rnn.parameters(),"lr": 1e-5}],
                                lr = 1e-7, weight_decay=1e-3)
    net.train()
    devices = try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none").cuda()

    RESUME = False
    
    start_epoch = 0
    
    if RESUME:
        path_checkpoint = checkpoint_path + "/bestmodel.pkl"
        checkpoint = torch.load(path_checkpoint)
        net.load_state_dict(checkpoint['net'])
        net.cuda()
        trainer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    main(net, train_iter, test_iter, val_iter, val1_iter, loss, trainer, num_epochs, start_epoch, devices)

