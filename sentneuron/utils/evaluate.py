
def evaluate_sentiment_neuron(neuron, seq_dataset, batch_size, seq_length, test_shard_path):
    h_init = self.neuron(batch_size)

    shard_content = seq_dataset.read(open(test_shard_path, "r"))
    sequence = seq_dataset.encode_sequence(shard_content)
    sequence = self.__batchify_sequence(torch.tensor(sequence, dtype=torch.uint8, device=self.device), batch_size)

    n_batches = sequence.size(0)//seq_length

    smooth_loss = -torch.log(torch.tensor(1.0/seq_dataset.encoding_size)).item() * seq_length

    for batch_ix in range(n_batches - 1):
        batch = sequence.narrow(0, batch_ix * seq_length, seq_length + 1).long()

        h = h_init

        loss = 0
        for t in range(seq_length):
            h, y = neuron(batch[t], h)
            loss += loss_function(y, batch[t+1])

        h_init = (ag.Variable(h[0].data), ag.Variable(h[1].data))

        smooth_loss = smooth_loss * 0.999 + loss.item() * 0.001

    return smooth_loss
