import torch
import torch.nn as nn

class PromptEncoderPrefixLSTM(torch.nn.Module):
    def __init__(self, hidden_size, tokenizer, device):
        super().__init__()
        self.cloze_length = 3
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.device = device
        # init model
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=0.0,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.cloze_length * self.hidden_size))


    def tokenize(self, query, prompt_tokens):
        token_ids = self.tokenizer(' ' + query)['input_ids']
        return prompt_tokens * self.cloze_length + token_ids

    def forward(self, tokens=None, embeddings=None):

        bsz, seqlen, embed_dim = embeddings.shape
        input_embeddings = embeddings[:, self.cloze_length:, :] # bsz, seqlen, embed_dim

        # run the lstm, pool the output, and upproject it so we have the right number of embeddings
        lstm_output = self.lstm_head(input_embeddings)[0]
        output_embeddings = self.mlp_head(torch.max(lstm_output, dim=1).values) # max pool
        output_embeddings = output_embeddings.reshape(bsz, self.cloze_length, self.hidden_size)
        result_embeddings = torch.cat((output_embeddings, embeddings[:, self.cloze_length:]), dim=1) #  output_embeddings

        return result_embeddings