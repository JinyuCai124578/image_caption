import torch
from torch import nn
import torchvision


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        # (batch_size, 2048, image_size/32, image_size/32)
        out = self.resnet(images)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # [b, num_pixels, attention_dim]
        att1 = self.encoder_att(encoder_out)
        # [b, attention_dim]
        att2 = self.decoder_att(decoder_hidden)
        # [b, num_pixels]
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        # [b, num_pixels]
        alpha = self.softmax(att)
        # [b, encoder_dim]
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim,
                 vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state [b, decoder_dim]
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        # [b, num_pixels, encoder_dim]
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths
        # [b, 1] -> [b], [b]
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        # [b, max_len, embed_dim]
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        # [b, decoder_dim]
        h, c = self.init_hidden_state(encoder_out)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        # [b, max_len, vocab_size]
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        # [b, num_pixels, vocab_size]
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            # [b, encoder_dim], [b, num_pixels] -> [batch_size_t, encoder_dim], [batch_size_t, num_pixels]
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            # [batch_size_t, encoder_dim]
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar,
            attention_weighted_encoding = gate * attention_weighted_encoding
            # [batch_size_t, decoder_dim]
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            # [batch_size_t, vocab_size]
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def sample(self, encoder_out, startseq_idx, endseq_idx=-1, max_len=40,
               return_alpha=False, method='beam', beam_size=3):
        """
        Samples captions in batch for given image features (Greedy search/ Beam search).
        :param encoder_out = [b, enc_image_size, enc_image_size, 2048]
        :return [b, max_len]
        """
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        batch_size = encoder_out.size(0)


        sampled_ids = []  # list of [b,]
        alphas = []

        assert method in ['beam', 'greed']
        if method == 'beam':
            for instance in range(batch_size):
                # prepare beam search
                topk_logprobs = torch.zeros(beam_size).to(encoder_out.device)
                done_beam = []
                # [beam_size, enc_image_size, emc_image_size, 2048]
                encoder_out_instance = encoder_out[instance].unsqueeze(0).repeat(
                    beam_size, *([1] * len(encoder_out[instance].shape)))
                # [beam_size, enc_image_size, enc_image_size, 2048] -> [beam_size, num_pixels, 2048]
                encoder_out_instance = encoder_out_instance.view(beam_size, -1, encoder_dim)
                # [beam, num_pixels, ]
                h, c = self.init_hidden_state(encoder_out_instance)
                # [beam_size, 1]
                prev_timestamp_word_instance = torch.LongTensor([
                    [startseq_idx]] * beam_size).to(encoder_out.device)
                for t in range(max_len):
                    # [beam_size, 1] -> [beam_size, emdeb_dim]
                    embeddings = self.embedding(prev_timestamp_word_instance).squeeze(1)
                    # ([beam_size, encoder_dim], [beam_size, num_pixels])
                    awe, alpha = self.attention(encoder_out_instance, h)
                    # [beam_size, enc_image_size, enc_image_size] -> [beam_size, 1, enc_image_size, enc_image_size]
                    alpha = alpha.view(-1, enc_image_size, enc_image_size).unsqueeze(1)
                    # [beam_size, embed_dim]
                    gate = self.sigmoid(self.f_beta(h))  # gating scalar
                    # [beam_size, embed_dim]
                    awe = gate * awe
                    # ([beam_size, decoder_dim], )
                    h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
                    # [beam_size, vocab_size]
                    predicted_prob = self.fc(h)
                    
                    logprobs_t = torch.log_softmax(predicted_prob, dim=1) + topk_logprobs.unsqueeze(1)
                    if t == 0:
                        topk_logprobs, topk_words = logprobs_t[0].topk(
                            beam_size, 0 , True, True)
                    else:
                        topk_logprobs, topk_words = logprobs_t.view(-1).topk(
                            beam_size, 0, True, True)
                    # prev_words_beam = topk_words // self.vocab_size  # [beam_size,]  
                    prev_word_beam = torch.div(topk_words, self.vocab_size, rounding_mode="trunc")
                    next_word = topk_words % self.vocab_size

                    if t == 0:
                        seqs_instance = next_word.unsqueeze(1)
                    else:
                        seqs_instance = torch.cat([
                            seqs_instance[prev_word_beam],
                            next_word.unsqueeze(1)], dim=1)
                    
                    is_end = next_word == endseq_idx
                    if t == max_len - 1:
                        is_end.fill_(1)

                    for beam_idx in range(beam_size):
                        if is_end[beam_idx]:
                            final_beam = {
                                'seq': seqs_instance[beam_idx].clone(),
                                'score': topk_logprobs[beam_idx].item() / (t + 1)
                            }
                            done_beam.append(final_beam)
                    topk_logprobs[is_end] -= 1000
                    # [beam_size] -> [beam_size, 1]
                    prev_timestamp_word_instance = next_word.unsqueeze(1)
                done_beam = sorted(done_beam, key=lambda x: -x['score'])
                sampled_ids.append(done_beam[0]['seq'])
                alphas.append(alpha[0])
                # [b, max_len]
            sampled_ids = torch.stack(sampled_ids, 0)
            return (sampled_ids, torch.cat(alphas, 1)) if return_alpha else sampled_ids
                    
        else:
            # [b, enc_image_size, enc_image_size, 2048] -> [b, num_pixels, 2048]
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
            # [b, num_pixels, ]
            h, c = self.init_hidden_state(encoder_out)
            # [b, 1]
            prev_timestamp_words = torch.LongTensor([
                [startseq_idx]] * batch_size).to(encoder_out.device)
            for t in range(max_len):
                # [b, 1] -> [b, embed_dim]
                embeddings = self.embedding(prev_timestamp_words).squeeze(1)
                # ([b, encoder_dim], [b, num_pixels])
                awe, alpha = self.attention(encoder_out, h)
                # [b, enc_image_size, enc_image_size] -> [b, 1, enc_image_size, enc_image_size]
                alpha = alpha.view(-1, enc_image_size, enc_image_size).unsqueeze(1)

                # [b, embed_dim]
                gate = self.sigmoid(self.f_beta(h))  # gating scalar
                # [b, embed_dim]
                awe = gate * awe

                # ([b, decoder_dim], )
                h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
                # [b, vocab_size]
                predicted_prob = self.fc(h)
                # [b]
                predicted = predicted_prob.argmax(1)

                sampled_ids.append(predicted)
                alphas.append(alpha)

                # [b] -> [b, 1]
                prev_timestamp_words = predicted.unsqueeze(1)
            # [b, max_len]
            sampled_ids = torch.stack(sampled_ids, 1)
            return (sampled_ids, torch.cat(alphas, 1)) if return_alpha else sampled_ids


class Captioner(nn.Module):
    def __init__(self, encoded_image_size, attention_dim, embed_dim,
        decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5, **kwargs):
        super().__init__()
        self.encoder = Encoder(encoded_image_size=encoded_image_size)
        self.decoder = DecoderWithAttention(attention_dim, embed_dim,
            decoder_dim, vocab_size, encoder_dim, dropout)

    def forward(self, images, encoded_captions, caption_lengths):
        """
        :param images: [b, 3, h, w]
        :param encoded_captions: [b, max_len]
        :param caption_lengths: [b,]
        :return:
        """
        encoder_out = self.encoder(images)
        decoder_out = self.decoder(encoder_out, encoded_captions,
                                   caption_lengths.unsqueeze(1))
        return decoder_out

    def sample(self, images, startseq_idx, endseq_idx=-1, max_len=40,
               method='beam', return_alpha=False):
        encoder_out = self.encoder(images)
        return self.decoder.sample(encoder_out=encoder_out,
            startseq_idx=startseq_idx, endseq_idx=endseq_idx, max_len=max_len,
            method=method, return_alpha=return_alpha)
