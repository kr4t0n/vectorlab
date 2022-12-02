import torch


class Seq2Seq(torch.nn.Module):
    r"""A sequence to sequence based model.

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder used to encode inputs to latent space.
    decoder : torch.nn.Module
        The decoder used to reconstruct inputs from latent space.
    encoder_embedding : torch.nn.Module
        The encoder embedding to embed inputs of encoder.
    decoder_embedding : torch.nn.Module, optional
        The decoder embedding to embed inputs of decoder.
    start_token : int, optional
        The initialized first input for decoder.
    end_token : int, optional
        The end symbol to stop decoder inference.
    teacher_forcing : float, optional
        The probability to feed actual output instead of inferred output.

    Attributes
    ----------
    encoder_ : torch.nn.Module
        The user defined encoder.
    decoder_ : torch.nn.Module
        The user defined decoder.
    encoder_embedding_ : torch.nn.Module
        The user defined encoder embedding.
    decoder_embedding_ : torch.nn.Module, optional
        The user defined decoder embedding.
    start_token_ : int, optional
        The initialized first input for decoder.
    end_token_ : int, optional
        The end symbol to stop decoder inference.
    teacher_forcing_ : float, optional
        The probability to feed actual output instead of inferred output.
    """

    def __init__(self, encoder, decoder,
                 encoder_embedding, decoder_embedding=None,
                 start_token=None, end_token=None,
                 teacher_forcing=.5):

        super().__init__()

        self.encoder_ = encoder
        self.decoder_ = decoder

        self.encoder_embedding_ = encoder_embedding

        if decoder_embedding is None:
            self.decoder_embedding_ = self.encoder_embedding_
        else:
            self.decoder_embedding_ = decoder_embedding

        self.start_token_ = start_token
        self.end_token_ = end_token

        self.teacher_forcing_ = teacher_forcing

        return

    def _fetch_input(self, y, t):
        """Fetch corresponding output in specified time step.

        Parameters
        ----------
        y : tensor
            The actual output samples.
        t : int
            The specified time step.

        Returns
        -------
        tensor
            The actual output sample in specified time step.
        """

        if self.decoder_.decoder_.batch_first:
            y = y[:, t:t + 1]
        else:
            y = y[t:t + 1]

        return y

    def _fill_y_hat(self, y_hat, t, o_prob):
        r"""Fill y_hat with output probabilities in specified time step.

        Parameters
        ----------
        y_hat : tenosr
            The output samples.
        t : int
            The specified time step.
        o_prob : tensor
            The output sample in specified time step.
        """

        if self.decoder_.decoder_.batch_first:
            y_hat[:, t:t + 1, :] = o_prob
        else:
            y_hat[t:t + 1] = o_prob

        return

    def forward(self, x, y):
        r"""The forward process to obtain output samples.

        Parameters
        ----------
        x : tensor
            The input samples.
        y : tensor
            The actual output samples

        Returns
        -------
        tensor
            The inferred output samples.
        """

        if self.decoder_.decoder_.batch_first:
            n_steps = y.shape[1]
        else:
            n_steps = y.shape[0]

        y_hat = torch.zeros(
            y.shape[0], y.shape[1], self.decoder_.out_dims_,
            device=y.device
        )

        # forward encoder process
        emb_x = self.encoder_embedding_(x)
        h = self.encoder_(emb_x)

        # forward decoder process
        use_teacher_forcing = torch.rand([]) < self.teacher_forcing_

        # start token
        start = \
            torch.empty_like(self._fetch_input(y, 0)).fill_(self.start_token_)

        for t in range(n_steps):

            if t == 0:
                y_ = start
            else:
                if use_teacher_forcing:
                    y_ = self._fetch_input(y, t - 1)
                else:
                    y_ = o

            emb_y = self.decoder_embedding_(y_)
            o, h = self.decoder_(emb_y, h)
            o_prob = torch.nn.functional.log_softmax(o, dim=-1)

            # notice that tensor is a reference, therefore we can
            # directly change on it
            self._fill_y_hat(y_hat, t, o_prob)

            # choose top k candidate
            # the shape o_prob should be
            #  batch_first = True : (batch_size, 1, out_dims)
            #  batch_first = False: (1, batch_size, out_dims)
            _, o = o_prob.topk(1, dim=-1)

            # the shape o should be
            #  batch_first = True : (batch_size, 1, 1)
            #  batch_first = False: (1, batch_size, 1)
            # remove the last dimension and detach
            o = o.squeeze(2).detach()

        return y_hat

    def forward_latent(self, x, y):
        r"""The forward process to obtain latent samples.

        Parameters
        ----------
        x : tensor
            The input samples.
        y : tensor, ignore
            The actual output samples

        Returns
        -------
        tensor
            The latent samples.
        """

        if hasattr(self.encoder_, 'forward_latent'):
            z = self.encoder_.forward_latent(x)
        else:
            z = self.encoder_(x)

        return z

    def reset_parameters(self):
        r"""Reset the parameters inside.
        """

        self.encoder_.reset_parameters()
        self.decoder_.reset_parameters()

        self.encoder_embedding_.reset_parameters()
        self.decoder_embedding_.reset_parameters()

        return
