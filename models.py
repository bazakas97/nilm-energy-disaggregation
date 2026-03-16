import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedSeq2PointCNN(nn.Module):
    """
    A deeper/more advanced CNN model for Seq2Point-style NILM.
    You can tweak channels, kernel sizes, etc. as desired.
    """

    def __init__(self, input_dim=1, output_dim=1, window_size=160, predict_onoff=False):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.predict_onoff = bool(predict_onoff)
        
        # Convs...
        self.conv1 = nn.Conv1d(in_channels=self.input_dim, out_channels=32, kernel_size=8)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5)
        
        self.dropout = nn.Dropout(0.2)
        
        # Do a mock forward to find out flatten dimension:
        with torch.no_grad():
            x_test = torch.zeros((1, self.window_size, self.input_dim))  # batch=1
            x_test = x_test.permute(0, 2, 1)  # => [1, 1, 160]
            x_test = self.conv1(x_test)
            x_test = self.conv2(x_test)
            x_test = self.conv3(x_test)
            x_test = self.conv4(x_test)
            x_test = self.conv5(x_test)
            out_shape = x_test.shape  # e.g. [1, 128, 132]
            flattened_size = out_shape[1] * out_shape[2]
        
        self.fc1 = nn.Linear(flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        # Keep `fc3` as power head for checkpoint compatibility.
        self.fc3 = nn.Linear(256, self.output_dim)
        self.onoff_head = nn.Linear(256, self.output_dim) if self.predict_onoff else None


    def forward(self, x):
        # x shape: [batch_size, window_size, input_dim]
        # We need [batch_size, input_dim, window_size] for Conv1d
        x = x.permute(0, 2, 1)  # => [batch_size, 1, window_size]

        x = F.relu(self.conv1(x))  # => [batch_size, 32, ...]
        x = F.relu(self.conv2(x))  # => [batch_size, 64, ...]
        x = F.relu(self.conv3(x))  # => [batch_size, 64, ...]
        x = F.relu(self.conv4(x))  # => [batch_size, 128, ...]
        x = F.relu(self.conv5(x))  # => [batch_size, 128, ...]

        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        power = self.fc3(x)
        if self.onoff_head is not None:
            onoff_logits = self.onoff_head(x)
            return {"power": power, "onoff_logits": onoff_logits}
        return power


class NILMTransformer(nn.Module):
    """
    Lightweight Transformer encoder for multi-device Seq2Point NILM.
    """

    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        window_size=160,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        predict_onoff=False,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.window_size = int(window_size)
        self.predict_onoff = bool(predict_onoff)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.window_size + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )
        self.onoff_head = None
        if self.predict_onoff:
            self.onoff_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, output_dim),
            )

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        # x: [batch, window_size, input_dim]
        batch_size = x.shape[0]
        h = self.input_proj(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos_embedding[:, : h.size(1), :]
        h = self.encoder(h)
        pooled = self.norm(h[:, 0, :])
        power = self.head(pooled)
        if self.onoff_head is not None:
            onoff_logits = self.onoff_head(pooled)
            return {"power": power, "onoff_logits": onoff_logits}
        return power


class NFResUnit(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, dilation=1, stride=1, bias=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                bias=bias,
                padding="same",
            ),
            nn.GELU(),
            nn.BatchNorm1d(c_out),
        )
        if c_in > 1 and c_in != c_out:
            self.match_residual = True
            self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1)
        else:
            self.match_residual = False

    def forward(self, x):
        if self.match_residual:
            return self.conv(x) + self.layers(x)
        return x + self.layers(x)


class NFDilatedBlock(nn.Module):
    def __init__(self, c_in=1, c_out=72, kernel_size=3, dilation_list=None, bias=True):
        super().__init__()
        if dilation_list is None:
            dilation_list = [1, 2, 4, 8]
        layers = []
        for i, dilation in enumerate(dilation_list):
            if i == 0:
                layers.append(NFResUnit(c_in, c_out, kernel_size=kernel_size, dilation=dilation, bias=bias))
            else:
                layers.append(NFResUnit(c_out, c_out, kernel_size=kernel_size, dilation=dilation, bias=bias))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NFDiagonalMaskedSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, head_dim, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x):
        batch, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(batch, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(batch, seqlen, self.n_heads, self.head_dim)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.einsum("blhe,bshe->bhls", xq, xk)
        diag_mask = torch.diag(torch.ones(seqlen, dtype=torch.bool, device=x.device)).view(1, 1, seqlen, seqlen)
        attn = self.attn_dropout(torch.softmax(scale * scores.masked_fill(diag_mask, float("-inf")), dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attn, xv)
        return self.out_dropout(self.wo(output.reshape(batch, seqlen, -1)))


class NFPositionWiseFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dp_rate=0.0):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim, bias=True)
        self.layer2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, x):
        return self.layer2(self.dropout(F.gelu(self.layer1(x))))


class NFEncoderLayer(nn.Module):
    def __init__(self, d_model=96, n_head=8, pffn_ratio=4, dropout=0.2, norm_eps=1e-5):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_head ({n_head})")

        self.attention_layer = NFDiagonalMaskedSelfAttention(
            dim=d_model,
            n_heads=n_head,
            head_dim=d_model // n_head,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.pffn = NFPositionWiseFeedForward(
            dim=d_model,
            hidden_dim=d_model * int(pffn_ratio),
            dp_rate=dropout,
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attention_layer(x)
        x = self.norm2(x)
        x = x + self.dropout(self.pffn(x))
        return x


class NILMFormerSeq2Seq(nn.Module):
    """
    Paper-style NILMFormer adaptation: sequence-to-sequence output.
    Input: [B, L, C], output: [B, L, output_dim].
    """

    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        window_size=160,
        predict_onoff=False,
        c_embedding=8,
        kernel_size=3,
        kernel_size_head=3,
        dilations=None,
        conv_bias=True,
        n_encoder_layers=3,
        d_model=96,
        dropout=0.2,
        pffn_ratio=4,
        nhead=8,
        norm_eps=1e-5,
    ):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4.")

        self.window_size = int(window_size)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.predict_onoff = bool(predict_onoff)
        self.c_embedding = int(c_embedding)
        self.d_model = int(d_model)

        d_model_embed = 3 * self.d_model // 4
        if dilations is None:
            dilations = [1, 2, 4, 8]

        self.embed_block = NFDilatedBlock(
            c_in=1,
            c_out=d_model_embed,
            kernel_size=int(kernel_size),
            dilation_list=list(dilations),
            bias=bool(conv_bias),
        )
        self.proj_embedding = nn.Conv1d(
            in_channels=self.c_embedding,
            out_channels=self.d_model // 4,
            kernel_size=1,
        )
        self.proj_stats1 = nn.Linear(2, self.d_model)
        self.proj_stats2 = nn.Linear(self.d_model, 2)

        encoder_layers = [
            NFEncoderLayer(
                d_model=self.d_model,
                n_head=int(nhead),
                pffn_ratio=int(pffn_ratio),
                dropout=float(dropout),
                norm_eps=float(norm_eps),
            )
            for _ in range(int(n_encoder_layers))
        ]
        encoder_layers.append(nn.LayerNorm(self.d_model))
        self.encoder_block = nn.Sequential(*encoder_layers)

        self.power_head = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.output_dim,
            kernel_size=int(kernel_size_head),
            padding=int(kernel_size_head) // 2,
            padding_mode="replicate",
        )
        self.onoff_head = None
        if self.predict_onoff:
            self.onoff_head = nn.Conv1d(
                in_channels=self.d_model,
                out_channels=self.output_dim,
                kernel_size=int(kernel_size_head),
                padding=int(kernel_size_head) // 2,
                padding_mode="replicate",
            )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _build_positional_channels(self, batch_size, seq_len, device, dtype):
        pos = torch.linspace(0.0, 1.0, seq_len, device=device, dtype=dtype).view(1, 1, seq_len)
        channels = []
        n_pairs = max(1, self.c_embedding // 2)
        for i in range(n_pairs):
            freq = 2.0 ** i
            channels.append(torch.sin(2.0 * torch.pi * freq * pos))
            channels.append(torch.cos(2.0 * torch.pi * freq * pos))
        encoding = torch.cat(channels, dim=1)[:, : self.c_embedding, :]
        return encoding.expand(batch_size, -1, -1)

    def forward(self, x):
        # x: [B, L, C]
        batch_size, seq_len, _ = x.shape
        mains = x[:, :, :1].permute(0, 2, 1)  # [B, 1, L]

        if self.input_dim > 1 and x.shape[-1] > 1:
            encoding = x[:, :, 1:].permute(0, 2, 1)  # [B, e, L]
            if encoding.shape[1] != self.c_embedding:
                if encoding.shape[1] > self.c_embedding:
                    encoding = encoding[:, : self.c_embedding, :]
                else:
                    pad = self.c_embedding - encoding.shape[1]
                    encoding = F.pad(encoding, (0, 0, 0, pad), mode="constant", value=0.0)
        else:
            encoding = self._build_positional_channels(batch_size, seq_len, mains.device, mains.dtype)

        inst_mean = torch.mean(mains, dim=-1, keepdim=True).detach()
        inst_std = torch.sqrt(torch.var(mains, dim=-1, keepdim=True, unbiased=False) + 1e-6).detach()
        mains_norm = (mains - inst_mean) / inst_std

        x_embed = self.embed_block(mains_norm)
        e_embed = self.proj_embedding(encoding)
        h = torch.cat([x_embed, e_embed], dim=1).permute(0, 2, 1)  # [B, L, d_model]

        stats_token = self.proj_stats1(torch.cat([inst_mean, inst_std], dim=1).permute(0, 2, 1))  # [B, 1, d_model]
        h = torch.cat([h, stats_token], dim=1)  # [B, L+1, d_model]
        h = self.encoder_block(h)
        h = h[:, :-1, :]  # [B, L, d_model]

        h_conv = h.permute(0, 2, 1)  # [B, d_model, L]
        power_seq = self.power_head(h_conv)

        stats_out = self.proj_stats2(stats_token)
        out_mean = stats_out[:, :, 0].unsqueeze(-1)
        out_std = stats_out[:, :, 1].unsqueeze(-1)
        power_seq = power_seq * out_std + out_mean  # [B, c_out, L]
        power_seq = power_seq.permute(0, 2, 1)  # [B, L, c_out]

        if self.onoff_head is not None:
            onoff_logits = self.onoff_head(h_conv).permute(0, 2, 1)
            return {"power": power_seq, "onoff_logits": onoff_logits}
        return power_seq


def build_model(model_config, output_dim, window_size):
    cfg = model_config or {}
    model_type = str(cfg.get("type", "cnn")).strip().lower()
    input_dim = int(cfg.get("input_dim", 1))
    predict_onoff = bool(cfg.get("predict_onoff", False))

    if model_type == "cnn":
        return AdvancedSeq2PointCNN(
            input_dim=input_dim,
            output_dim=output_dim,
            window_size=window_size,
            predict_onoff=predict_onoff,
        )

    if model_type in {"transformer", "nilmformer"}:
        return NILMTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            window_size=window_size,
            d_model=int(cfg.get("d_model", 128)),
            nhead=int(cfg.get("nhead", 8)),
            num_layers=int(cfg.get("num_layers", 4)),
            dim_feedforward=int(cfg.get("dim_feedforward", 256)),
            dropout=float(cfg.get("dropout", 0.1)),
            predict_onoff=predict_onoff,
        )

    if model_type in {"nilmformer_paper", "nilmformer_seq2seq"}:
        return NILMFormerSeq2Seq(
            input_dim=input_dim,
            output_dim=output_dim,
            window_size=window_size,
            predict_onoff=predict_onoff,
            c_embedding=int(cfg.get("c_embedding", 8)),
            kernel_size=int(cfg.get("kernel_size", 3)),
            kernel_size_head=int(cfg.get("kernel_size_head", 3)),
            dilations=list(cfg.get("dilations", [1, 2, 4, 8])),
            conv_bias=bool(cfg.get("conv_bias", True)),
            n_encoder_layers=int(cfg.get("n_encoder_layers", 3)),
            d_model=int(cfg.get("d_model", 96)),
            dropout=float(cfg.get("dropout", 0.2)),
            pffn_ratio=int(cfg.get("pffn_ratio", 4)),
            nhead=int(cfg.get("nhead", 8)),
            norm_eps=float(cfg.get("norm_eps", 1e-5)),
        )

    raise ValueError(f"Unsupported model type: {model_type}")
