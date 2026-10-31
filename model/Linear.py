import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.linear = nn.Linear(
            self.seq_len,
            self.pred_len,
        )

    def forecast(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
    ):
        # x_enc:
        # [B, L, N]

        x = x_enc.permute(0, 2, 1)

        # [B, N, pred_len]
        x = self.linear(x)

        # [B, pred_len, N]
        x = x.permute(0, 2, 1)

        return x

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None,
    ):
        return self.forecast(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
        )
