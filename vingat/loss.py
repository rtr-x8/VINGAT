import torch


class BPRLoss(torch.nn.Module):
    def __init__(self, reg_lambda=0.01):
        super(BPRLoss, self).__init__()
        self.reg_lambda = reg_lambda  # 正則化パラメータ

    def forward(self, pos_scores, neg_scores, model_params):
        # pos_scores: 正のアイテムの予測スコア
        # neg_scores: 負のアイテムの予測スコア
        # model_params: モデルのパラメータ（正則化に使用）

        # BPRのペアワイズ損失計算
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

        # SCHGNのペアワイズ損失計算
        # bpr_loss = torch.log(torch.sigmoid(pos_scores - neg_scores))
        # bpr_loss = -torch.sum(bpr_loss)

        # L2正則化項
        reg_loss = 0
        for param in model_params:
            reg_loss += torch.norm(param, p=2)

        return loss + self.reg_lambda * reg_loss
