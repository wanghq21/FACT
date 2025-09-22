# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)



import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiVariableACF_Loss(nn.Module):
    def __init__(self, max_lag=10, eps=1e-8):
        """
        自定义损失函数：计算多变量时间序列预测值与真实值的 ACF 差异
        :param max_lag: 最大滞后阶数
        :param eps: 防止除零的小值
        """
        super(MultiVariableACF_Loss, self).__init__()
        self.max_lag = max_lag
        self.eps = eps

    def forward(self, y_true, y_pred):
        """
        :param y_true: 真实值张量 (batch_size, sequence_length, variable_number)
        :param y_pred: 预测值张量 (batch_size, sequence_length, variable_number)
        :return: ACF 差异的均方误差 (MSE)
        """
        batch_size, seq_len, var_num = y_true.shape

        # 确保输入为浮点类型
        y_true = y_true.float()
        y_pred = y_pred.float()

        def compute_acf(series):
            """
            计算多变量时间序列的 ACF
            :param series: 形状为 (batch_size, sequence_length, variable_number)
            :return: 形状为 (batch_size, variable_number, max_lag) 的 ACF 张量
            """
            batch_size, seq_len, var_num = series.shape
            device = series.device  # 关键：获取当前张量的设备
            acf_values = []
            for lag in range(1, self.max_lag + 1):
                if lag >= seq_len:
                    acf_values.append(torch.zeros(batch_size, var_num, device=device))  # 指定设备
                    continue

                # 提取滞后 lag 的部分
                x = series[:, lag:, :]  # shape: (batch_size, seq_len - lag, var_num)
                y = series[:, :-lag, :]  # shape: (batch_size, seq_len - lag, var_num)

                # 去均值化
                x_mean = torch.mean(x, dim=1, keepdim=True)  # shape: (batch_size, 1, var_num)
                y_mean = torch.mean(y, dim=1, keepdim=True)  # shape: (batch_size, 1, var_num)

                x_centered = x - x_mean
                y_centered = y - y_mean

                # 协方差和方差
                cov = torch.mean(x_centered * y_centered, dim=1)  # shape: (batch_size, var_num)
                var = torch.mean(x_centered ** 2, dim=1)  # shape: (batch_size, var_num)

                # 自相关系数
                acf = cov / (var + self.eps)  # 添加小值防止除零
                acf_values.append(acf)

            return torch.stack(acf_values, dim=1)  # shape: (batch_size, max_lag, var_num)

        # 计算真实值和预测值的 ACF
        acf_true = compute_acf(y_true)  # shape: (batch_size, max_lag, var_num)
        acf_pred = compute_acf(y_pred)  # shape: (batch_size, max_lag, var_num)

        # 调整维度以便计算 MSE
        # 将 (batch_size, max_lag, var_num) 展平为 (batch_size * max_lag, var_num)
        acf_true_flat = acf_true.view(-1, var_num)
        acf_pred_flat = acf_pred.view(-1, var_num)

        # 计算 ACF 差异的均方误差 (MSE)
        loss = F.mse_loss(acf_true_flat, acf_pred_flat, reduction='mean')
        return loss