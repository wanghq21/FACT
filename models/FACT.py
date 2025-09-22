import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.nn.init as init

class Inception2d(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, dilation=1, groups=1, padding_mode='zeros', init_weight=True):
        super(Inception2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.dilation = dilation
        self.kernels = nn.ModuleList([
                            nn.Conv2d(in_channels, out_channels, kernel_size=2 * (i+1) + 1, 
                                dilation=dilation, padding=dilation * (i+1), 
                                padding_mode=padding_mode, groups=groups) 
                            for i in range(self.num_kernels-1)])
                             
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                dilation=1, padding=0, 
                                padding_mode=padding_mode, groups=groups)
       
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        tmp = self.conv1(x)
        for i in range(self.num_kernels-1):
            tmp = tmp + self.kernels[i](x) 
        return tmp/self.num_kernels



class Conv_block2d(nn.Module):
    def __init__(self, configs, dilation=3, d_model=512, d_ff=2048, head=1):
        super(Conv_block2d, self).__init__()
        if configs.freq == 't':
            self.new_channel = configs.enc_in+5 
        elif configs.freq == 'h':
            self.new_channel = configs.enc_in+4
        elif configs.freq == 'd':
            self.new_channel = configs.enc_in+3
        else:
            self.new_channel = configs.enc_in
        self.channel_patch = int(math.sqrt(self.new_channel)) +1
        self.channel_patch2 = self.channel_patch
        
        self.d_model = d_model
        self.head = head
        self.dilation = dilation
        self.num_kernel = configs.num_kernels

        self.conv = nn.ModuleList([  nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            Inception2d(self.d_model, self.d_model, num_kernels=self.num_kernel, 
                dilation=d+1, padding_mode='zeros', groups=self.d_model),
            )     for d in range(self.dilation)])


        self.conv_ffn = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            torch.nn.Conv2d(in_channels=self.d_model, out_channels=configs.d_ff,
                    kernel_size=1,stride=1,padding=0,groups=1),
            torch.nn.GELU( ),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Conv2d(in_channels=configs.d_ff, out_channels=self.d_model,
                    kernel_size=1,stride=1,padding=0,groups=1) 
            )   


    def forward(self, x):
        batch, seq, channel = x.shape
        x = torch.cat((x, x[:,:,:(self.channel_patch*self.channel_patch2-channel)]),dim=-1)
        x = x.reshape(batch, seq, self.channel_patch, self.channel_patch2)

        time =  x
        for i in range(self.dilation):
            time = time + self.conv[i](x)
        x = time

        x = x + self.conv_ffn(x)
        x = x.reshape(batch, seq, -1)[:, :, :channel]
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.use_norm = configs.use_norm
        self.freq = configs.freq
        self.d_model = configs.d_model
        
        self.dilation = configs.dilation
        self.layer = len(self.dilation)

        self.num_kernel = configs.num_kernels

        self.emb = torch.nn.Conv1d(in_channels=self.seq_len, out_channels=configs.d_model,
                    kernel_size=1,stride=1,padding=0,groups=1)

        self.alpha = configs.core    # weight factor for time and freq modeling

        if self.alpha == 1.0:
            self.projection = torch.nn.Conv1d(in_channels=configs.d_model   , out_channels=configs.pred_len,
                        kernel_size=1,stride=1,padding=0,groups=1)
            self.conv_time = nn.ModuleList([Conv_block2d(configs, dilation=self.dilation[i], 
                d_model=self.d_model, d_ff=configs.d_ff, head=8)  for i in range(self.layer)])
        elif self.alpha == 0.0:
            self.projection = torch.nn.Conv1d(in_channels=configs.d_model, out_channels=configs.pred_len,
                        kernel_size=1,stride=1,padding=0,groups=1)
            self.conv_amp = nn.ModuleList([Conv_block2d(configs, dilation=self.dilation[i], 
                    d_model=self.d_model//2+1, d_ff=configs.d_ff, head=1)  for i in range(self.layer)])
            self.conv_phase = nn.ModuleList([Conv_block2d(configs, dilation=self.dilation[i], 
                    d_model=self.d_model//2+1, d_ff=configs.d_ff, head=1)  for i in range(self.layer)])
        else:
            self.projection = torch.nn.Conv1d(in_channels=configs.d_model  , out_channels=configs.pred_len,
                        kernel_size=1,stride=1,padding=0,groups=1)
            self.conv_time = nn.ModuleList([Conv_block2d(configs, dilation=self.dilation[i], 
                d_model=self.d_model, d_ff=configs.d_ff, head=8)  for i in range(self.layer)])
            self.conv_amp = nn.ModuleList([Conv_block2d(configs, dilation=self.dilation[i], 
                    d_model=self.d_model//2+1, d_ff=configs.d_ff, head=1)  for i in range(self.layer)])
            self.conv_phase = nn.ModuleList([Conv_block2d(configs, dilation=self.dilation[i], 
                    d_model=self.d_model//2+1, d_ff=configs.d_ff, head=1)  for i in range(self.layer)])


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch, seq, channel = x_enc.shape

        if self.freq == 't' or self.freq == 'h' or self.freq == 'd':
            x_enc = torch.cat((x_enc, x_mark_enc), dim=-1)
        
        batch, seq, channel = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        x_enc = self.emb(x_enc)   # (batch, seq_len, d_model)

        if self.alpha == 1.0:
            for i in range(self.layer):
                x_enc = self.conv_time[i](x_enc)
            enc_out = self.projection(x_enc)
        elif self.alpha == 1.0:
            f = torch.fft.rfft(x_enc, dim=1)
            amplitude = torch.abs(f) 
            phase = torch.angle(f) 
            for i in range(self.layer):
                amplitude = self.conv_amp[i](amplitude)
                phase = self.conv_phase[i](phase)
            phase = phase - 2 * torch.pi * torch.round(phase / (2 * torch.pi))
            real = amplitude * torch.cos(phase)
            imag = amplitude * torch.sin(phase)
            freq = torch.complex(real, imag)
            enc_out = torch.fft.irfft(freq, dim=1).to(torch.float32) 
            enc_out = self.projection(enc_out)
        else:
            f = torch.fft.rfft(x_enc, dim=1)
            amplitude = torch.abs(f) 
            phase = torch.angle(f) 
            for i in range(self.layer):
                amplitude = self.conv_amp[i](amplitude)
                phase = self.conv_phase[i](phase)
            phase = phase - 2 * torch.pi * torch.round(phase / (2 * torch.pi))
            real = amplitude * torch.cos(phase)
            imag = amplitude * torch.sin(phase)
            freq = torch.complex(real, imag)
            freq = torch.fft.irfft(freq, dim=1).to(torch.float32) 

            time = x_enc
            for i in range(self.layer):
                time = self.conv_time[i](time)

            enc_out = self.projection( self.alpha * time + (1-self.alpha)  * freq )  

        if self.use_norm:
            enc_out = enc_out * stdev + means

        return enc_out[:,:,:self.enc_in] 

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        _, L, N = x_enc.shape


        x_enc = self.emb(x_enc)

        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
            # x_enc = self.model2[i](x_enc)

        enc_out = self.projection((x_enc).transpose(1, 2)).transpose(1, 2)
        # enc_out = x_enc
        
        enc_out = enc_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len  , 1))
        enc_out = enc_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len  , 1))

        return enc_out


    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, L, N = x_enc.shape

        # Embedding
        x_enc = self.emb(x_enc)

        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
            # x_enc = self.model2[i](x_enc)

        dec_out = self.projection((x_enc).transpose(1, 2)).transpose(1, 2)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

