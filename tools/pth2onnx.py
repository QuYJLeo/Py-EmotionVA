#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

device = torch.device("cpu")

# input = torch.rand(1, 3, 224, 224)  # 初始化输入
input = torch.rand(1, 3, 112, 112)  # 初始化输入
input = input.to(device)


from models.DDAM import DDAMNet
model = DDAMNet(num_class=8, num_head=2, pretrained=False)

checkpoint = torch.load("./checkpoints/mp_MFN_epoch24.pth", weights_only= False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 把模型设置成推理模式



output = model(input)  # 推理
print("Beginning to export onnx ...")
torch.onnx.export(model,  # 要转换的模型
                  input,  # 输入
                  "./checkpoints/mp_MFN_epoch24.onnx", # 到处的模型，以onnx格式保存
                  export_params=True,  # 训练参数是否和模型一同导出，一般设置为true
                  # opset_version=12,  # 导出onnx模型的版本
                  verbose=False,  # 是否打印参数
                  input_names=["input"],  # 输如名称
                  output_names=["output"],  # 输出名称
                  dynamic_axes={
                                "input": {0: "batch_size"},
                                "output": {0: "batch_size"},
                                }  # 把输入的批量维度设置为动态
                  )
print("Exporting onnx successfully!")






