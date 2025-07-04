%%writefile config.yaml

data_path: /kaggle/input/waveform-inversion
model: 
    name: UNet
    unet_params:
        init_features: 32
        depth: 5
read_weights: null
batch_size: 64
print_freq: 1000
max_epochs: 20
es_epochs: 4
seed: 100
valid_frac: 16
train_frac: 2
optimizer:
    lr: 0.0005
    weight_decay: 0.001
scheduler:
    params:
        factor: 0.316227766
        patience: 1
