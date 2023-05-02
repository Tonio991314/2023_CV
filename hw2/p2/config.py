################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name   = 'adam_pre_da' # name of experiment

# Model Options
model_type = 'resnet18' # 'mynet' or 'resnet18'

# Learning Options
epochs     = 100           # train how many epochs
batch_size = 16           # batch size for dataloader 
use_adam   = True         # Adam or SGD optimizer
lr         = 1e-3         # learning rate
milestones = [16, 45, 64, 93] # reduce learning rate at 'milestones' epochs
