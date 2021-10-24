from torch.optim import lr_scheduler


SGD_config = {
  'optimizer': 'SGD',
  'optim_hparas':{
      "lr": 0.1, 
      "momentum": 0.9,
      "nesterov": True,
      'weight_decay': 5e-4
  },
}

MultiStep_config = {
  'scheduler': 'MultiStepLR',
  'sched_hparas':{
      "milestones": [60, 120, 160],
      "gamma": 0.2
  }
}

SGD_config_lenet = {
  'optimizer': 'SGD',
  'optim_hparas':{
    "lr": 0.01, 
    "momentum": 0.9,
    "nesterov": True,
    # 'weight_decay': 5e-4
  },
}

Step_config = {
  'scheduler': 'StepLR',
  'sched_hparas':{
      'step_size': 5,
      'gamma': 0.9
  }
}


LA_config = {
  "la_steps": 5, 
  "la_alpha": 0.8,
  "pullback_momentum": "pullback"
}

SWA_config = {
  "SWA_lr": 0.05,
  "SWA_ratio":0.75
}
