/* eslint-disable @typescript-eslint/camelcase */
export type ModelStructure = {
  layer: object;
  connection: object;
}

export interface IFinetuneConfigObject {
  model: string;
  data_module: object;
  min_epochs: number;
  max_epochs: number;
  optimizer_type: string;
  optimizer_property: object;
  lr_scheduler_type: string;
  lr_scheduler_property: object;
  loss_function: string;
}

export type FinetuneConfig = 
  | Partial<IFinetuneConfigObject>
  | IFinetuneConfigObject;

export const DEFAULT_FINETUNE_CONFIG: FinetuneConfig = {
  model: '',
  data_module: {dataset_name: 'CIFAR10', batch_size: 4},
  min_epochs: 10,
  max_epochs: 15,
  optimizer_type: 'Adam',
  optimizer_property: {
    betas: [0.9, 0.99],
    eps: 1e-08,
    weight_decay: 0,
    amsgrad: false
  },
  lr_scheduler_type: 'StepLR',
  lr_scheduler_property: {lr: 0.01, step_size: 30},
  loss_function: 'torch.nn.CrossEntropyLoss'
}


export const DEFAULT_CONFIG_SCHEMA = {
  'title': 'Finetune Settings',
  'type': 'object',
  'oneOf': [
    {
      'title': 'Select Dataset',
      'properties': {
        'dataset_name': {
          'type': 'string',
          default: 'CIFAR10',
          enum: ['CIFAR10']
        }
      }
    },
    { 'title': 'Upload Custom Dataset',
      'description': 'Please uploda compressed dataset in zip file format',
      'properties': {
        'dataset_path': {
          'type': 'string',
          'format': 'data-url',
          'title': 'Single File with'
        },
      }
    }
  ]
}