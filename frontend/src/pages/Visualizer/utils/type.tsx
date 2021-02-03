/**
 * TODO replace object with interface(or use json schema)
 */
export interface FinetuneConfigObject {
    model: string
    data_module: object
    min_epochs: number
    max_epochs: number
    optimizer_type: string
    optimizer_property: object
    lr_scheduler_type: string
    lr_scheduler_property: object
    loss_function: string
  }
  
  export type FinetuneConfig = 
    | Partial<FinetuneConfigObject>
    | FinetuneConfigObject;
  
  export const DEFAULT_FINETUNE_CONFIG: FinetuneConfigObject = {
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
    loss_function: 'torch.nn.L1Loss'
  }

export interface ModelStructure{
    layer: object
    connection: object
}