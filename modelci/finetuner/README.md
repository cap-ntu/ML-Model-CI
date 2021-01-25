# Finetuner

\[Experimental\] Finetuner is a tiny module of ModelCI for easy fine-tuning pre-trained model on a new dataset. 

Finetuner **packs the training job** with specified Transfer Learning algorithm, dataset, and pre-trained model 
written on various ML framework (e.g. PyTorch). Such training jobs are **managed and schedules** by a coordinator. 
We are to support various training environment, such as local machine, private cluster, and public cloud platform. 

We leverage [PyTorch Lihgtning](https://github.com/PyTorchLightning/pytorch-lightning) as the training backend for
PyTorch models.

## Structure
```text
├── coordinator           # manages and schedules training jobs
├── torch_data_module.py  # creates data loader from dataset
├── tl_module.py          # defines transfer learning algorithm
└── trainer.py            # controls and abstract the training job for various framework, dataset, TL algorithm
```

## Capability

<table>
    <tbody>
        <tr style="text-align: center; vertical-align: bottom">
            <td>
            </td>
            <td>
                <b>Frameworks & Libraries</b>
            </td>
            <td>
                <b>TL Algorithm</b>
            </td>
            <td>
                <b>Dataset</b>
            </td>
            <td>
                <b>Training Environment</b>
            </td>
        </tr>
        <tr>
            <td style="text-align: center; vertical-align: middle">
                <b>Built-in</b>
            </td>
            <td>
                <ul>
                    <li>PyTorch with 
                        <a href="https://github.com/PyTorchLightning/pytorch-lightning">PyTorch Lightning</a>
                    </li>
                    <li> TensorFlow (TODO) </li>
                    <li> Keras (TODO) </li>
                </ul>
            </td>
            <td>
                <ul>
                    <li> Fine-tune </li>
                </ul>
            </td>
            <td>
                <ul>
                    <li><a href="https://pytorch.org/docs/stable/torchvision/datasets.html">Vision</a>
                        <ul>
                            <li>MNIST</li>
                            <li>CIFAR10</li>
                            <li>ImageNet</li>
                            <li>COCO</li>
                        </ul>
                    </li>
                </ul>
            </td>
            <td>
                <ul>
                    <li> Local Machine (CPU / Single GPU) </li>
                    <li> Private Cluster </li>
                    <li> Public Cluster </li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

## Tutorials

1. Fine tune a ResNet18
2. Fine tune a BERT-base-cased
