from typing import List, Optional

from rich import box
from rich.console import Console
from rich.table import Table

from modelci.types.bo import ModelBO, Framework, Engine, ModelVersion, IOShape, Weight, Status
from modelci.types.trtis_objects import ModelInputFormat

console = Console()


def single_model_view(model_bo: Optional[ModelBO], top=False):
    if model_bo is None:
        return ''

    framework_mapper = {
        Framework.PYTORCH: 'PyTorch',
        Framework.TENSORFLOW: 'TensorFlow',
    }

    engine_mapper = {
        Engine.TORCHSCRIPT: 'TorchScript',
        Engine.TFS: 'TensorflowServing',
        Engine.ONNX: 'ONNX',
        Engine.TRT: 'Triton',
        Engine.TVM: 'TVM',
        Engine.CUSTOMIZED: 'Customized',
        Engine.NONE: 'None',
    }

    status_mapper = {
        Status.PASS: '[green]✔[/green] Pass',
        Status.RUNNING: '[yellow]🚧[/yellow] Running',
        Status.FAIL: '[red]✖[/red] Fail',
        Status.UNKNOWN: '[grey]💔[/grey] Unknown'
    }

    if top:
        model_bo_name = f'[bold]{model_bo.name}[/bold]'
    else:
        model_bo_name = model_bo.name

    return (
        model_bo_name,
        framework_mapper[model_bo.framework],
        str(model_bo.version),
        model_bo.dataset,
        f'{model_bo.acc:.2f}',
        model_bo.task,
        engine_mapper[model_bo.engine],
        status_mapper[model_bo.status],
        '...'
    )


def model_view(model_bo_groups: List[List[ModelBO]], all=False):
    table = Table(box=box.SIMPLE)

    table.add_column('Model Name')
    table.add_column('Framework')
    table.add_column('Version')
    table.add_column('Pretrained Dataset')
    table.add_column('Accuracy')
    table.add_column('Task')
    table.add_column('Serving Engine')
    table.add_column('Status')
    table.add_column('Profiling Results')

    # flatten list
    model_bos_args = list()
    for model_bo_group in model_bo_groups:
        # build arguments for pass into `single_model_view`
        model_bo_group_args = list(map(lambda x: (x, False), model_bo_group))
        model_bo_group_args[0] = model_bo_group_args[0][0], True

        model_bos_args.extend(model_bo_group_args)
        # add group separator
        model_bos_args.append((None, False))

    # cut last separator
    if len(model_bos_args) > 0:
        model_bos_args.pop()

    # show all
    if all:
        for model_bo, top in model_bos_args:
            table.add_row(*single_model_view(model_bo, top))
    else:  # not show all
        if len(model_bos_args) <= 5:
            for model_bo, top in model_bos_args:
                table.add_row(*single_model_view(model_bo, top))
        else:
            # head
            for model_bo, top in model_bos_args[:4]:
                table.add_row(*single_model_view(model_bo, top))
            # middle
            table.add_row(*['...'] * len(table.columns))
            # tail
            table.add_row(*single_model_view(*model_bos_args[-1]))

    console.print(table)


if __name__ == '__main__':
    model = ModelBO(
        'ResNet50',
        framework=Framework.PYTORCH,
        engine=Engine.TRT,
        version=ModelVersion(1),
        dataset='ImageNet',
        acc=0.8,
        task='image classification',
        inputs=[IOShape([-1, 3, 224, 224], dtype=float, format=ModelInputFormat.FORMAT_NCHW)],
        outputs=[IOShape([-1, 1000], dtype=int)],
        weight=Weight(bytes([123])),
        status=Status.PASS,
    )

    model2 = ModelBO(
        'ResNet50',
        framework=Framework.PYTORCH,
        engine=Engine.TRT,
        version=ModelVersion(1),
        dataset='ImageNet',
        acc=0.8,
        task='image classification',
        inputs=[IOShape([-1, 3, 224, 224], dtype=float, format=ModelInputFormat.FORMAT_NCHW)],
        outputs=[IOShape([-1, 1000], dtype=int)],
        weight=Weight(bytes([123])),
        status=Status.RUNNING,
    )

    model3 = ModelBO(
        'ResNet50',
        framework=Framework.PYTORCH,
        engine=Engine.TRT,
        version=ModelVersion(1),
        dataset='ImageNet',
        acc=0.8,
        task='image classification',
        inputs=[IOShape([-1, 3, 224, 224], dtype=float, format=ModelInputFormat.FORMAT_NCHW)],
        outputs=[IOShape([-1, 1000], dtype=int)],
        weight=Weight(bytes([123])),
        status=Status.FAIL,
    )

    model4 = ModelBO(
        'ResNet50',
        framework=Framework.PYTORCH,
        engine=Engine.TRT,
        version=ModelVersion(1),
        dataset='ImageNet',
        acc=0.8,
        task='image classification',
        inputs=[IOShape([-1, 3, 224, 224], dtype=float, format=ModelInputFormat.FORMAT_NCHW)],
        outputs=[IOShape([-1, 1000], dtype=int)],
        weight=Weight(bytes([123])),
        status=Status.UNKNOWN,
    )

    model_view([[model], [model2, model3, model4], [model2, model3]], all=True)
