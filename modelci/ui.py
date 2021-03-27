from datetime import datetime
from typing import List, Optional

import humanize
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from modelci.types.models import MLModel
from modelci.types.models.common import Status

console = Console()

status_mapper = {
    'Pass': '[green]✔[/green] Pass',
    'Running': '[yellow]🚧[/yellow] Running',
    'Fail': '[red]✖[/red] Fail',
    'Unknown': '[grey]💔[/grey] Unknown'
}

BOX_NO_BORDER = box.Box('\n'.join([' ' * 4] * 8))


def single_model_view(model: Optional[MLModel], top=False):
    if model is None:
        return ''

    if top:
        model_name = f'[bold]{model.architecture}[/bold]'
    else:
        model_name = model.architecture

    # TODO reorganize and format and the parameters
    return (
        str(model.id),
        model_name,
        model.framework.name,
        model.engine.name,
        str(model.version),
        model.dataset,
        list(model.metric.keys())[0].name,
        f'{list(model.metric.values())[0]:.2f}',
        model.task.name,
        status_mapper[model.status.name],
    )


def model_view(model_group: List[MLModel], quiet=False, list_all=False):
    if quiet:
        # print only ID of models
        model_ids = [str(model.id) for model in model_group]
        console.print(*model_ids, sep='\n')
        return

    table = Table(show_header=True, header_style="bold magenta")

    table.add_column('ID')
    table.add_column('ARCH NAME')
    table.add_column('FRAMEWORK')
    table.add_column('ENGINE')
    table.add_column('VERSION')
    table.add_column('DATASET')
    table.add_column('METRIC')
    table.add_column('SCORE')
    table.add_column('TASK', width=15)
    table.add_column('STATUS')

    # flatten list
    model_args = list()
    for index, model in enumerate(model_group):
        # build arguments for pass into `single_model_view`
        model_group_args = (model, False if index == 0 else True)
        model_args.append(model_group_args)
        # add group separator
        model_args.append((None, False))

    # cut last separator
    if len(model_args) > 0:
        model_args.pop()

    # show all
    if list_all:
        for model, top in model_args:
            table.add_row(*single_model_view(model, top))
    else:  # not show all
        if len(model_args) <= 5:
            for model, top in model_args:
                table.add_row(*single_model_view(model, top))
        else:
            # head
            for model, top in model_args[:4]:
                table.add_row(*single_model_view(model, top))
            # middle
            table.add_row(*['...'] * len(table.columns))
            # tail
            table.add_row(*single_model_view(*model_args[-1]))

    console.print(table)


def model_detailed_view(model: MLModel):
    # TODO: update the print fields
    dim_color = 'grey62'

    grid = Table.grid(padding=(0, 2))
    # Basic Info
    grid.add_row('ID', 'Architecture', 'Framework', 'Version', 'Pretrained Dataset', 'Metric', 'Score', 'Task',
                 style='b')
    grid.add_row(
        str(model.id),
        model.architecture,
        model.framework.name,
        str(model.version),
        model.dataset,
        list(model.metric.keys())[0].name,  # TODO: display multiple metrics
        str(list(model.metric.values())[0]),
        model.task.name.replace('_', ' ')
    )

    converted_grid = Table.grid(padding=(0, 2))
    for i in range(5):
        converted_grid.add_column(style=dim_color, justify='right')
        converted_grid.add_column()
    # Converted model info
    time_delta = humanize.naturaltime(datetime.now().astimezone() - model.create_time)
    converted_grid.add_row(Text('Converted Model Info', style='bold cyan3', justify='left'))
    converted_grid.add_row(
        'Serving Engine', model.engine.name,
        'Status', status_mapper[Status(model.status).name],
        'Creator', model.creator,
        'Created', time_delta,
    )

    first = True
    for input_ in model.inputs:
        converted_grid.add_row(
            Text('Inputs', style=f'b {dim_color}') if first else '', '',
            'Name', input_.name,
            'Shape', str(input_.shape),
            'Data Type', input_.dtype.name.split('_')[-1],
            'Format', input_.format.name,
        )
        first = False
    first = True
    for output_ in model.outputs:
        converted_grid.add_row(
            Text('Outputs', style=f'b {dim_color}') if first else '', '',
            'Name', output_.name,
            'Shape', str(output_.shape),
            'Data Type', output_.dtype.name.split('_')[-1],
            'Format', output_.format.name,
        )
        first = False
    converted_grid.add_row()

    # Profiling results
    converted_grid.add_row(Text('Profiling Results', style='bold cyan3', justify='left'))
    if not model.profile_result:
        converted_grid.add_row('N.A.')
    else:
        spr = model.profile_result['static_result']
        dprs = model.profile_result['dynamic_results']

        # Static profiling result
        converted_grid.add_row(Text('Static Result', style='bold turquoise2', justify='left'))
        if spr == 'N.A.':
            converted_grid.add_row(Text('N.A.', justify='left'))
        else:
            pass
        converted_grid.add_row()

        # Dynamic profiling results
        converted_grid.add_row(Text('Dynamic Results', style='bold turquoise2', justify='left'))
        for dpr in dprs:
            create_time = datetime.strptime(dpr['create_time'], '%Y-%m-%dT%H:%M:%S.%f')
            time_delta = humanize.naturaltime(datetime.now() - create_time)
            converted_grid.add_row(
                'Device Name', dpr['device_name'],
                'IP', dpr['ip'],
                'Device ID', dpr['device_id'],
                'Batch Size', str(dpr['batch']),
                'Created', humanize.naturaltime(time_delta),
            )

            memory = dpr['memory']
            converted_grid.add_row(Text('GPU', style='b', justify='left'))
            converted_grid.add_row(
                'Memory Used', f'{humanize.naturalsize(memory["memory_usage"], binary=True)} '
                               f'/ {humanize.naturalsize(memory["total_memory"], binary=True)}',
                'Util', f'{memory["utilization"] * 100:.1f} %'
            )
            latency = dpr['latency']['inference_latency']
            converted_grid.add_row(Text('Inference Latency', style='b', justify='left'))
            converted_grid.add_row(
                'Average', f'{latency["avg"]:.3f} ms',
                'P50', f'{latency["p50"]:.3f} ms',
                'P95', f'{latency["p95"]:.3f} ms',
                'P99', f'{latency["p99"]:.3f} ms',
            )

            converted_grid.add_row(Text('Throughput', style='b', justify='left'))
            throughput = dpr['throughput']['inference_throughput']
            converted_grid.add_row('Inference', f'{throughput:.3f} req/s')

    console.print(grid)
    console.print(converted_grid)
