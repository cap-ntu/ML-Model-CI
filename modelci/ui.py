from datetime import datetime
from typing import List, Optional

import humanize
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()

status_mapper = {
    'Pass': '[green]✔[/green] Pass',
    'Running': '[yellow]🚧[/yellow] Running',
    'Fail': '[red]✖[/red] Fail',
    'Unknown': '[grey]💔[/grey] Unknown'
}

BOX_NO_BORDER = box.Box('\n'.join([' ' * 4] * 8))


def single_model_view(model: Optional[dict], top=False):
    if model is None:
        return ''

    if top:
        model_name = f'[bold]{model["name"]}[/bold]'
    else:
        model_name = model['name']

    return (
        model['id'],
        model_name,
        model['framework'],
        model['engine'],
        str(model['version']),
        model['dataset'],
        f'{model["acc"]:.2f}',
        model['task'],
        status_mapper[model['status']],
    )


def model_view(model_groups: List[List[dict]], quiet=False, list_all=False):
    if quiet:
        # print only ID of models
        model_ids = [model['id'] for model_group in model_groups for model in model_group]
        console.print(*model_ids, sep='\n')
        return

    table = Table(box=box.SIMPLE)

    table.add_column('ID')
    table.add_column('ARCH NAME')
    table.add_column('FRAMEWORK')
    table.add_column('ENGINE')
    table.add_column('VERSION')
    table.add_column('DATASET')
    table.add_column('ACCURACY')
    table.add_column('TASK', width=15)
    table.add_column('STATUS')

    # flatten list
    model_args = list()
    for model_group in model_groups:
        # build arguments for pass into `single_model_view`
        model_group_args = list(map(lambda x: (x, False), model_group))
        if len(model_group_args) > 0:
            model_group_args[0] = model_group_args[0][0], True

        model_args.extend(model_group_args)
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


def model_detailed_view(model: dict):
    def build_name_value_view(name, value):
        return f'[{dim_color}]{name}:[/{dim_color}] {value}'

    dim_color = 'grey62'

    grid = Table.grid(padding=(0, 2))
    # Basic Info
    grid.add_row('ID', 'Model Name', 'Framework', 'Version', 'Pretrained Dataset', 'Accuracy', 'Task',
                 style='b')
    grid.add_row(
        model['id'], model['name'], model['framework'], str(model['version']), model['dataset'], str(model['acc']),
        model['task']
    )

    converted_grid = Table.grid(padding=(0, 2))
    for i in range(5):
        converted_grid.add_column(style=dim_color, justify='right')
        converted_grid.add_column()
    # Converted model info
    create_time = datetime.strptime(model['create_time'], '%Y-%m-%dT%H:%M:%S.%f')
    time_delta = humanize.naturaltime(datetime.now() - create_time)
    converted_grid.add_row(Text('Converted Model Info', style='bold cyan3', justify='left'))
    converted_grid.add_row(
        'Serving Engine', model['engine'],
        'Status', status_mapper[model['status']],
        'Creator', model['creator'],
        'Created', time_delta,
    )

    first = True
    for input_ in model['inputs']:
        converted_grid.add_row(
            Text('Inputs', style=f'b {dim_color}') if first else '', '',
            'Name', input_['name'],
            'Shape', str(input_['shape']),
            'Data Type', input_['dtype'].split('_')[-1],
            'Format', input_['format'],
        )
        first = False
    first = True
    for output_ in model['outputs']:
        converted_grid.add_row(
            Text('Outputs', style=f'b {dim_color}') if first else '', '',
            'Name', output_['name'],
            'Shape', str(output_['shape']),
            'Data Type', output_['dtype'].split('_')[-1],
            'Format', output_['format'],
        )
        first = False
    converted_grid.add_row()

    # Profiling results
    converted_grid.add_row(Text('Profiling Results', style='bold cyan3', justify='left'))
    if not model['profile_result']:
        converted_grid.add_row('N.A.')
    else:
        spr = model['profile_result']['static_result']
        dprs = model['profile_result']['dynamic_results']

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
            converted_grid.add_row(f'Inference', f'{throughput:.3f} req/s')

    console.print(grid)
    console.print(converted_grid)
