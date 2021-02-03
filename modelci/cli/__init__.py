#  Copyright (c) NTU_CAP 2021. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import click
from modelci.cli.modelci_service import *
from modelci.cli import model_cli

@click.group()
@click.version_option()
def cli():
    """A complete MLOps platform for managing, converting and profiling models and then deploying models as cloud services (MLaaS)"""
    pass


cli.add_command(service)
cli.add_command(model_cli.commands)	
cli.add_command(model_cli.models)

if __name__ == '__main__':
    cli()

