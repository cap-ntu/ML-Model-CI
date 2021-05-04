from typing import List
from modelci.types.models import Node, Edge, Graph


def get_node_shape(node):
    shape = None
    if node.type.tensor_type.shape and node.type.tensor_type.shape.dim:
        shape = [dim.dim_value for dim in node.type.tensor_type.shape.dim if dim.dim_value]
    return shape


# reference: https://github.com/onnx/onnxmltools/blob/master/onnxmltools/utils/visualize.py
def get_nodes(graph) -> List[Node]:
    initializer_names = [init.name for init in graph.initializer]
    weight_nodes = {node.name: node for node in graph.input}

    graph_nodes = []
    label_type = None
    for i, node in enumerate(graph.node, 0):
        shape = None
        name = node.name
        weight_input = list(filter(lambda input: ".weight" in input, node.input))
        if len(weight_input ):
            name = weight_input[0].replace(".weight", "")
            input_node = weight_nodes.get(weight_input[0])
            shape = get_node_shape(input_node)
            label_type = "html"
        graph_nodes.append(
            Node(id=i, label=node.op_type, meta={"shape": shape, "name": name}, labelType=label_type)
        )

    graph_nodes.extend(
        [
            Node(id=i, label=node.name, meta={"shape": get_node_shape(node), "name": node.name}, labelType="html")
            for i, node in enumerate(graph.input, len(graph_nodes))
            if node.name not in initializer_names
        ]
    )
    graph_nodes.extend(
        [
            Node(id=i, label=node.name, meta={"shape": get_node_shape(node), "name": node.name}, labelType="html")
            for i, node in enumerate(graph.output, len(graph_nodes) + 1)
        ]
    )
    return graph_nodes


# reference: https://github.com/onnx/onnxmltools/blob/master/onnxmltools/utils/visualize.py
def get_edges(graph) -> List[Edge]:
    initializer_names = [init.name for init in graph.initializer]
    nodes = [
        node
        for node in graph.node
        if node.name not in graph.initializer
    ]
    output_node_hash = {}
    edge_list = []
    for i, node in enumerate(nodes, 0):
        for output in node.output:
            if output in output_node_hash.keys():
                output_node_hash[output].append(i)
            else:
                output_node_hash[output] = [i]
    input_node = [
        node
        for node in graph.input
        if node.name not in initializer_names
    ]
    for i, inp in enumerate(input_node, len(nodes)):
        output_node_hash[inp.name] = [i]
    for i, node in enumerate(nodes, 0):
        for input in node.input:
            if input in output_node_hash.keys():
                edge_list.extend([Edge(source=node_id, target=i)
                                  for node_id in output_node_hash[input]])
    for i, output in enumerate(graph.output, len(nodes) + len(input_node) + 1):
        if output.name in output_node_hash.keys():
            edge_list.extend([Edge(source=node_id, target=i)
                              for node_id in output_node_hash[output.name]])
        else:
            pass
    return edge_list


def visualize_model(onnx_model):
    graph = onnx_model.graph
    nodes = get_nodes(graph)
    links = get_edges(graph)
    return Graph(nodes=nodes, links=links)
