from typing import Dict, List

from pydantic.main import BaseModel


class Node(BaseModel):
    id: int
    label: str
    meta: Dict
    labelType: str


class Edge(BaseModel):
    source: int
    target: int


class Graph(BaseModel):
    nodes: List[Node]
    links: List[Edge]