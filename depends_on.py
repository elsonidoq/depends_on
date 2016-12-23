import inspect
import os

import networkx as nx

here = os.path.dirname(__file__)


def depends_on(thing):
    """
    Main decorator to tie things together
    :param thing: What you wanna tie with

    see pipeline.py or english_proficiency.py for examples
    """

    def decorator(cls):
        if cls not in GraphDependency.dependency_graph:
            GraphDependency.dependency_graph[cls] = []
        GraphDependency.dependency_graph[cls].append(thing)
        return cls

    return decorator


class GraphDependency(object):
    """
    Main class to both manipulate the graph and inherit to get cool graph powers
    """
    dependency_graph = {}

    @staticmethod
    def has_dependencies(klass):
        return len(GraphDependency.dependency_graph.get(klass, [])) == 0

    @staticmethod
    def get_dependency_graph():
        """
        Converts this guy to a networkx graph to easily perform crazy computations with this graph
        :return:
        """
        g = nx.DiGraph()
        for cls in get_subclasses(GraphDependency):
            g.add_node(cls)

        for src_obj, dst_objs in GraphDependency.dependency_graph.iteritems():
            src_objs = [src_obj]
            if inspect.isclass(src_obj): src_objs.extend(get_subclasses(src_obj))

            for sub_src_obj in src_objs:
                for dst_obj in dst_objs:
                    g.add_edge(sub_src_obj, dst_obj)
        return g

    @staticmethod
    def get_direct_dependencies(klass):
        return GraphDependency.dependency_graph[klass][:]

    @staticmethod
    def get_dependency_clausure(*classes_or_instances, **kwargs):
        classes = []
        for i, e in enumerate(classes_or_instances):
            if not inspect.isclass(e) and not inspect.isfunction(e):
                # instance object are turned into their corresponding types
                e = type(e)
            classes.append(e)

        g = GraphDependency.get_dependency_graph()
        res = nx.topological_sort(g, classes, reverse=True)
        include_requested = kwargs.get('include_requested', False)
        if not include_requested: res = res[:-1]
        return res


def get_subclasses(cls):
    stack = [cls]
    res = []
    while len(stack) > 0:
        direct_subclasses = stack.pop().__subclasses__()
        res.extend(direct_subclasses)
        stack.extend(direct_subclasses)
    return res
