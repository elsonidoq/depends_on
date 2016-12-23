from depends_on import GraphDependency, depends_on


class SomeStep(GraphDependency): pass


class OtherStep(GraphDependency): pass


@depends_on(SomeStep)
@depends_on(OtherStep)
class SomeCompoundProcess(GraphDependency): pass


@depends_on(SomeCompoundProcess)
@depends_on(SomeStep)
class SomeMixedProcess(GraphDependency): pass


@depends_on(SomeMixedProcess)
@depends_on(SomeStep)
class AndTheFinalAlgorithmIs(GraphDependency): pass


for klass in AndTheFinalAlgorithmIs, SomeCompoundProcess, SomeMixedProcess:
    print "Dependencies for {}".format(klass)
    print GraphDependency.get_dependency_clausure(klass)
