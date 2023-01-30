from multiprocessing.reduction import ForkingPickler, AbstractReducer

class ForkingPickler5(ForkingPickler):
    def __init__(self, *args):
        if len(args) > 1:
            args[1] = 2
        else:
            args.append(2)
        super().__init__(*args)

    @classmethod
    def dumps(cls, obj, protocol=5):
        return ForkingPickler.dumps(obj, protocol)


def dump(obj, file, protocol=5):
    ForkingPickler5(file, protocol).dump(obj)


class Pickle5Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler5
    register = ForkingPickler5.register
    dump = dump