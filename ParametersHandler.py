import sys
from typing import List


class ParametersHandler:
    _instance = None

    name: str
    load: bool
    learning_rate: float
    input_size: int
    h_layers: List[int]
    output_size: int
    variable_learning_rate: bool
    epochs: int
    samples_per_epoch: int

    def __init__(self, args):
        valid_parameters = ParametersHandler.check_args(args)
        if not valid_parameters:
            print("Invalid parameters")
            sys.exit(1)
        ParametersHandler.convert_args(args)

    @staticmethod
    def check_args(args):
        try:
            ParametersHandler.load = ParametersHandler.str_to_bool(args[2])
        except ValueError:
            return False

        if not ParametersHandler.load:
            if not len(args) == 10:
                return False

        elif len(args) > 3:
            print('Load parameter set to "True", subsequent parameters overlooked!!!')

        return True

    @staticmethod
    def convert_args(args):
        ParametersHandler.name = args[1]

        try:
            ParametersHandler.learning_rate = float(args[3])
            ParametersHandler.input_size = int(args[4])
            ParametersHandler.h_layers = [int(x) for x in args[5].split(',')]
            ParametersHandler.output_size = int(args[6])
            ParametersHandler.variable_learning_rate = ParametersHandler.str_to_bool(args[7])
            ParametersHandler.epochs = int(args[8])
            ParametersHandler.samples_per_epoch = int(args[9])
            print("All parameters are converted successfully")

        except Exception as e:
            print("Error converting parameters: " + str(e))
            sys.exit(1)

    @staticmethod
    def str_to_bool(s: str) -> bool:
        if s.lower() in ('true', '1', 'yes'):
            return True
        elif s.lower() in ('false', '0', 'no'):
            return False
        else:
            raise ValueError(f"Cannot convert string '{s}' to bool")
