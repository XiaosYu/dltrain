class DLTrainError(Exception):
    def __init__(self, message):
        super(DLTrainError, self).__init__(message)


def raise_error(title, message):
    raise DLTrainError(f"Raise error on {title}:"
                       f"{message}")


def check_length(variable_a, variable_b, variable_name_a, variable_name_b):
    if len(variable_a) != len(variable_b):
        raise_error(f"check length of {variable_name_a} and {variable_name_b}",
                    f"{variable_name_a} lengths({len(variable_a)}) is not equal to {variable_name_b} lengths({len(variable_b)}")


def try_convert(variable, convert_function, variable_name, target_type_name):
    try:
        variable = convert_function(variable)
        return variable
    except:
        raise_error(f"convert {variable_name} to {target_type_name}",
                    f"Could not convert {variable_name} type({type(variable)}) to {target_type_name}")