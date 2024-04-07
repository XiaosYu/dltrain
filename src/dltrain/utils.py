__NAMEOF_COUNTER__ = 0


class LoggerContext:
    __TEXT_TEMPLATE__ = '[{0}] {1}\t|{2}'

    def __init__(self, callback=None):
        self._contents = {
            'datetime': [],
            'category': [],
            'content': []
        }

        self._callback = callback if callback is not None else lambda x: print(x)

        from datetime import datetime
        self._time_generator = datetime.now

    def log(self, time, category, content, *args, **kwargs):
        args_text = ','.join(args)
        kwargs_text = ','.join(map(lambda x: f'{x[0]}={x[1]}', kwargs.items()))

        if args_text != '':
            content += f'({args_text})'

        if kwargs_text != '':
            content += f'({kwargs_text})'

        text = LoggerContext.__TEXT_TEMPLATE__.format(time, category, content)
        self._callback(text)
        self._contents['datetime'].append(time)
        self._contents['category'].append(category)
        self._contents['content'].append(content)

    def error(self, exception):
        self.log(self._time_generator(), 'error', str(exception))

    def warn(self, content):
        self.log(self._time_generator(), 'warn', content)

    def info(self, content, *args, **kwargs):
        self.log(self._time_generator(), 'info', content, *args, **kwargs)

    def save(self, path, mode='txt'):
        import pandas as pd
        if mode == 'csv':
            frame = pd.DataFrame(data=self._contents, index=range(len(self._contents['content'])))
            frame.to_csv(path)
        elif mode == 'xlsx':
            frame = pd.DataFrame(data=self._contents, index=range(len(self._contents['content'])))
            frame.to_excel(path)
        else:
            with open(path, mode='w', encoding='utf-8') as f:
                for (time, category, content) in zip(self._contents['datetime'], self._contents['category'],
                                                     self._contents['content']):
                    text = LoggerContext.__TEXT_TEMPLATE__.format(time, category, content)
                    f.write(f'{text}\n')


def generate_runs_dir(root, prefix='exp'):
    import os, string, random
    if os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    dir_name = ''.join(random.choices(string.digits + string.ascii_letters, k=5))
    if os.path.exists(f'{root}/{prefix}{dir_name}'):
        generate_runs_dir(root)
    else:
        os.makedirs(f'{root}/{prefix}{dir_name}', exist_ok=True)
        return dir_name


def typeof(obj) -> type:
    if isinstance(obj, type):
        return obj
    else:
        return type(obj)


def nameof(*obj) -> str:
    import inspect, re
    global __NAMEOF_COUNTER__
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    frame_info = inspect.getframeinfo(caller_frame)
    source = frame_info.code_context[0]
    all_count = source.count('nameof')
    if all_count == 1:
        name = re.findall(r"nameof\((.*?)\)", source)[0]
        if name == '':
            __NAMEOF_COUNTER__ = 0
    else:
        name = re.findall(r"nameof\((.*?)\)", source)[__NAMEOF_COUNTER__]
        __NAMEOF_COUNTER__ += 1
        if __NAMEOF_COUNTER__ == all_count:
            __NAMEOF_COUNTER__ = 0

    if ',' in name:
        return name.split(',')
    else:
        return name


def to_object(obj: dict) -> object:
    class Namespace:
        def __init__(self, obj):
            self.__dict__.update(obj)

    return Namespace(obj)


def to_dict(obj: object) -> dict:
    return vars(obj)


def set_plt(font_sans_serif=None, axes_unicode_minus=False):
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        if font_sans_serif is None:
            font_sans_serif = ["kaiti"]
        plt.rcParams["font.sans-serif"] = font_sans_serif
        plt.rcParams["axes.unicode_minus"] = axes_unicode_minus
    except ModuleNotFoundError as e:
        pass


def set_seed(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ModuleNotFoundError as e:
        pass

    try:
        import sklearn
        sklearn.utils.check_random_state(seed)
    except ModuleNotFoundError as e:
        pass

    try:
        import numpy
        numpy.random.seed(seed)
    except ModuleNotFoundError as e:
        pass

    import random, os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def enumerable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def json_loads(data):
    import json
    return json.loads(data)


def json_dumps(data):
    import json
    is_numpy, is_torch = True, True
    try:
        import numpy as np
    except ModuleNotFoundError as e:
        is_numpy = False

    try:
        import torch
    except ModuleNotFoundError as e:
        is_torch = False

    def encode(obj):
        if is_numpy and isinstance(obj, np.ndarray):
            return obj.tolist()
        elif is_torch and isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        else:
            return obj

    return json.dumps(data, ensure_ascii=False, default=encode)
