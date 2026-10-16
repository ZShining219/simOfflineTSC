"""Parent-project registration shim; implementation lives in reproduction/tarl_tsc."""
from common.registry import Registry
from reproduction.tarl_tsc.parent_adapter import make_tarl_agent

for _method in ('sensor', 'gat', 'concat', 'selfattn', 'attention', 'gating',
                'crossq'):
    _cls = make_tarl_agent(_method)
    _cls = Registry.register_model(f'tarl_{_method}')(_cls)
    globals()[_cls.__name__] = _cls
