from .hook import Hook
from semilearn.core.utils import EMA


class openSSL_EMAHook(Hook):
    """
    EMA model Hook for updating ema version of the model
    """

    def before_run(self, algorithm):
        algorithm.ema = EMA(algorithm.model['model_c'], algorithm.ema_m)
        algorithm.ema.register()

        if algorithm.resume is True:
            algorithm.ema.load(algorithm.ema_model)


        # if type(algorithm.model) is dict:
        #     algorithm.ema = {}
        #     for name in algorithm.model.keys():
        #         algorithm.ema[name] = EMA(algorithm.model[name], algorithm.ema_m)
        # else:
        #     algorithm.ema = EMA(algorithm.model, algorithm.ema_m)
        #     algorithm.ema.register()
        #
        # if algorithm.resume is True:
        #     if type(algorithm.model) is dict:
        #         for name in algorithm.model.keys():
        #             algorithm.ema[name].load(algorithm.ema_model[name])
        #     else:
        #         algorithm.ema.load(algorithm.ema_model)

    def after_train_step(self, algorithm):
        if algorithm.ema is not None:
            algorithm.ema.update()
            algorithm.ema_model.load_state_dict(algorithm.model['model_c'].state_dict())
            algorithm.ema_model.load_state_dict(algorithm.ema.shadow, strict=False)

        # if algorithm.ema is not None:
        #     if type(algorithm.model) is dict:
        #         for name in algorithm.model.keys():
        #             algorithm.ema[name].update()
        #             algorithm.ema_model[name].load_state_dict(algorithm.model[name].state_dict())
        #             algorithm.ema_model[name].load_state_dict(algorithm.ema[name].shadow, strict=False)
        #     else:
        #         algorithm.ema.update()
        #         algorithm.ema_model.load_state_dict(algorithm.model.state_dict())
        #         algorithm.ema_model.load_state_dict(algorithm.ema.shadow, strict=False)
