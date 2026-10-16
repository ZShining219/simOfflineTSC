from common.registry import Registry


@Registry.register_model('base')
class BaseAgent(object):
    '''
    BaseAgent Class is mainly used for creating a base agent and base methods.
    '''
    def __init__(self, world):
        # revise if it is multi-agents in one model
        self.world = world
        self.sub_agents = 1
        # Scene collection is deliberately opt-in. Existing agents continue
        # to train without SGA, while a configured provider can expose one
        # immutable SceneContext for every control timestamp.
        self.scene_replay_enabled = False
        self._scene_context_provider = None

    def configure_scene_replay(self, provider):
        """Enable replay-side scene snapshots without enabling SGA.

        ``provider`` receives the synchronized simulation timestamp and must
        return ``utils.scene_context.SceneContext`` or ``SceneSnapshot``. The trainer calls it at
        both ends of each control interval; no trainable scene embedding is
        retained in replay.
        """
        if not callable(provider):
            raise TypeError('scene context provider must be callable')
        self._scene_context_provider = provider
        self.scene_replay_enabled = True
        return self

    def disable_scene_replay(self):
        self._scene_context_provider = None
        self.scene_replay_enabled = False

    def scene_context_at(self, observed_at):
        if not self.scene_replay_enabled or self._scene_context_provider is None:
            raise RuntimeError('Scene replay is not configured for this agent')
        context = self._scene_context_provider(observed_at)
        from utils.scene_context import SceneContext, SceneSnapshot
        if not isinstance(context, (SceneContext, SceneSnapshot)):
            raise TypeError('scene context provider must return SceneContext or SceneSnapshot')
        if abs(float(context.observed_at) - float(observed_at)) > 1e-8:
            raise ValueError('scene context timestamp does not match control step')
        return context

    def get_ob(self):
        raise NotImplementedError()

    def get_reward(self):
        raise NotImplementedError()

    def get_action(self, ob, phase):
        raise NotImplementedError()

    def get_action_prob(self, ob, phase):
        return None
