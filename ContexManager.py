# Local Application import
from PyGameFacade import init
from Repository import store_state
import traceback


class Visualization(object):
    def __init__(self, state : dict, pure_steps : list = [], side_effect_steps : list = []):
        self.steps = pure_steps + (side_effect_steps if state["visualization"] else [])
        self.state = state
    
    def __enter__(self):
        if self.state["visualization"] : init()
        return self.steps

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is KeyboardInterrupt: 
            store_state(self.state)
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_traceback)
        if self.state["visualization"]: 
            quit()