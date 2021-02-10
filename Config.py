# Standard application import
from time import time as now
from datetime import datetime

# Local application import
from PyGameFacade import *
from Universe import Environment



# Configuration of object environment
NUM_LANDMARKS = 10
NUM_AGENTS = 10
SIZE_SCREEN = (512, 512)
BORDER = 80
CONSTRAIN_GEN = (BORDER, BORDER, SIZE_SCREEN[0]-BORDER, SIZE_SCREEN[1]-BORDER)


environment_param = {
    "agent_param" : {
        "num" : NUM_AGENTS,
        "setup" : {
            "diam" : 15, 
            "color": RED,
            "constrain" : CONSTRAIN_GEN,
        }
    },
    "landmark_param" : {
        "num" : NUM_LANDMARKS,
        "setup" : {
            "diam" : 50, 
            "color" : GREEN, 
            "constrain" : CONSTRAIN_GEN,
        }
    },
    "goal_param" : {
        "num" : NUM_LANDMARKS,
        "setup" : {
            "diam" : 70, 
            "color" : GRAY,
            "constrain" : CONSTRAIN_GEN,
            "stroke_weight" : 1,
        }
    },
    "size_screen" : SIZE_SCREEN,
    "dist_wall" : BORDER,
    "force_wall" : 6,
    "eta" : 0.1,
    "size_channel" : 6, 
    "num_steps" : 1000
}


# Initial state of game
initial_state = {
    "time" : {
        "prev" : now(), 
        "real" : 0.0, 
        "discret" : 0,
        "delay" : 0.1
    },
    "in_game" : True,
    "environment" : Environment(**environment_param)
}


# Definition of arg parser parameters
arg_parser = {
    "description_info" : {
        "description" : "This program is an implementation of Com-MARL to solve the game of coordination to push objects to some positions."
    },
    "project_name_arg" : {
        "name" : "--project-name", 
        "kwargs" : {
            "metavar" : "project-name", 
            "type" : str, 
            "nargs" : "?", 
            "help" : "Project name.", 
            "default" : "comarl"
        }
    },

    "visualization_off_arg" : {
        "name" : "--visualization-off", 
        "kwargs" : {
            "action" : "store_true",
            "help" : "Disable the render of simulation.", 
        }
    },

    "visualization_on_arg" : {
        "name" : "--visualization-on", 
        "kwargs" : {
            "action" : "store_true",
            "help" : "Enable the render of simulation.", 
        }
    },

    "train_arg" : {
        "name" : "--train", 
        "kwargs" : {
            "action" : "store_true",
            "help" : "Train model.", 
        }
    },    

    "teste_arg" : {
        "name" : "--test", 
        "kwargs" : {
            "action" : "store_true",
            "help" : "Test model.", 
        }
    }
}