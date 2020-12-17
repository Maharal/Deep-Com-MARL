# Standard import libraries
from pickle import load, dump
from argparse import ArgumentParser, Namespace
import os

# Local application import
from Config import arg_parser
from PyGameFacade import set_size



def load_state(project_name : str) -> dict:
    with open(f"{project_name}/state.s", "rb") as f:
        state = load(f)
        state["environment"].load(project_name)  
        return state



def store_state(state : dict):
    if not os.path.exists(state["project_name"]):
        os.makedirs(state["project_name"])
    state["environment"].store(state["project_name"])
    with open(f"{state['project_name']}/state.s", "wb" ) as f:
        dump(state, f)



def get_arguments() -> Namespace:
    parser = ArgumentParser(**arg_parser["description_info"])
    parser.add_argument(arg_parser["project_name_arg"]["name"], **arg_parser["project_name_arg"]["kwargs"])
    parser.add_argument(arg_parser["visualization_off_arg"]["name"], **arg_parser["visualization_off_arg"]["kwargs"])
    parser.add_argument(arg_parser["visualization_on_arg"]["name"], **arg_parser["visualization_on_arg"]["kwargs"])
    parser.add_argument(arg_parser["train_arg"]["name"], **arg_parser["train_arg"]["kwargs"])
    parser.add_argument(arg_parser["teste_arg"]["name"], **arg_parser["teste_arg"]["kwargs"])
    args = parser.parse_args()
    assert not args.visualization_off or not args.visualization_on 
    assert not args.train or not args.test 
    return args



def set_arg(state : dict, attr : str, f_true : bool, f_false :  bool) -> dict:
    if not state[attr] and f_true:
        state[attr] = True
    elif state[attr] and f_false:
        state[attr] = False
    return state