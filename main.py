# Standard package import
from functools import reduce
from time import time as now
from os.path import exists
from time import sleep

# Local application import
from PyGameFacade import *
from Config import initial_state
from Repository import *
from ContexManager import Visualization

global screen

def start_state() -> dict:
    global screen
    args = get_arguments()
    if not exists(args.project_name):
        initial_state["project_name"] = args.project_name
        initial_state["visualization"] = not args.visualization_off or args.visualization_on
        initial_state["train"] = not args.test or args.train
        if initial_state["visualization"] :
            screen = set_size(initial_state["environment"].size_screen)
        store_state(initial_state)
        return initial_state
    else:
        state = load_state(args.project_name)
        state = set_arg(state, "visualization", args.visualization_on, args.visualization_off)
        state = set_arg(state, "train", args.train, args.test)
        if not state["visualization"] and args.visualization_on:
            screen = set_size(state["environment"].size_screen)
        elif state["visualization"] and (not args.visualization_off or args.visualization_on):
            screen = set_size(state["environment"].size_screen)
        return state


def compute_time(state : dict) -> dict:
    time_now = now()
    time = state["time"]
    time["real"] += time_now - time["prev"]
    time["prev"] = time_now
    time["discret"] += 1
    if not state["train"]:
        sleep(time["delay"])
    return state


def events(state : dict) -> dict:
    for event in get_event():
        if event.type is QUIT:
            store_state(state)
            state["in_game"] = False
    return state


def update(state : dict) -> dict:
    state["environment"].next_state()
    return state


def draw(state : dict) -> dict:
    background(screen, BLACK)
    state["environment"].draw(screen)
    update_screen()
    return state


if __name__ == "__main__":
    state = start_state()
    with Visualization(state, [compute_time, update], [events, draw]) as steps:
        while state["in_game"]:
            state = reduce(lambda x, f : f(x), steps, state)
