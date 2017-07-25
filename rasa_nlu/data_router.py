from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import logging
import os
import tempfile
import json
import multiprocessing

from builtins import object
from typing import Text

from rasa_nlu import utils
from rasa_nlu.agent import Agent
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import InvalidModelError
from rasa_nlu.train import do_train

logger = logging.getLogger(__name__)


class AlreadyTrainingError(Exception):
    """Raised when a training request is received for an Agent already being trained.

    Attributes:
        message -- explanation of why the request is invalid
    """

    def __init__(self):
        self.message = 'The agent is already being trained!'

    def __str__(self):
        return self.message


class DataRouter(object):
    DEFAULT_AGENT_NAME = "default"

    def __init__(self, config, component_builder):
        self.config = config
        self.responses = DataRouter._create_query_logger(config['response_log'])
        self._train_procs = []
        self.model_dir = config['path']
        self.token = config['token']
        self.emulator = self._create_emulator()
        self.component_builder = component_builder if component_builder else ComponentBuilder(use_cache=True)
        self.agent_store = self._create_agent_store()

    @staticmethod
    def _create_query_logger(response_log_dir):
        """Creates a logger that will persist incoming queries and their results."""

        # Ensures different log files for different processes in multi worker mode
        if response_log_dir:
            # We need to generate a unique file name, even in multiprocess environments
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            log_file_name = "rasa_nlu_log-{}-{}.log".format(timestamp, os.getpid())
            response_logfile = os.path.join(response_log_dir, log_file_name)
            # Instantiate a standard python logger, which we are going to use to log requests
            query_logger = logging.getLogger('query-logger')
            query_logger.setLevel(logging.INFO)
            utils.create_dir_for_file(response_logfile)
            ch = logging.FileHandler(response_logfile)
            ch.setFormatter(logging.Formatter('%(message)s'))
            # Prevents queries getting logged with parent logger --> might log them to stdout
            query_logger.propagate = False
            query_logger.addHandler(ch)
            logger.info("Logging requests to '{}'.".format(response_logfile))
            return query_logger
        else:
            # If the user didn't provide a logging directory, we wont log!
            logger.info("Logging of requests is disabled. (No 'request_log' directory configured)")
            return None

    def _remove_finished_procs(self):
        """Remove finished training processes from the list of running training processes."""
        self._train_procs = [p for p in self._train_procs if p.is_alive()]

    def _add_train_proc(self, p):
        """Adds a new training process to the list of running processes."""
        self._train_procs.append(p)

    @property
    def train_procs(self):
        """Instead of accessing the `_train_procs` property directly, this method will ensure that trainings that
        are finished will be removed from the list."""

        self._remove_finished_procs()
        return self._train_procs

    def train_proc_ids(self):
        """Returns the ids of the running trainings processes."""
        return [p.ident for p in self.train_procs]

    def _create_agent_store(self):
        agents = []

        if os.path.isdir(self.config['path']):
            agents = os.listdir(self.config['path'])

        agent_store = {}

        for agent in agents:
            agent_store[agent] = Agent(self.config, self.component_builder, agent)

        if not agent_store:
            agent_store[self.DEFAULT_AGENT_NAME] = Agent()
        return agent_store

    def _create_emulator(self):
        mode = self.config['emulate']
        if mode is None:
            from rasa_nlu.emulators import NoEmulator
            return NoEmulator()
        elif mode.lower() == 'wit':
            from rasa_nlu.emulators.wit import WitEmulator
            return WitEmulator()
        elif mode.lower() == 'luis':
            from rasa_nlu.emulators.luis import LUISEmulator
            return LUISEmulator()
        elif mode.lower() == 'api':
            from rasa_nlu.emulators.api import ApiEmulator
            return ApiEmulator()
        else:
            raise ValueError("unknown mode : {0}".format(mode))

    def extract(self, data):
        return self.emulator.normalise_request_json(data)

    def parse(self, data):
        agent = data.get("agent") or self.DEFAULT_AGENT_NAME
        model = data.get("model")

        if agent not in self.agent_store:
            agents = os.listdir(self.config['path'])
            if agent not in agents:
                raise InvalidModelError("No agent found with name '{}'.".format(agent))
            else:
                try:
                    self.agent_store[agent] = Agent(self.config, self.component_builder, agent)
                except Exception as e:
                    raise InvalidModelError("No agent found with name '{}'. Error: {}".format(agent, e))

        response = self.agent_store[agent].parse(data['text'], data.get('time', None), model)

        if self.responses:
            log = {"user_input": response, "model": agent, "time": datetime.datetime.now().isoformat()}
            self.responses.info(json.dumps(log, sort_keys=True))
        return self.format_response(response)

    def format_response(self, data):
        return self.emulator.normalise_response_json(data)

    def get_status(self):
        # This will only count the trainings started from this process, if run in multi worker mode, there might
        # be other trainings run in different processes we don't know about.
        num_trainings = len(self.train_procs)
        agents = {name: agent.as_dict() for name, agent in self.agent_store.items()}
        return {
            "available_agents": agents,
            "trainings_under_this_process": num_trainings,
            "training_process_ids": self.train_proc_ids()
        }

    def start_train_process(self, data, config_values):
        logger.info("Starting model training")
        f = tempfile.NamedTemporaryFile("w+", suffix="_training_data.json", delete=False)
        f.write(data)
        f.close()
        # TODO: fix config handling
        _config = self.config.as_dict()
        for key, val in config_values.items():
            _config[key] = val
        _config["data"] = f.name
        train_config = RasaNLUConfig(cmdline_args=_config)

        agent = _config.get("name")
        if not agent:
            raise InvalidModelError("No agent found with name '{}'".format(agent))
        if agent in self.agent_store:
            if self.agent_store[agent].status == 1:
                raise AlreadyTrainingError
            else:
                self.agent_store[agent].status = 1

        process = multiprocessing.Process(target=do_train, args=(train_config, self.component_builder))
        self._add_train_proc(process)
        process.start()
        logger.info("Training process {} started".format(process))
