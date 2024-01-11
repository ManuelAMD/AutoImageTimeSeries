import datetime
from typing import List
from app.common.arquitecture_factory import *
from app.dataset_files.dataset import *
from app.common.model_communication import *
import optuna
#Search algorithm for a TPE
from optuna.samplers import TPESampler
from app.common.repeat_pruner import *

class OptimizationStrategy(object):

    def __init__(self, model_architecture_factory: ModelArchitectureFactory, dataset: Dataset, parameters):
        self.model_architecture_factory: ModelArchitectureFactory = model_architecture_factory
        self.dataset: Dataset = dataset
        self.parameters = parameters
        #Optuna structure
        self.storage = optuna.storages.InMemoryStorage()
        self.main_study: optuna.Study = optuna.create_study(study_name=dataset.get_tag(),
                                                            storage=self.storage,
                                                            load_if_exists=True,
                                                            pruner=RepeatPruner(),
                                                            direction='minimize',
                                                            sampler=TPESampler(n_ei_candidates=5000, n_startup_trials=30))
        self.study_id = 0

        self.experiment_id = "{}-{}".format(dataset.get_tag(), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.phase: Phase = Phase.EXPLORATION
        self.search_space_type = self.model_architecture_factory.get_search_space().get_type()
        self.search_space_hash = self.model_architecture_factory.get_search_space().get_hash()
        print("Hash #{}".format(self.search_space_hash))
        self.exploration_trials = parameters['exploration_size']
        self.hall_of_fame_size = parameters['hall_of_fame_size']

        print("{}, {}, {}, {}, {}".format(self.exploration_trials, self.hall_of_fame_size, self.search_space_type, self.search_space_hash, self.experiment_id))
        
        #Listo for model comunication
        self.exploration_models_requests: List[ModelTrainingRequest] = list()
        self.exploration_models_completed: List[CompletedModel] = list()
        self.hall_of_fame: List[CompletedModel] = list()
        self.deep_training_models_requests: List[ModelTrainingRequest] = list()
        self.deep_training_models_completed: List[CompletedModel] = list()

    def recommend_model(self) -> ModelTrainingRequest:
        if self.phase == Phase.EXPLORATION:
            return self._recommend_model_exploration()
        elif self.phase == Phase.DEEP_TRAINING:
            return self._recommend_model_deep_training()
        
    
    def _recommend_model_exploration(self) -> ModelTrainingRequest:
        #Se crea un nuevo trial de optuna
        trial = self._create_new_trial()
        params = self.model_architecture_factory.generate_model_params(trial, self.dataset.get_input_shape())
        print("++ Generated trial {} ++".format(trial.number))
        if trial.should_prune():
            self._on_trial_pruned(trial)
            print("++ Prunned trial {} ++".format(trial.number))
            return self.recommend_model()
        model_training_request = ModelTrainingRequest(
            id = trial.number,
            training_type = self.parameters['arch_type'],
            experiment_id = self.experiment_id,
            architecture = params,
            epochs = self.parameters['exploration_epochs'],
            early_stopping_patience = self.parameters['exploration_early_stopping_patience'],
            is_partial_training = True,
            search_space_type = self.search_space_type.value,
            search_space_hash = self.search_space_hash,
            dataset_tag = self.dataset.get_tag()
        )
        self.exploration_models_requests.append(model_training_request)
        return model_training_request

    def _recommend_model_deep_training(self) -> ModelTrainingRequest:
        hof_model: CompletedModel = self.hall_of_fame.pop(0)
        model_training_request: ModelTrainingRequest = hof_model.model_training_request
        model_training_request.epochs = self.parameters['deep_training_epochs']
        model_training_request.early_stopping_patience = self.parameters['deep_training_early_stopping_patience']
        model_training_request.is_partial_training = False
        self.deep_training_models_requests.append(model_training_request)
        return model_training_request

    def _create_new_trial(self) -> optuna.Trial:
        trial_id = self.storage.create_new_trial(self.study_id)
        trial = optuna.Trial(self.main_study, trial_id)
        return trial
    
    def _on_trial_pruned(self, trial: optuna.Trial):
        self.storage.set_trial_state_values(trial.number, TrialState.PRUNED)


class Phase(Enum):
    EXPLORATION = 1
    DEEP_TRAINING = 2

class Action(Enum):
    GENERATE_MODEL = 1
    WAIT = 2
    START_NEW_PHASE = 3
    FINISH = 4