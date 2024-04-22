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
                                                            direction='maximize',
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
        
        #List for model comunication
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
            dataset_tag = self.dataset.get_tag(),
            horizon  = self.parameters['horizon'],
            window_size = params.window_size
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
    
    def should_generate(self) -> bool:
        print("** Should generate another model? **")
        if self.phase == Phase.EXPLORATION:
            return self._should_generate_exploration()
        elif self.phase == Phase.DEEP_TRAINING:
            return self._should_generate_hof()
        
    def _should_generate_exploration(self) -> bool:
        print("Generated exploration models {} / {}".format(len(self.exploration_models_requests), self.exploration_trials))
        should_generate = False
        pending_to_generate = self.exploration_trials - len(self.exploration_models_requests)
        if pending_to_generate > 0:
            should_generate = True
        return should_generate
    
    def _should_generate_hof(self) -> bool:
        print("Generated hall of fame models {} / {}".format(len(self.deep_training_models_requests), self.hall_of_fame_size))
        should_generate = False
        if len(self.deep_training_models_requests) < self.hall_of_fame_size:
            should_generate = True
        return should_generate
    
    def should_wait(self):
        print("** Should wait? **")
        if self.phase == Phase.EXPLORATION:
            return self._should_wait_exploration()
        elif self.phase == Phase.DEEP_TRAINING:
            return self._should_generate_hof()
        
    def _should_wait_exploration(self) -> bool:
        print("Recieved exploration models {} / {}".format(len(self.exploration_models_completed), self.exploration_trials))
        should_wait = True
        if len(self.exploration_models_requests) == len(self.exploration_models_completed):
            should_wait = False
        return should_wait
    
    def _should_wait_hof(self) -> bool:
        print("Recieved hall of fame models {} / {}".format(self.deep_training_models_requests, self.hall_of_fame_size))
        should_wait = True
        if len(self.deep_training_models_completed) == self.hall_of_fame_size:
            should_wait = False
        return should_wait
    
    def get_training_total(self) -> int:
        if self.phase == Phase.EXPLORATION:
            return self.exploration_trials
        elif self.phase == Phase.DEEP_TRAINING:
            return self.hall_of_fame_size
    
    def is_finished(self):
        return not self._should_wait_exploration() and not self._should_wait_hof()
    
    def _build_hall_of_fame_ITS(self):
        print("** Building hall of fame for ITS problem **")
        #If need to be the hihgest score, use reverse= True
        #stored_completed_models = sorted(self.exploration_models_completed, key= lambda completed_model: completed_model.performance)
        stored_completed_models = sorted(self.exploration_models_completed, key= lambda completed_model: completed_model.performance, reverse= True)
        self.hall_of_fame = stored_completed_models[0 : self.hall_of_fame_size]
        for model in self.hall_of_fame:
            print(model)
    
    def get_best_model(self):
        if self.search_space_type == SearchSpaceType.IMAGE_TIME_SERIES:
            return self.get_best_ITS_model()     

    def report_model_response(self, model_training_response: ModelTrainingResponse):
        print("** Trial {} reported a score of {} **".format(model_training_response.id, model_training_response.performance))
        if self.search_space_type == SearchSpaceType.IMAGE_TIME_SERIES:
            if self.phase == Phase.EXPLORATION:
                return self._report_model_response_exploration_ITS(model_training_response)
            elif self.phase == Phase.DEEP_TRAINING:
                return self._report_model_response_hof_ITS(model_training_response)
        
    def _report_model_response_exploration_ITS(self, model_training_response: ModelTrainingResponse):
        #Checar si el trial_id existe en el studio, pues truena si no.
        #performance = model_training_response.performance
        #if model_training_response.finished_epochs is True:
            #Se recompensa por terminar en epocas
        #    performance = performance * 1.03
        loss = model_training_response.performance
        self.storage.set_trial_state_values(model_training_response.id, TrialState.COMPLETE, [model_training_response.performance])
        self._register_completed_model(model_training_response)
        best_trial = self.get_best_exploration_ITS_model()
        print("Best exploration trial so far is #{} with a score of {}".format(best_trial.model_training_request.id, best_trial.performance))
        if self.should_generate():
            return Action.GENERATE_MODEL
        elif self.should_wait():
            return Action.WAIT
        elif not self._should_generate_exploration() and not self._should_wait_exploration():
            self._build_hall_of_fame_ITS()
            self.phase = Phase.DEEP_TRAINING
            return Action.START_NEW_PHASE
        

    def _report_model_response_hof_ITS(self, model_training_response: ModelTrainingResponse):
        print("** Recieved Deep training model response **")
        completed_model = next(
            model
            for model in self.exploration_models_completed
            if model.model_training_request.id == model_training_response.id
        )
        completed_model.performance_2 = model_training_response.performance
        self.deep_training_models_completed.append(completed_model)
        best_trial = self.get_best_ITS_model()
        print("Best HoF trial so far is #{} with a score of {}".format(best_trial.model_training_request.id, best_trial.performance_2))
        if self.should_generate():
            return Action.GENERATE_MODEL
        elif self.should_wait():
            return Action.WAIT
        elif (not self._should_generate_hof() and not self._should_wait_hof()):
            return Action.FINISH

    def _register_completed_model(self, model_training_response: ModelTrainingResponse):
        model_training_request = next(
            request
            for request in self.exploration_models_requests
            if request.id == model_training_response.id
        )
        performance = model_training_response.performance
        completed_model = CompletedModel(model_training_request, performance)
        self.exploration_models_completed.append(completed_model)

    def get_best_exploration_ITS_model(self):
        #best_model = min(self.exploration_models_completed, key= lambda completed_model: completed_model.performance)
        best_model = max(self.exploration_models_completed, key= lambda completed_model: completed_model.performance)
        return best_model

    def get_best_ITS_model(self):
        #best_model = min(self.deep_training_models_completed, key= lambda completed_model: completed_model.performance_2)
        best_model = max(self.deep_training_models_completed, key= lambda completed_model: completed_model.performance_2)
        return best_model

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