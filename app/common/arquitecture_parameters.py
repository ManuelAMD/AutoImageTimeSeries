from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin

@dataclass
class ModelArchitectureParameters(DataClassJsonMixin):

    @staticmethod
    def new():
        pass