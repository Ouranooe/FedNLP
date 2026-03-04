# Script to find FedML aggregator import path
import fedml.simulation.sp.fedavg as m
print("Module contents:", dir(m))

# Try different import paths
try:
    from fedml.simulation.sp.fedavg.fedavg_api import FedAvgAPI
    print("FedAvgAPI found in fedavg_api")
except ImportError as e:
    print(f"fedavg_api: {e}")

try:
    from fedml.simulation.sp.fedavg.my_model_trainer_classification import ModelTrainerCLS
    print("ModelTrainerCLS found")
except ImportError as e:
    print(f"my_model_trainer_classification: {e}")

# Check FedAvgAPI for aggregator
try:
    from fedml.simulation.sp.fedavg.fedavg_api import FedAvgAPI
    print("FedAvgAPI attributes:", [x for x in dir(FedAvgAPI) if 'aggregate' in x.lower()])
except:
    pass

# Search for aggregator module
import os
import fedml
fedml_path = os.path.dirname(fedml.__file__)
sp_path = os.path.join(fedml_path, 'simulation', 'sp', 'fedavg')
if os.path.exists(sp_path):
    print(f"Files in {sp_path}:")
    for f in os.listdir(sp_path):
        print(f"  {f}")
