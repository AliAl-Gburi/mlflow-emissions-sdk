from mlflow import MlflowClient
from codecarbon import EmissionsTracker

class EmissionsTrackerMlflow:
    experiment_tracking_params=None
    tracking_uri=""
    run_id=""
    emissions=0
    emissions_tracker = None
    
    def __init__(self):
        self.emissions_tracker = EmissionsTracker()
    
    def read_params(self, experiment_tracking_params):
        self.experiment_tracking_params = experiment_tracking_params

    def set_tracking_uri(self):
        self.tracking_uri = self.experiment_tracking_params['tracking_uri']
    
    



    def start_training_job(self):
        client = MlflowClient(tracking_uri=self.experiment_tracking_params['tracking_uri'])
        expm = client.get_experiment_by_name(self.experiment_tracking_params['experiment_name'])
        if expm:
            expm = dict(expm)
            exp_id = expm['experiment_id']
        else:
            exp_id = client.create_experiment(self.experiment_tracking_params['experiment_name'])
        run_id = dict(client.create_run(exp_id))
        print("runid", dict(run_id['info'])['run_id'])
        self.run_id = dict(run_id['info'])['run_id']
        self.emissions_tracker.start()
        
       
        

    def end_training_job(self, model):
        client = MlflowClient(tracking_uri=self.experiment_tracking_params['tracking_uri'])
        emissions = self.emissions_tracker.stop()
        client.log_metric(self.run_id, "emissions", emissions)
        self.emissions = emissions
        for k, v in dict(model.get_params()).items():
            client.log_param(self.run_id, k, v)

    def accuracy_per_emission(self, model, test_data):
        testlen = len(test_data)
        correctpred = 0
        for test_d, actual in test_data:
            predicted = model.predict(test_d)

            if predicted == test_d:
                correctpred += 1
        acc = correctpred/testlen
        client = MlflowClient(tracking_uri=self.experiment_tracking_params['tracking_uri'])
        client.log_metric(self.run_id, "acc_per_emission", acc/self.emissions)


        
       
        
        
            

