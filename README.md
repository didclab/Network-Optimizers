# Network-Optimizers
A repo on my work on optimizing networks.

The Flow and Working of the Network Optimizers for One Data Share - Vamshi Krishna Kyatham

Overview and Workflow:

1. Purpose: The repository is designed to optimize network transfers, likely for large datasets or files, using different optimization algorithms. It includes implementations for various optimization techniques, such as Bayesian optimization and reinforcement learning (e.g., DDPG, PPO).

2. Core Components:
    (i) Optimizers: The 'app/optimizers' directory contains different optimization algorithms, including implementations for DDPG (Deep Deterministic Policy Gradient), PPO (Proximal Policy Optimization), and Bayesian optimization.
    (ii) Models: The 'app/api/models.py' file defines data models using Pydantic, which are used for data validation and serialization. This includes models for transfer requests, configurations, and options.
    (iii) Database Interaction: The 'app/db' directory contains helper functions to interact with a database (InfluxDB) for monitoring and logging transfer metrics.
    (iv) Environment: The 'app/environments' directory defines the environment in which the optimization algorithms operate, likely simulating the transfer process and providing feedback to the algorithms.

3. API: The FastAPI framework is used to create an API for interacting with the optimization processes. The 'app/api/optimizer_routes.py' file defines routes for optimizing transfers, uploading/downloading models, and managing configurations.

4. Configuration Management: The repository includes functionality for managing configurations for different optimization runs, allowing users to specify parameters for the optimization algorithms.

5. Storage: The app/storage directory contains classes for managing model storage, either on the filesystem or in an S3 bucket, allowing for easy saving and loading of trained models.

Example Workflow I can think of:

1. Submit a Transfer Request: A user sends a POST request to the /optimize endpoint with a TransferJobRequest object, specifying the source, destination, and optimization options.

2. Run Optimization: The system processes the request, selecting the appropriate optimization algorithm based on the user's input. It runs the optimization in the background, adjusting parameters as needed based on feedback from the environment.

3. Monitor Performance: Throughout the optimization process, the system logs performance metrics, which can be queried later for analysis.

4. Retrieve Results: Once the optimization is complete, users can retrieve the results, including the optimized parameters and performance metrics.

Let's write the logic behind each file:

1. app/main.py

This is the entry point of the FastAPI application. It sets up the main FastAPI app and includes two routes: optimizer_api and config_router. The /api/ health is just for checking if API is alive. Funny that I am mentioning this as well.

2. app/api/optimizer_routes.py (IMP)

It defines the API routes for optimizer related operations. It includes endpoints for optimizing transfers, downloading/uploading optimizers, listing optimizers, and removing optimizers. The 'optimize_transfer' function is particularly important as it handles the creation and excecution of optimization tasks based on the transfer request and optimizer options from the request and config for that particular optimization model.

It receives a TransferJobRequest and processes it based on the optimizerRequestType:

For TRAIN, it creates a runner using RunnerFactory, loads the model, and starts training in the background (not implemented for many -> Basically Q-Learning or DQN)
For EVALUATE, it creates an EvaluateRunner, loads the model, and starts evaluation in the background.
For TUNE, it creates a TuneRunner, loads the model, and starts tuning in the background.

It also includes endpoints for downloading, uploading, listing, and removing optimizers.

3. app/api/config_routes.py:

It defines the API routes for configuration management. It includes endpoints for creating, retrieving, deleting, and listing configurations for different optimizer types.

4. app/api/models.py:

It contains Pydantic models (Better way to model the data as a class and best for serialisation and deseralisation) that define the structure of various data objects used throughout the application. It includes models for transfer requests, optimizer options, configurations for different optimizer types, and more.

5. app/optimizers/RunnerFactory.py:

This factory class is responsible for creating the appropriate runner (e.g., DdpgTrainRunner) based on the optimizer type specified in the configuration.

6. app/optimizers/TrainRunner.py:

This is an abstract base class for training runners. It defines the interface for training, saving, and loading models which is used by our RL based optimization algorithm models like DDPG and PPO in their respective directory structures.

7. app/optimizers/ddpg/ddpg_train_runner.py:

It implements the DDPG (Deep Deterministic Policy Gradient) training runner. It sets up the DDPG model, the environment (InfluxEnv passing the correct params), and handles the training process (Model from other repo?).

8. app/optimizers/bayesian/bayesian_optimizer.py:

It implements the Bayesian optimization algorithm. It uses the 'skopt' library for Gaussian Process optimization and includes methods for running the optimization process, graphing results, and managing the optimization lifecycle. (Get back post reading more on BO)

9. app/environemnts/ods_real_transfer_env.py:

It defines the InfluxEnv class, which is a custom Gym environment for the transfer optimization task. It interacts with InfluxDB to get transfer metrics and provides the interface for the reinforcement learning algorithms to interact with the transfer process. The step method is particularly important, as it applies the chosen action and returns the new state, reward, and other information back to InfluxDB with metadata of owner (Part of Learning about RL).

10. app/db/influx_db.py:

It contains the InfluxDb class, which handles interactions with the InfluxDB database. It includes methods for querying transfer data and job data.

11. app/db/db_helper.py:

It contains helper functions for interacting with the transfer service and monitoring service. It includes functions for submitting transfer requests, querying job status, and sending application parameters.

12. app/storage/OptimizerStore.py:

It defines the interface and implementations for storing and retrieving optimizer models. It includes both filesystem and S3 storage options.

13. app/storage/ConfigStore.py:

Similar to OptimizerStore, it defines the interface and implementations for storing and retrieving configuration data. It also includes both filesystem and S3 storage options.

14. app/storage/StorageFactory.py:

This factory class provides methods to get the appropriate storage implementation (filesystem or S3) for both configs and optimizers based on the environment configuration.

Connecting the dots from the logic in each file:

1. When a transfer optimization request comes in through the /optimize endpoint in optimizer_routes.py, the system uses the RunnerFactory to create the appropriate optimizer runner (e.g., DdpgTrainRunner).

2. The runner sets up the environment (InfluxEnv) which interacts with the actual transfer process through InfluxDB and the transfer service API (for sending application parameters).

3. The optimization process runs in the background, continuously querying the transfer metrics from InfluxDB and adjusting the transfer parameters (concurrency, parallelism) based on the chosen algorithm (DDPG, PPO, or Bayesian Optimization).

For training (DDPG example):
    The DdpgTrainRunner creates a DDPG model and starts the training process.
    In each step, the model chooses an action (concurrency and parallelism values).
    The InfluxEnv.step method applies this action by calling send_application_params_tuple from db_helper.py.
    It then queries InfluxDB for the resulting state an calculates the reward.
    This process repeats for the specified number of episodes.

4. The optimization results (models, configurations) are stored using the appropriate storage implementation (filesystem or S3) determined by the StorageFactory.

5. Throughout the process, the system interacts with the transfer service and monitoring service using the helper functions in db_helper.py to submit transfer requests, query job status, and send updated transfer parameters.

6. TODO (May be in future): The API provides endpoints for users to manage optimizers and configurations, allowing them to create, retrieve, list, and delete these resources.