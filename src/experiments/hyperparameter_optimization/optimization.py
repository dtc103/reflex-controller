import optuna
import subprocess
import sys
import os
import time

LOCK_NAME = "gpu_startup.lock"

class FileLock:
    def __init__(self, lock_file="startup.lock", timeout=300):
        self.lock_file = lock_file
        self.timeout = timeout

    def acquire(self):
        start_time = time.time()
        while True:
            try:
                # Try to create the file exclusively
                fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.close(fd)
                return True
            except FileExistsError:
                # Lock exists, wait
                if time.time() - start_time > self.timeout:
                    raise TimeoutError("Could not acquire lock, previous process stuck?")
                time.sleep(1)

    def release(self):
        if os.path.exists(self.lock_file):
            try:
                os.remove(self.lock_file)
            except OSError:
                pass

def objective(trial: optuna.Trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [60, 70, 80, 90])
    batch_size = trial.suggest_categorical("batch_size", [10, 12, 14])
    gamma = trial.suggest_float("gamma", 0.99, 0.9999, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    clip_param = trial.suggest_float("clip_param", 0.01, 0.2)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.01)
    desired_kl = trial.suggest_float("desired_kl", 0.001, 0.01)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.2, 1.0)
    init_noise = trial.suggest_float("init_noise", 0.1, 0.5)
    value_loss_coeff = trial.suggest_float("value_loss_coeff", 0.8, 1.2)
    num_learning_epochs = trial.suggest_categorical("num_learning_epochs", [4, 5, 6, 7])

    log_dir = os.path.join("logs", "optuna", f"trial_{trial.number}")
    result_path = os.path.join(log_dir, "optuna_result.txt")

    cmd = [
        sys.executable, "test_train.py",
        "--task", "Muscle-Walk-Unitree-Go2-Direct-v0",
        "--max_iterations", "1",
        "--trial_id", str(trial.number),
        "--n_steps", str(n_steps),
        "--init_noise", str(init_noise),
        "--value_loss_coeff", str(value_loss_coeff),
        "--clip_param", str(clip_param),
        "--ent_coef", str(ent_coef),
        "--num_learning_epochs", str(num_learning_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--gamma", str(gamma),
        "--gae_lambda", str(gae_lambda),
        "--desired_kl", str(desired_kl),
        "--max_grad_norm", str(max_grad_norm),
        "--headless"
    ]

    print(f"[Trial {trial.number}] Starting process...")

    startup_lock = FileLock(LOCK_NAME)
    startup_lock.acquire()

    print(f"[Trial {trial.number}] Lock acquired. Starting process...")

    process = None
    lock_released = False

    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in iter(process.stdout.readline, ''):
            print(line)
            if "---ISAAC_LAB_INIT_COMPLETE---" in line:
                print(f"[Trial {trial.number}] Initialization complete. Releasing lock.")
                startup_lock.release()
                lock_released = True

            if "Traceback" in line or "Error" in line:
                pass

        process.wait()
    except Exception as e:
        if process and process.poll() is None:
            process.kill()
            
        print(f"[Trial {trial.number}] CRASHED")
        raise optuna.TrialPruned()
    finally:
        if not lock_released:
            print(f"[Trial {trial.number}] Process ended before signal. Releasing lock.")
            startup_lock.release()

    if process.returncode != 0:
        print(f"[Trial {trial.number}] Process failed with return code {process.returncode}")
        raise optuna.TrialPruned()

    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            result_str = f.read().strip()
            avg_return = float(result_str)
        
        return avg_return
    else:
        raise optuna.TrialPruned()

if __name__ == "__main__":
    if os.path.exists(LOCK_NAME):
        os.remove(LOCK_NAME)
    os.makedirs(os.path.join("logs", "optuna"), exist_ok=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=150, n_jobs=1)