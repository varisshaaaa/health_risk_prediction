from prefect import flow, task
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.utils.retrain import run_retraining_job
from backend.utils.import_precautions import import_precautions

@task(name="Run Scraper")
def run_scraper_task():
    """
    Task to execute the Scrapy spider.
    Note: In a real deployment, this might trigger a separate container or process.
    Here we simulate or assume the file is generated manually.
    """
    print("Simulating scraper execution... (Ensure 'precautions.json' is updated)")
    # os.system("scrapy crawl precautions -O precautions.json")
    return True

@task(name="Import Precautions")
def import_precautions_task():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, 'webscraping', 'precautions.json')
    if os.path.exists(json_path):
        import_precautions(json_path)
    else:
        print(f"Precautions file not found at {json_path}")

@task(name="Retrain Model")
def retrain_model_task():
    run_retraining_job()

@flow(name="Health System Orchestration")
def health_system_flow():
    # 1. Pipeline: Ingest Data
    import_precautions_task()
    
    # 2. Pipeline: Retrain Model
    retrain_model_task()
    
    print("Orchestration flow completed.")

if __name__ == "__main__":
    health_system_flow()
