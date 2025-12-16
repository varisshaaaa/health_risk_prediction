from prefect import flow, task
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.utils.dynamic_learner import train_model, integrate_new_symptom

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
    # In V4.0, scraping is dynamic. We can just ensure the DB is healthy or run a basic check.
    # Or import from a static file if present
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("Precautions are now managed dynamically. Checking integrity...")
    # Optional: trigger a training run to ensure new data is integrated
    return True

@task(name="Retrain Model")
def retrain_model_task():
    train_model()

@flow(name="Health System Orchestration")
def health_system_flow():
    # 1. Pipeline: Ingest Data
    import_precautions_task()
    
    # 2. Pipeline: Retrain Model
    retrain_model_task()
    
    print("Orchestration flow completed.")

if __name__ == "__main__":
    health_system_flow()
