import json 
import random 
from nltk.tokenize import word_tokenize
from omegaconf import DictConfig, OmegaConf
from graph import create_graph 
from state import State, Config 
from tqdm import tqdm
import hydra
from time import sleep
import os 
import nltk 
from warnings import filterwarnings
import logging 
import wandb 


nltk.download('punkt_tab')
filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


slience_list = [
    "httpx", 
    "httpcore", 
    "qdrant_client", 
    "urllib3", 
    "nltk"
]

for logger_name in slience_list:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


class Benchmark: 


    def __init__(self, config, data_path: str):

        self.data_path = data_path 
        self.config = config 
        
        with open(data_path, 'r') as file:
            self.data = json.load(file)

    
        self.results = {
            "mean_iter_num": 0.0,
            "mean_f1_score": 0.0
        }

        self.workflow = create_graph()
        


    def sample_data(self, method: str = "in_range"): 


        if method == "random":
            sample_size = self.config.get("sample_size", 10)
            sample_size = min(sample_size, len(self.data))
            random.seed(self.config.get("seed", 42))
            return random.sample(self.data, sample_size)

        elif method == "in_range": 
            part = self.config.get("part", 0)

            part_item = len(self.data) // self.config.get("num_parts", 1)
            start = part * part_item
            end = start + part_item if part < self.config.get("num_parts", 1) - 1 else len(self.data)
            return self.data[start:end]


    def calculate_f1_score(self, ground_truth: str, pred: str): 

        preds_token = set(word_tokenize(pred.lower()))
        ground_truth_token = set(word_tokenize(ground_truth.lower()))

        common_token  = preds_token.intersection(ground_truth_token)

        precision = len(common_token) / len(preds_token) if preds_token else 0
        recall = len(common_token) / len(ground_truth_token) if ground_truth_token else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f1_score


    def invoke_graph(self, question): 

        state_config = Config(
            backbone = self.config.get("backbone", "gemini-2.0-flash"),
            early_stopping = self.config.get("early_stopping", 3),
            k = self.config.get("k", 2)
        )

        init_state = State(
            question = question,
            config = state_config,
        )

        output_state = self.workflow.invoke(init_state)

        pred = output_state.get("final_answer", "")
        early_stopping = output_state.get("config").early_stopping

        num_iter = self.config.get("early_stopping") - early_stopping

        return pred, num_iter 



    def save(self, output_path: str, tracking_data: dict = None): 
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as file:
            json.dump({
                "metrics": self.results,
                "config": self.config,
                "sampled_data": tracking_data
            }, file, indent=2)
            


    
    def run_benchmark(self, id: str = "default"):

        sampled_data = self.sample_data(method=self.config.get("sampling_method", "in_range"))

        tracking_data = []

        for item in tqdm(sampled_data): 
            query = item.get("query", "")
            ground_truth = item.get("answer", "")

            pred, num_iter = self.invoke_graph(query)
            f1_score = self.calculate_f1_score(ground_truth, pred)

            self.results["mean_f1_score"] += f1_score
            self.results["mean_iter_num"] += num_iter


            tracking_data.append({
                "query": query,
                "ground_truth": ground_truth,
                "prediction": pred,
                "f1_score": f1_score,
                "num_iterations": num_iter
            })

            wandb.log({
                "f1_score": f1_score,
                "num_iterations": num_iter,
            })
            self.save(f"outputs/{id}/results_{id}.json", tracking_data)

            # sleep(40)


        self.results["mean_f1_score"] = self.results["mean_f1_score"] / len(sampled_data) if sampled_data else 0.0
        self.results["mean_iter_num"] = self.results["mean_iter_num"] / len(sampled_data) if sampled_data else 0.0
        
        wandb.log({
            "mean_f1_score": self.results["mean_f1_score"],
            "mean_iter_num": self.results["mean_iter_num"],
        })
        part = self.config.get("part", 0)
        self.save(f"outputs/{id}/results_{id}_part{str(part)}.json", tracking_data)

@hydra.main(config_path = "config", config_name = "default") 
def run(cfg: DictConfig): 

    wandb.login(
        key = os.getenv("WANDB_API_KEY")
    )

    config = OmegaConf.to_container(cfg, resolve=True)

    wandb.init(
        project = "MultiHopRAG",
        name = cfg.id, 
        config = config, 

    )

    print("Running with config: ") 
    print(OmegaConf.to_yaml(config))

    benchmark = Benchmark(
        config = config,
        data_path = os.path.join("data", "MultiHopRAG.json")
    )

    benchmark.run_benchmark(id=cfg.id)
    wandb.finish()

    print("Done!") 




if __name__ == "__main__":
    run()
