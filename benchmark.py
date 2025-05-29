import json 
import random 
from nltk.tokenize import word_tokenize
from omegaconf import DictConfig, OmegaConf
from graph import create_graph 
from state import State, Config 
from tqdm import tqdm
import hydra

class Benchmark: 


    def __init__(self, config, data_path: str):

        self.data_path = data_path 
        self.config = config 
        
        with open(data_path, 'r') as file:
            self.data = json.load(file)

    
        self.results = {
            "f1" : [], 
            "num_iterations" : [],
            "mean_f1_score": 0.0
        }

        self.workflow = create_graph()


    def sample_data(self, method: str = "random"): 

        sample_size = self.config.get("sample_size", 10)
        sample_size = min(sample_size, len(self.data))

        
        if method == "random":
            random.seed(self.config.get("seed", 42))
            return random.sample(self.data, sample_size)

        elif method == "first_n":
            return self.data[:sample_size]


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

        num_iter = self.config.get("early_stopping") - output_state.config.early_stopping

        return pred, num_iter 



    def save(self, output_path: str, tracking_data: dict = None): 

        with open(output_path, 'w') as file:
            json.dum({
                "metrics": self.results,
                "config": self.config,
                "sampled_data": tracking_data
            }, file, indent=2)
            


    
    def run_benchmark(self, id: str = "default"):

        sampled_data = self.sample_data(method=self.config.get("sampling_method", "random"))

        tracking_data = {
            "questions": [],
            "ground_truths": [],
            "predictions": [],
        }

        for item in tqdm(sampled_data): 
            query = item.get("query", "")
            ground_truth = item.get("answer", "")

            pred, num_iter = self.invoke_graph(query)

            tracking_data["questions"].append(query)
            tracking_data["ground_truths"].append(ground_truth)
            tracking_data["predictions"].append(pred)

            f1_score = self.calculate_f1_score(ground_truth, pred)

            self.results["f1"].append(f1_score)
            self.results["num_iterations"].append(num_iter)

        self.results["mean_f1_score"] = sum(self.results["f1"]) / len(self.results["f1"]) if self.results["f1"] else 0.0

        self.save(f"output/benchmark_results_{id}.json", tracking_data)



@hydra.main(config_path = "config", config_name = "default") 
def run(cfg: DictConfig): 

    config = OmegaConf.to_container(cfg, resolve=True)
    benchmark = Benchmark(
        config = config,
        data_path = "data/MultiHopRAG.json"
    )


    benchmark.run_benchmark(id=cfg.id)

    print("Done!") 




if __name__ == "__main__":
    run()
