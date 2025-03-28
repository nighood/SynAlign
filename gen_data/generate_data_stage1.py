import json
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KDTree
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import os
import re
import logging
import time
import seaborn as sns
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

from prompt_gen_stage_1 import All_prompt_gen_stage_1
from label_map import All_label_map
from m_util import get_data_df

# load openai api key and base url
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Set up logging, output both to console and file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# name the log file with current date and time
log_file = f"logs/generate_data_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU device to use, if needed


class Data_fit_selector:
    def __init__(self, data_file, model_name="distilbert-base-uncased", rbf_length_scale=0.1, label_filter=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Set device to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Load data
        self.data = get_data_df(data_file, label_filter=label_filter)   # label_filter is a list of label id, e.g. [0, 1]
        self.sentences = self.data['sentence'].tolist()
        self.labels = self.data['label'].tolist()
        
        # Compute embeddings
        self.embeddings = self._compute_embeddings(self.sentences)
        
        # Initialize Gaussian Process with RBF kernel
        self.rbf_length_scale = rbf_length_scale
        self.kernel = RBF(length_scale=self.rbf_length_scale)
        self.gp = GaussianProcessRegressor(kernel=self.kernel)

        self.kd_tree = KDTree(self.embeddings)
        self.X_ori = self.embeddings
        self.remaining_indices = set(range(len(self.X_ori)))
        
        # Use numpy array to store X_train instead of list
        self.X_train = np.empty((0, self.X_ori.shape[1]))
        self.indices_train = []

    def _compute_embeddings(self, sentences):
        logger.info("Computing embeddings...")
        embeddings = []
        batch_size = 32  # adjust this value according to GPU memory

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():  # no need to compute gradients to save memory
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)
        logger.info(f"Embeddings computed. Total: {len(embeddings)}")
        return embeddings

    def get_k_example(self, k_example):
        # Check if we have enough data
        if len(self.remaining_indices) < k_example:
            raise ValueError("Not enough data to select")
        
        if self.X_train.shape[0] == 0:
            # Randomly select one point to start
            idx = np.random.choice(list(self.remaining_indices))
            self.X_train = np.vstack((self.X_train, self.X_ori[idx]))
            self.indices_train.append(idx)
            self.remaining_indices.remove(idx)
            
        # Fit GP
        self.gp.fit(self.X_train, np.zeros(self.X_train.shape[0]))
        # Find the point with the largest variance in remaining
        _, variances = self.gp.predict(self.X_ori, return_std=True)
        remaining_variances = {idx: variances[idx] for idx in self.remaining_indices}
        idx = max(remaining_variances, key=remaining_variances.get)
        self.X_train = np.vstack((self.X_train, self.X_ori[idx]))
        self.indices_train.append(idx)
        self.remaining_indices.remove(idx)
        
        # Find the nearest k point to the selected point (including the selected point)
        _, indices = self.kd_tree.query(self.X_ori[idx].reshape(1, -1), k=k_example)
        for i in indices[0]:
            if i in self.remaining_indices:
                self.X_train = np.vstack((self.X_train, self.X_ori[i]))
                self.indices_train.append(i)
                self.remaining_indices.remove(i)
        
        # We will still select k examples even if some of them are already selected before
        cur_selected_indices = indices[0].tolist()
        
        return [(self.sentences[i], self.labels[i]) for i in cur_selected_indices]
    
    def get_k_example_random(self, k_example):
        # Check if we have enough data
        if len(self.remaining_indices) < k_example:
            raise ValueError("Not enough data to select")
        
        selected_indices = np.random.choice(list(self.remaining_indices), k_example, replace=False)
        self.X_train = np.vstack((self.X_train, self.X_ori[selected_indices]))
        self.indices_train.extend(selected_indices)
        self.remaining_indices = self.remaining_indices - set(selected_indices)
        
        return [(self.sentences[i], self.labels[i]) for i in selected_indices]

    @staticmethod
    def plot_tsne(embedding_lst: list, label_lst: list, alpha_ls: list, save_path: str = None):
        """
        Overview:
            Plot t-SNE visualization of sentence embeddings.
        Arguments:
            - embedding_lst (:obj:`list` of :obj:`np.ndarray`): List of sentence embeddings.
            - label_lst (:obj:`list` of :obj:`str`): List of labels for each set of embeddings.
            - alpha_ls (:obj:`list` of :obj:`float`): List of alpha values for each set of embeddings.
            - save_path (:obj:`str`, optional): Path to save the plot. If not provided, the plot will be displayed.
        """
        all_embeddings = np.vstack(embedding_lst)
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_tsne = tsne.fit_transform(all_embeddings)
        
        plt.figure(figsize=(12, 6))
        # Get colors from the tab10 palette
        palette = sns.color_palette("tab10")
        
        cur_idx = 0
        for i, embedding in enumerate(embedding_lst):
            color = palette[i]
            sns.scatterplot(
                x=embeddings_tsne[cur_idx:cur_idx+len(embedding), 0],
                y=embeddings_tsne[cur_idx:cur_idx+len(embedding), 1],
                alpha=alpha_ls[i],
                label=label_lst[i],
                color=color,
            )
            cur_idx += len(embedding)
        
        plt.title('t-SNE visualization of sentence embeddings')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def show_tsne(self, save_folder=None, gene_data_path=None):
        """
        Overview:
            Plot t-SNE visualization of sentence embeddings.
        Arguments:
            - save_folder (:obj:`str`, optional): Path to save the plot. If not provided, the plot will be displayed.
            - gene_data_path (:obj:`str`, optional): Path to the generated data file. If provided, the generated data will be plotted in t-SNE.
        """
        # if all data are selected, directly return
        if len(self.remaining_indices) == 0:
            logger.info("All data are selected. Skip t-SNE plot.")
            return
        # if num of X_ori is less than 30, directly return. for t-SNE plot need at least 30 data
        if len(self.X_ori) < 30:
            logger.info("Number of data is less than 30. Skip t-SNE plot.")
            return
        # plot remain data and selected data in t-SNE, default save to save_folder
        # Get embeddings of remaining data
        remaining_embeddings = self.X_ori[list(self.remaining_indices)]
        selected_embeddings = self.X_ori[self.indices_train]
        embeddings_lst = [remaining_embeddings, selected_embeddings]
        label_lst = ['Original Data', 'Selected Data']
        alpha_ls = [0.5, 1.0]
        save_path = os.path.join(save_folder, 'tsne_selected.png') if save_folder else None
        Data_fit_selector.plot_tsne(embeddings_lst, label_lst, alpha_ls, save_path=save_path)

        # if given gene_data_path, plot gene data in t-SNE
        if gene_data_path:
            with open(gene_data_path, 'r') as f:
                gene_data = json.load(f)
            # the key will be 'sentence' or 'text'
            if 'sentence' in gene_data[0]:
                gene_sentences = [data['sentence'] for data in gene_data]
            elif 'text' in gene_data[0]:
                gene_sentences = [data['text'] for data in gene_data]
            gene_embeddings = self._compute_embeddings(gene_sentences)
            embeddings_lst = [remaining_embeddings, selected_embeddings, gene_embeddings]
            label_lst = ['Original Data', 'Selected Data', 'Generated Data']
            alpha_ls = [0.5, 0.5, 1.0]
            save_path = os.path.join(save_folder, 'tsne_gene.png') if save_folder else None
            Data_fit_selector.plot_tsne(embeddings_lst, label_lst, alpha_ls, save_path=save_path)
        

    def save_selected_data(self, save_path):
        # save selected data to a json file
        selected_data = [{"text": self.sentences[i], "label": self.labels[i]} for i in self.indices_train]
        with open(save_path, 'w') as f:
            json.dump(selected_data, f, indent=2)
        logger.info(f"Selected data saved to {save_path}")


class Data_generator:
    def __init__(
            self,
            model_type,
            model_name,
            data_file,
            dataset: str = "sst-2",
            rbf_length_scale=0.1,
            gen_split_ids=None,
            ):
        self.model_type = model_type
        self.model_name = model_name
        if self.model_type == "llama":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto",)
            self.max_length = self.tokenizer.model_max_length
        elif self.model_type == "openai":
            self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            self.encoding = tiktoken.encoding_for_model(model_name)
            self.max_length = 16385  # adjust this value according to the model's context length limit
        else:
            raise ValueError(f"Invalid model type: {model_type}")
            
        assert dataset in All_label_map, f"Dataset {dataset} not found in dataset supported list"
        # self.dataset_description = dataset_description[dataset]
        # self.summary_description = summary_description[dataset]
        # Set label map for we will use text label in prompt
        self.label2id = {label: idx for idx, label in enumerate(All_label_map[dataset])}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.prompt_template = All_prompt_gen_stage_1[dataset]

        # for data selection
        self.data_file = data_file
        self.rbf_length_scale = rbf_length_scale
        self.gen_split_ids = gen_split_ids

    def generate_data(self, k_example, n_epoch, gen_size, save_file, sample_method='gp'):
        self.k_example = k_example
        self.n_epoch = n_epoch
        self.gen_size = gen_size

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        # all_data = []
        generated_data_num = 0
        while generated_data_num < self.n_epoch:
            logger.info("\n--------------------------------")
            logger.info(f"\nEpoch {generated_data_num+1}/{self.n_epoch}")
            if sample_method == 'gp':
                examples = self.selector.get_k_example(k_example=self.k_example)
            elif sample_method == 'random':
                examples = self.selector.get_k_example_random(k_example=self.k_example)
            prompt = self._create_prompt(examples)
            generated_summary = self._generate_data_from_prompt(prompt)
            
            # Extend all_data with new generated data
            if generated_summary is None:
                logger.error(f"Failed to generate data for epoch {generated_data_num+1}. Skipping...")
                continue
            
            # Append the new data to the file
            with open(save_file, 'a') as f:
                json.dump(generated_summary, f) # generated_summary is a dict
                f.write('\n')  # output file should be jsonl format
                generated_data_num += 1
            
            logger.info(f"Appended data from epoch {generated_data_num+1} to {save_file}")

        logger.info(f"\n\nData generation completed. All data saved in {save_file}")
        logger.info(f"Total generated data: {generated_data_num}, expected: {self.n_epoch}")


    def caculate_tokens_num(self, prompt: str):
        if self.model_type == "openai":
            return len(self.encoding.encode(prompt))
        else:
            return self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        
    def _create_prompt(self, examples: list[tuple[str, int]]):
        # Combine examples into a prompt
        chosen_example = examples[0]    # tuple: (sentence, label_id)
        example_dict = {"text": chosen_example[0], "label": self.id2label[chosen_example[1]]}
        self.cur_example = example_dict # we need to add label info after generation
        # note: here we just use one example to summarize
        example_str = json.dumps(example_dict)
        prompt = self.prompt_template.format(
            examples_str=example_str,
        )
        # we ignore the case that the prompt is too long, for we just use one example to summarize
        logger.info(f"Prompt:\n{prompt}\n")
        message = [
            # {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
            ]
        return message

    def extract_json_from_string(self, input_string: str):
        """
        Overview:
            Extract JSON data from the generated string.
        Arguments:
            - input_string (:obj:`str`): The generated string.
        Returns:
            - dict_data (:obj:`dict`): The extracted JSON data with label and text info.
        """
        # Use regex to find all JSON arrays between '{' and '}'
        dict_match = re.findall(r'\{.*?\}', input_string, re.DOTALL)
        if dict_match:
            dict_str = dict_match[0]    # only use the first matched dict
            try:
                dict_data = json.loads(dict_str)
                if 'label' not in dict_data:
                    # add label/text info to data
                    dict_data['label'] = self.cur_example['label']
                    dict_data['text'] = self.cur_example['text']
                return dict_data
            except Exception as e:
                logger.error(f"Failed to load dict data from input string: {input_string}. Error: {e}")
                return None
        else:
            logger.error("No JSON data found in input string, the json_str:\n", input_string)
            return None
    
    def _generate_data_from_prompt(self, prompt: list[dict]):
        """
        Overview:
            Generate one summary dict data using the prompt.
        Arguments:
            - prompt (:obj:`list` of :obj:`dict`): The prompt message to generate data.
        Returns:
            - data_dict (:obj:`dict`): The generated data dict.
        """
        if self.model_type == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt,
                    # temperature=0.7,
                    # max_tokens=max_tokens,
                    # top_p=0.9,
                    # top_k=1,
                )
                generated_data = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Failed to generate data with openai api: {e}")
                return None
        elif self.model_type == "llama":
            # Change the padding token to eos token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # use chat template to generate data for better performance
            inputs = self.tokenizer.apply_chat_template(prompt,
                                                        return_tensors="pt",
                                                        add_generation_prompt=True,
                                                        tokenize=True,
                                                        return_dict=True,)
            inputs = inputs.to(self.model.device)

            gen_args = {"max_new_tokens": 6144, "do_sample": True, "temperature": 0.1, "top_k": 1}
            outputs = self.model.generate(**inputs, **gen_args)
            outputs = outputs[:, inputs["input_ids"].shape[1]:]
            generated_data = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_data = generated_data[0]
        logger.info(f'Generated Data:\n {generated_data}')
        data_dict = self.extract_json_from_string(generated_data)
        return data_dict

    def generate_save_data(
            self,
            k_example,
            n_epoch,
            gen_size,
            save_folder,
            sample_method='gp',
            gen_per_label='average',    # or 'proportion'
            ):
        """
        Overview:
            Generate data and save the selected data and t-SNE plots.
        Arguments:
            - k_example (:obj:`int`): Number of examples to select in each epoch.
            - n_epoch (:obj:`int`): Number of epochs to run.
            - gen_size (:obj:`int`): Number of generated examples in each epoch.
            - save_folder (:obj:`str`): Path to save the generated data and t-SNE plots.
            - sample_method (:obj:`str`, optional): Method to sample data, 'gp' for Gaussian Process, 'random' for random sampling.
            - gen_per_label (:obj:`str`, optional): Method to generate data for each label, 'average' for average number of data, 'proportion' for proportion of data.
        """
        assert sample_method in ['gp', 'random'], f"sample_method should be 'gp' or 'random', not {sample_method}"
        if self.gen_split_ids is not None:
            # generate data for each label, and save the selected data and t-SNE plot
            id_filters = self.gen_split_ids
            # we generate n_epoch number of data totally,
            # so we need to divide n_epoch by proportion/average of each label in data_file
            # e.g. proportion mode. if we have 3 labels, and the proportion is [0.2, 0.3, 0.5], then we generate n_epoch*[0.2, 0.3, 0.5] data for each label
            all_data = get_data_df(self.data_file)
            total_num = len(all_data)

            for id_filter in id_filters:
                cur_label_num = len(all_data[all_data['label'] == id_filter])
                if gen_per_label == 'proportion':
                    cur_label_gen_epoch = int(n_epoch * cur_label_num / total_num) + 1
                elif gen_per_label == 'average':
                    cur_label_gen_epoch = min(n_epoch // len(id_filters), cur_label_num)
                    if cur_label_gen_epoch == cur_label_num:
                        # use random sample method, avoid error
                        sample_method = 'random'
                    else:
                        sample_method = 'gp'
                        
                logger.info(f"\nGenerate data for label {id_filter}, total data num: {cur_label_num}, generate data num: {cur_label_gen_epoch}\n")

                save_folder_id = os.path.join(save_folder, f'label_{id_filter}')
                if not os.path.exists(save_folder_id):
                    os.makedirs(save_folder_id)
                file_path = os.path.join(save_folder_id, 'gen_summary_data.jsonl')

                self.selector = Data_fit_selector(data_file=self.data_file, rbf_length_scale=self.rbf_length_scale, label_filter=[id_filter])
                # gen_size is not used here, for we just use one example to summarize
                self.generate_data(k_example, cur_label_gen_epoch, gen_size, file_path, sample_method=sample_method)
                self.selector.show_tsne(save_folder=save_folder_id)
                self.selector.save_selected_data(os.path.join(save_folder_id, 'selected_data.json'))
        else:
            file_path = os.path.join(save_folder, 'gen_summary_data.jsonl')
            # generate data for all labels, and save the selected data and t-SNE plot
            self.selector = Data_fit_selector(data_file=self.data_file, rbf_length_scale=self.rbf_length_scale)
            self.generate_data(k_example, n_epoch, gen_size, file_path, sample_method=sample_method)
            self.selector.show_tsne(save_folder=save_folder)
            self.selector.save_selected_data(os.path.join(save_folder_id, 'selected_data.json'))


if __name__ == '__main__':
    # dataset = "sst-2"
    # data_file = 'data/original/SST-2/train.tsv'
    # dataset = "amazon"
    # data_file = 'data/original/amazon-attrprompt/train.jsonl'
    dataset = "agnews"
    data_file = 'data/original/agnews-attrprompt/train.jsonl'
    k_example = 2   # note: we just use first example to summarize, other examples for rubst selection
    n_epoch = 4*600  # 80*23 for amazon, 600*4 for agnews
    gen_size = 0
    rbf_length_scale = 0.1
    
    # set random seed
    np.random.seed(42)

    # Initialize Data_generator
    # model_type = "llama"
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_type = "openai"
    # model_name = "gpt-4o-2024-05-13"
    model_name = "gpt-3.5-turbo-0125"
    data_generator = Data_generator(
        model_type=model_type,
        model_name=model_name,
        data_file=data_file,
        dataset=dataset,
        rbf_length_scale=rbf_length_scale,
        gen_split_ids=list(range(4)),
        )

    # Generate and save data
    sample_method = 'gp'
    gen_per_label = 'average' # or 'proportion'
    # gen_per_label = 'proportion'
    data_save_fold = f"your_result/{dataset}_na_{model_name}_{sample_method}_n{n_epoch}_rbf{rbf_length_scale}_{gen_per_label}_{time.strftime('%Y-%m-%d_%H-%M')}"
    if not os.path.exists(data_save_fold):
        os.makedirs(data_save_fold)
    data_generator.generate_save_data(
        k_example=k_example,
        n_epoch=n_epoch,
        gen_size=gen_size,
        save_folder=data_save_fold,
        sample_method=sample_method,
        gen_per_label=gen_per_label,
        )
    logger.info(f"\nAPI using: {OPENAI_BASE_URL}")