"""
Date: July - December, 2023
@author: Maria Auxiliadora Mora
contact: mariamoracross@gmail.com
@version: v1.0
@description: Large Language Models (LLM) provide significant value in question answering (QA) scenarios and have practical application in complex decision-making contexts, such as biodiversity conservation. However, despite substantial performance improvements, they may still produce inaccurate outcomes. Consequently, incorporating uncertainty quantification alongside predictions is essential for mitigating the potential risks associated with their use. This study introduces an exploratory analysis of the application of Monte Carlo Dropout (MCD) and Expected Calibration Error (ECE) to assess the uncertainty of generative language models. To that end, we analyzed two publicly available language models (Falcon-7B and DistilGPT- 2). Our findings suggest the viability of employing ECE as a metric to estimate uncertainty in generative LLM. The findings from this research contribute to a broader project aiming at facilitating free and open access to standardized and integrated data and services about Costa Rica’s biodiversity to support the development of science, education, and biodiversity conservation.


This file contains the functions used to run the experiments that support the research results of the publication: Maria Mora-Cross and Saul Calderon-Ramirez (2024). Uncertainty Estimation in Large Language Models to Support Biodiversity Conservation. Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies.

Two language models were used GPT-2 and Falcon-7B. The results of all experiments were stored in a PostgreSQL database.

@license: MIT License

© 2024 Maria Mora-Cross

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Libraries
import sys
import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import random
import torch.nn.functional as F
from datetime import date
from decimal import Decimal

# sklearn
from sklearn.preprocessing import MinMaxScaler

# imbalace library
from imblearn.under_sampling import RandomUnderSampler

# Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config
from huggingface_hub import notebook_login
import datasets
from datasets import load_dataset
from datasets import DatasetDict
from collections import defaultdict

# Metrics
from datasets import load_metric
from transformers import pipeline

import matplotlib.pyplot as plt

# hugging face metrics
from evaluate import load

# Bibliotecas requeridas para SPARK
from pyspark.context import SparkContext
#from pyspark.sql.functions import *
#from pyspark.sql.types import *
from datetime import date, timedelta, datetime
import time


from pyspark.sql import SparkSession, Row, dataframe
from pyspark.sql.functions import col

import findspark
SPARK_PATH = '/opt/spark/spark-3.5.0-bin-hadoop3'
findspark.init(SPARK_PATH)

import transformers
import huggingface_hub

# ============================= Global variables =======================================================
is_gpu_available = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Results are stored in a PosgreSQL database
# PostgreSQL connection
POSTGRESQL_URL = "jdbc:postgresql://localhost:5432/ece"
POSTGRESQL_USER = "ece"
POSTGRESQL_PASSWORD = "ece"

#bleu_metric = load_metric("sacrebleu")

# Metrics
bleu_metric = load("sacrebleu")
bertscore = load("bertscore")

def create_spark_session():
    """
    This function builds a Spark Session
    return the main entry of a Spark DataFrame
    """
    spark = SparkSession \
      .builder \
      .appName("Basic JDBC pipeline") \
      .config("spark.driver.extraClassPath", "/home/mmora/Documents/postgres/postgresql-42.6.0.jar") \
      .config("spark.executor.extraClassPath", "/home/mmora/Documents/postgres/postgresql-42.6.0.jar") \
      .getOrCreate()
    return spark

# Spark session
spark = create_spark_session()




# ============================= SPARK ===========================================

# Save data teh database PostgreSQL
def write_spark_df_to_db(spark_df, table_name):
    """
    This function writes Spark dataframe to DB. It creates the table. 
    """
    spark_df \
        .write \
        .format("jdbc") \
        .mode('overwrite') \
        .option("url", POSTGRESQL_URL) \
        .option("user", POSTGRESQL_USER) \
        .option("password", POSTGRESQL_PASSWORD) \
        .option("dbtable", table_name) \
        .save()

#---------------------------------------------------------------------------------
def insert_spark_df_to_db(spark_df, table_name):
    """
    This function insert Spark dataframe to DB
    """
    spark_df \
        .write \
        .format("jdbc") \
        .mode('append') \
        .option("url", POSTGRESQL_URL) \
        .option("user", POSTGRESQL_USER) \
        .option("password", POSTGRESQL_PASSWORD) \
        .option("dbtable", table_name) \
        .save()
    #print('=============== insert_spark_df_to_db')
    spark_df.show()

#---------------------------------------------------------------------------------
def read_dataset_from_db(query_text):
    """
    This function reads data from a table in the postgresql database

    param:
       query_text: the complete select ....
    """
    df = spark.read \
           .format("jdbc") \
           .option("url", POSTGRESQL_URL) \
           .option("user", POSTGRESQL_USER) \
           .option("password", POSTGRESQL_PASSWORD) \
           .option("query", query_text) \
           .load()
    #df.show()
    return df
    
#---------------------------------------------------------------------------------
def create_spark_session():
    """
    This function builds a Spark Session
    return the main entry of a Spark DataFrame
    """
    spark = SparkSession \
      .builder \
      .appName("Basic JDBC pipeline") \
      .config("spark.driver.extraClassPath", "/home/mmora/Documents/postgres/postgresql-42.6.0.jar") \
      .config("spark.executor.extraClassPath", "/home/mmora/Documents/postgres/postgresql-42.6.0.jar") \
      .getOrCreate()
    return spark

# ============================= # Utility functions =======================================================

def display_library_version(library):
    print(f"Using {library.__name__} v{library.__version__}")

#---------------------------------------------------------------------------------
def setup():
    # Check if we have a GPU
    if not is_gpu_available:
        print("No GPU was detected! This notebook can be *very* slow without a GPU 🐢")
        if is_colab:
            print("Go to Runtime > Change runtime type and select a GPU hardware accelerator.")
        if is_kaggle:
            print("Go to Settings > Accelerator and select GPU.")
    # Give visibility on versions of the core libraries
    display_library_version(transformers)
    display_library_version(datasets)
    # Disable all info / warning messages
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    # Logging is only available for the chapters that don't depend on Haystack
    if huggingface_hub.__version__ == "0.0.19":
        huggingface_hub.logging.set_verbosity_error()
 

#---------------------------------------------------------------------------------
def perplexity(text, model, tokenizer):
    """Compute perplexity for a text or a Dataset
    With Transformers, we can simply pass the input_ids as the labels to our model,
    and the average negative log-likelihood for each token is returned as the loss.
    With our sliding window approach, however, there is overlap in the tokens we pass
    to the model at each iteration. We don’t want the log-likelihood for the tokens we’re
    just treating as context to be included in our loss, so we can set these targets to -100
    so that they are ignored. The following is an example of how we could do this with a
    stride of 512. This means that the model will have at least 512 tokens for context when
    calculating the conditional likelihood of any one token (provided there are 512 preceding
    tokens available to condition on).
    Source https://huggingface.co/docs/transformers/perplexity
    Parameters:
       text: a string. if text is part of a Dataset it must be process (i.e. "\n\n".join(eli5_test['title']))
       model
       tokenizer
    """
    encodings = tokenizer(text, return_tensors="pt")

    if model.config._name_or_path=='tiiuae/falcon-7b':
       max_length = model.config.max_position_embeddings # mmora
    else: #gpt 
       max_length = model.config.n_positions
    	
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    #for begin_loc in tqdm(range(0, seq_len, stride)):
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return(ppl)

#------------------------------ expected_calibration_error ---------------------------
def expected_calibration_error_bin(mean_predicted_values, mean_true_values,  question_count_list, bin_list):
    """Compute the expected calibration error of predicted and true values.
    Parameters:
       mean_predicted_values, mean_true_values: Two numpy.ndarray with predicted_values, true_values
       bin_list : bins
    """
    
    # Compute the calibration error for each bin
    calibration_errors = [abs(mp - mt) for mp, mt in zip(mean_predicted_values, mean_true_values)]

    # Compute the weighted average of calibration errors
    num_samples = sum(question_count_list)
    ece = 0
    for i in range(len(bin_list)):
        ece = ece + abs(mean_predicted_values[i] - mean_true_values[i]) * Decimal(question_count_list[i])/num_samples

    return ece, calibration_errors
    
#--------------------------------------------------------------------------------------------    
def select_N_elements_each_class(class_list, num_repetitions):
    """Select N repetitions of each class from a list and replace the non-selected elements with 0."""
    # Create a dictionary to store the counts of each class
    class_counts = {}

    # Initialize the dictionary with counts of each class
    for class_item in class_list:
        if class_item not in class_counts:
            class_counts[class_item] = 0

    # Iterate through the list and randomly select 10 repetitions for each class
    for i, class_item in enumerate(class_list):
        if class_counts[class_item] < num_repetitions:
            # Select this element
            class_counts[class_item] += 1
        else:
            # Replace non-selected elements with 0
            class_list[i] = 0
    return(class_list)    
    
#--------------------------------------------------------------------------------------------

def expected_calibration_error(predicted_values, true_values, num_bins):
    """Compute the expected calibration error of predicted and true values.
    Parameters:
    predicted_values, true_values: Two numpy.ndarray with predicted_values, true_values
    num_bins : bins
    """
    # Bin the predicted and true values
    bins = np.linspace(0, 1, num_bins)

    bin_indices = np.digitize(predicted_values, bins)



    # Initialize lists to store mean predicted and true values
    mean_predicted_values = []
    mean_true_values = []

    # to visualize
    #mean_predicted_values2 = []
    #mean_true_values2 = []


    bins_with_data = []
    # Calculate mean values for each bin
    for bin in range(1, num_bins + 1):

        mask = bin_indices == bin
        
        if np.any(mask):

            mean_predicted = np.mean(predicted_values[mask])

            mean_true = np.mean(true_values[mask])
            mean_predicted_values.append(mean_predicted)
            mean_true_values.append(mean_true)
            bins_with_data.append(bin)
   

    # Compute the calibration error for each bin
    calibration_errors = [abs(mp - mt) for mp, mt in zip(mean_predicted_values, mean_true_values)]

    # Compute the weighted average of calibration errors
    weights = [np.sum(bin_indices == bin) / len(bin_indices) for bin in range(1, num_bins+1)]
    #print("+++++ weights", weights)

    weights = [x for x in weights if x > 0]

    ece = np.average(calibration_errors, weights=weights)
    
    # count the number of elements in each bin 
    list_of_numbers = list(range(1, 11))
    ocurrences = count_ocurrences_of_alist(bin_indices.tolist(), list_of_numbers)
    
    return ece, calibration_errors, bins_with_data, mean_predicted_values, mean_true_values, ocurrences

#--------------------------------------------------------------------------------------------

def expected_calibration_error_visualization(predicted_values, true_values, num_bins, num_elements_per_class):
    """Compute the expected calibration error of predicted and true values.
    Parameters:
    predicted_values, true_values: Two numpy.ndarray with predicted_values, true_values
    num_bins : bins
    """
    # Bin the predicted and true values
    bins = np.linspace(0, 1, num_bins)

    # Assign a class to each element  
    bin_indices = np.digitize(predicted_values, bins)
    
    # reduce the instances for each class to num_elements_per_class
    bin_indices = select_N_elements_each_class(bin_indices, num_elements_per_class)

    # Initialize lists to store mean predicted and true values
    mean_predicted_values = []
    mean_true_values = []

    # to visualize
    #mean_predicted_values2 = []
    #mean_true_values2 = []


    bins_with_data = []
    # Calculate mean values for each bin
    for bin in range(1, num_bins + 1):

        mask = bin_indices == bin

        if np.any(mask):
            mean_predicted = np.mean(predicted_values[mask])
            mean_true = np.mean(true_values[mask])
            mean_predicted_values.append(mean_predicted)
            mean_true_values.append(mean_true)
            bins_with_data.append(bin)
   

    # Compute the calibration error for each bin
    calibration_errors = [abs(mp - mt) for mp, mt in zip(mean_predicted_values, mean_true_values)]

    # Compute the weighted average of calibration errors
    weights = [np.sum(bin_indices == bin) / len(bin_indices) for bin in range(1, num_bins+1)]

    weights = [x for x in weights if x > 0]

    ece = np.average(calibration_errors, weights=weights)
    
    # count the number of elements in each bin 
    list_of_numbers = list(range(1, 11))
    ocurrences = count_ocurrences_of_alist(bin_indices.tolist(), list_of_numbers)
    
    return ece, calibration_errors, bins_with_data, mean_predicted_values, mean_true_values, ocurrences

# ------------------------------ BLEu per sample ---------------------------------------------
def BLEU_per_sample (predictions,references ):
    """Calculates the BLEU metric per sample and returns the average and a vector with the results"""
    a_reference = []

    individual_bleu_scores = []

    for sample_prediction, sample_references in zip(predictions, references):
        # Calcular BLEU para la muestra actual
        a_reference.append(sample_references)

        bleu_score =bleu_metric.compute(predictions=sample_prediction,
                                            references=a_reference)
        individual_bleu_scores.append(bleu_score['score'])
        a_reference=[]

    bleu_average = np.average(individual_bleu_scores)
    return bleu_average, individual_bleu_scores

# ----------------------------------------------------------------------------------------------------
def compute_perplexity_and_prediction(prompt, tokenizer, model, answer_length, topk, topp, padToken_id=50256):
    """
    Computes the model perplexity and prediction.
    """
    # System generated text
    predictions = []

    # Generate and append prediction
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attn_masks = inputs["attention_mask"]

    # attention_mask=tokenization.attention_mask
    with torch.no_grad():
        outputs = model.generate(input_ids.to(device), \
                                 max_new_tokens=answer_length,\
                                 pad_token_id = padToken_id, \
                                 do_sample=True, top_k=topk, top_p=topp, \
                                 attention_mask= attn_masks.to(device),
                                 renormalize_logits=True,
                                 return_dict_in_generate=True, output_scores=True)



    model_answer = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

    # process the model answer
    predictions.append(model_answer[0])


    # model perplexity
    answer_perplexity = perplexity(model_answer, model, tokenizer)

    return answer_perplexity, predictions

#----------------------------------------------------------------------------------------------------

def best_answer (answer_list, score_list):
    """ Select the best answer from a list of answers. The selection is guided by score_list.
        The text that correspond to the best score is returned.
    """
    max_score = 0
    for answer_i in range(len(answer_list)):

        if score_list[answer_i] > max_score:
            final_text = answer_list[answer_i]
            max_score = score_list[answer_i]
    return(final_text)




# ----------------------------------------------------------------------------------------------------                

def compute_perplexity_and_prediction_falcon(prompt, tokenizer, model, answer_length, topk, penalty_alpha, best_answer_resp, language):
    """
    Computes the model perplexity and prediction.
    """
    # System generated text
    predictions = []

    # Generate and append prediction
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attn_masks = inputs["attention_mask"]

    # attention_mask=tokenization.attention_mask
    with torch.no_grad():
        outputs = model.generate(input_ids.to(device), \
                                 attention_mask= attn_masks.to(device),
                                 max_length=answer_length,\
                                 do_sample=True, \
                                 top_k=topk, 
                                 penalty_alpha=penalty_alpha, \
				 num_return_sequences=1,
            			 eos_token_id=tokenizer.eos_token_id,
                                 )

    model_answer = tokenizer.batch_decode(outputs)

    # process the model answer
    predictions.append(model_answer[0])

    # model perplexity
    answer_perplexity = perplexity(model_answer, model, tokenizer)
    
    #bertscore
    bert_results = bertscore.compute(predictions=[model_answer[0]], references=[best_answer_resp],\
                   lang=language)
    

    return answer_perplexity, predictions, bert_results['f1'][0], bert_results['precision'][0], bert_results['recall'][0]


#---------------------------------------------------------
def prepare_data_for_evaluation_with_perplexity(samples_test, tokenizer, model, answer_length, topk, topp, padToken_id=50256):
    """
    Prepare samples to test a generative model with metrics such as BLEU and Bertscore. The function select al references
    samples available in the dataset. Additionally, this function computes the model perplexity.
    Documentation and examples about metrics are available at
    https://huggingface.co/docs/datasets/v1.5.0/using_metrics.html
    params:
       samples_test: a Dataset with test data.
       tokenizer: the model tokenizer.
       model: the model to test.
    """

    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    # Correct answers
    references = []

    # System generated text
    sys_batch = []

    # Scores of text generated by the model ( as the sum of the logarithms of probability)
    model_scores = []

    # model perplexity
    model_perplexity = []

    references_per_rediction = len(samples_test[0]['answers']['text'])

    for aSample in samples_test:
        
        sample_question = aSample['title'].strip()

        # Reference data or ground truth
        reference_batch = []

        # keep the min amount of answers
        references_per_rediction = references_per_rediction if references_per_rediction < len(aSample['answers']['text']) \
                                                              else len(aSample['answers']['text'])

        # References
        for answer_i in range(len(aSample['answers']['text'])):
            individual_reference = []
            #print("Verdad :", answer_i, aSample['answers']['text'][answer_i])

            # Append the ground truth to the reference_batch
            individual_reference.append(aSample['answers']['text'][answer_i])
            reference_batch.append(individual_reference)


        # Generate and append prediction
        inputs = tokenizer(sample_question, add_special_tokens=False, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attn_masks = inputs["attention_mask"]

        # attention_mask=tokenization.attention_mask
        with torch.no_grad():
            outputs = model.generate(input_ids.to(device), \
                                 max_new_tokens=answer_length,\
                                 pad_token_id = padToken_id, \
                                 do_sample=True, top_k=topk, top_p=topp, \
                                 attention_mask= attn_masks.to(device),
                                 renormalize_logits=True,
                                 return_dict_in_generate=True, output_scores=True)



        model_answer = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        # process the model answer
        sys_batch.append(model_answer)
        references.append(reference_batch)


        # model perplexity
        answer_perplexity = perplexity(model_answer, model, tokenizer)
        model_perplexity.append(answer_perplexity)

    model_perplexity_average = np.average(model_perplexity)

    return references, sys_batch, references_per_rediction, model_scores, \
                  model_perplexity, model_perplexity_average


#---------------------------------------------------------
def prepare_data_for_evaluation(samples_test, tokenizer, model, answer_length, topk, topp, padToken_id=50256):
    """
    Prepare samples to test a generative model with metrics such as BLEU and Bertscore.
    The function select all references samples available in the dataset.
    Documentation and examples about metrics are available at
    https://huggingface.co/docs/datasets/v1.5.0/using_metrics.html
    params:
       samples_test: a Dataset with test data.
       tokenizer_name: the model tokenizer
       model: the model to test.
    return:
       reference_batch: reference data (i.e. ground truth).
       sys_batch: data generated by the model.
       references_per_rediction = min references per prediction
    """

    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    # Correct answers
    references = []

    # System generated text
    sys_batch = []

    references_per_rediction = len(samples_test[0]['answers']['text'])

    for aSample in samples_test:
        sample_question = aSample['title'].strip()

        # Reference data or ground truth
        reference_batch = []

        # keep the min amount of answers
        references_per_rediction = references_per_rediction if references_per_rediction < len(aSample['answers']['text']) \
                                                              else len(aSample['answers']['text'])

        # References
        for answer_i in range(len(aSample['answers']['text'])):
            individual_reference = []

            # Append the ground truth to the reference_batch
            individual_reference.append(aSample['answers']['text'][answer_i])
            reference_batch.append(individual_reference)

        # Generate and append prediction
        inputs = tokenizer(sample_question, add_special_tokens=False, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attn_masks = inputs["attention_mask"]

        # attention_mask=tokenization.attention_mask
        with torch.no_grad():
            outputs = model.generate(input_ids.to(device), \
                                 max_new_tokens=answer_length,\
                                 pad_token_id = padToken_id, \
                                 do_sample=True, top_k=topk, top_p=topp, \
                                 attention_mask= attn_masks.to(device),
                                 renormalize_logits=True,
                                 return_dict_in_generate=True, output_scores=True)



        model_answer = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        # process the model answer
        sys_batch.append(model_answer)
        references.append(reference_batch)

    return references, sys_batch, references_per_rediction



#  ---------------------- Select a group of samples from ELI5 -----------------------
def select_samples(dataset, num_samples_to_retrieve):
    """Select randomly a group of samples from ELI5"""

    # Get the total number of samples in the dataset
    total_samples = len(dataset)

    # Generate a list of 10 unique random indices
    random_indices = random.sample(range(total_samples), num_samples_to_retrieve)

    # Filter rows according to a list of indices
    new_dataset = dataset.select(random_indices)

    return(new_dataset)

#  ---------------------- Select the same_number_of_references_for_each_prediction -----------------------

def same_number_of_references_for_each_prediction (references, number_of_elements):
   """
   BLEU requires as parameter the same number of reference answers for each prediction.
   This fuction takes a list of list with references (ground truths for each answers, more than one) y returns
   the same number of references for each prediction.
   :params
      references: a list of lists of reference answers for a question.
      number_of_elements: required number of answer for each question. The same for all of them.
   :returns
      processed_references
   """
   processed_references = []

   # for each reference select only the first number_of_elements.
   for aReference in references:
       one_group = []

       # number_of_elements
       for i in range(number_of_elements):
           one_group.append(aReference[i])
       processed_references.append(one_group)

   return processed_references


#------------------------------ prepare_data_for_bleu ---------------------------
def prepare_data_for_bleu(samples_test):
    """
    Prepare samples to test a generative model with metrics such as BLEU and Bertscore.
    The function select all references samples available in the dataset.
    Documentation and examples about metrics are available at
    https://huggingface.co/docs/datasets/v1.5.0/using_metrics.html
    params:
       samples_test: a Dataset with test data.

    return:
       reference_batch: reference data (i.e. ground truth).
       references_per_rediction = min references per prediction
    """

    # Correct answers
    references = []

    # Init amount of references per prediction
    references_per_rediction = len(samples_test[0]['answers']['text'])

    for aSample in samples_test:
        sample_question = aSample['title'].strip()

        # Reference data or ground truth
        reference_batch = []

        # keep the min amount of answers
        references_per_rediction = references_per_rediction if references_per_rediction < len(aSample['answers']['text']) \
                                                              else len(aSample['answers']['text'])

        # References
        for answer_i in range(len(aSample['answers']['text'])):
            individual_reference = []
            # Append the ground truth to the reference_batch
            individual_reference.append(aSample['answers']['text'][answer_i])
            reference_batch.append(individual_reference)

        references.append(reference_batch)

    return references, references_per_rediction



#------------------------------ generate_predictions ---------------------------
def generate_predictions (prompt, dropout, num_samples, tokenizer, model, answer_length, topk, topp, padToken_id=50256):
    """ 
    For model type GPT, it receives a model, the defined dropout, a prompt and the number of samples to generate
    and returns a list with the generated samples.
    return:
       predictions
    """

    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    # Set the dropout probability
    model.config.resid_pdrop = dropout  # dropout for all fully connected layers in the embeddings, encoder, and pooler.
    model.config.attn_pdrop  = dropout  # The dropout ratio for the attention.
    model.config.embd_pdrop  = dropout  # The dropout ratio for the embedding.

    # Not using dropout


    # System generated text
    predictions = []

    #sample_question = aSample['title'].strip()

    sample_question = prompt.strip()

    # Tokenize tokens
    inputs = tokenizer(sample_question, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attn_masks = inputs["attention_mask"]

    for i in range(num_samples):

        # attention_mask=tokenization.attention_mask
        with torch.no_grad():
            outputs = model.generate(input_ids.to(device), \
                                 max_new_tokens=answer_length,\
                                 pad_token_id = padToken_id, \
                                 do_sample=True, top_k=topk, top_p=topp, \
                                 attention_mask= attn_masks.to(device),
                                 renormalize_logits=True,
                                 return_dict_in_generate=True, output_scores=True)

        model_answer = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)


        # process the model answer
        predictions.append(model_answer[0])



    return predictions

#---------------------------------------------------------
def generate_predictions_falcon (prompt, dropout, num_samples, tokenizer, model, answer_length, topk, penalty_alpha):
    """
     For model type Falcon, it receives a model, the defined dropout, a prompt and the number of samples to generate
    and returns a list with the generated samples.
    params:
       tokenizer_name: the model tokenizer name
       model: the model to test.
    return:
       predictions
    """

    # Set the dropout probability
    model.config.hidden_dropout  = dropout
    #model.config.attention_dropout  = dropout

    # System generated text
    predictions = []
    
    # bertscore
    precision_list = []
    recall_list = []
    f1_list = []

    sample_question = prompt.strip()
    #print("===============================")
    #print(sample_question)

    # Tokenize tokens
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attn_masks = inputs["attention_mask"]

    for i in range(num_samples):

        with torch.no_grad():                                 
            outputs = model.generate(
    			input_ids.to(device),
    			attention_mask= attn_masks.to(device),
    			max_length=answer_length,
    			do_sample=True,
    			penalty_alpha=penalty_alpha, top_k= topk, # 5.2. Generating Text with Contrastive Search**
    			num_return_sequences=1,
    			#eos_token_id=tokenizer.eos_token_id,
  			)                      
                        #** https://huggingface.co/blog/introducing-csearch
                        
        model_answer = tokenizer.batch_decode(outputs)

        # process the model answer
        predictions.append(model_answer[0])


    return predictions


#------------------------------ generate_unordered_pairs ---------------------------
def generate_unordered_pairs(input_list):
    """
    Receives a list, generates a list of unordered pairs, and stores the head and tail in different lists.
    """
    heads = []
    tails = []
    for i in range(len(input_list)):
        for j in range(i + 1, len(input_list)):
            heads.append(input_list[i])
            tails.append(input_list[j])
    return heads, tails

#------------------------------ compute_uncertainty ---------------------------
def compute_uncertainty(predictions_per_sample, model, tokenizer):
    """Compute the mean prediction and the variance for the group of samples"""

    predictions_confidence = []

    # Compute model perplexity
    for model_answer in predictions_per_sample:
        answer_confindence = perplexity(model_answer, model, tokenizer)
        
        predictions_confidence.append(answer_confindence)
                                        
    mean_confidence = np.mean(np.array(predictions_confidence))

    # Compute the variance
    predictions_square_diference = []

    # All samples versus the rest
    #heads, tails = generate_unordered_pairs(predictions_confidence)
    #for i in range(len(heads)):
    #    for j in range(len(tails)):
    #        diference = (heads[i] - tails[j]) ** 2
    #        predictions_square_diference.append(diference)

    # All samples compared to the mean
    for i in range(len(predictions_confidence)):
        diference = (mean_confidence - predictions_confidence[i])** 2
        predictions_square_diference.append(diference)

    variance_per_sample = np.array(predictions_square_diference).sum()/(len(predictions_confidence)-1)

    return mean_confidence, variance_per_sample, predictions_confidence



def compute_uncertainty_bertscore(predictions_per_sample, model, tokenizer, best_answer_resp, lang):
    """Compute the mean prediction and the variance for the group of samples
    return 
    	mean_confidence: predictions_confidence average
    	variance_per_sample: predictions_confidence variance 
    	predictions_confidence: perplexity list for all predictions_per_sample 
    	f1_list: f1 list for all predictions_per_sample 
    	mean_bertscore_f1: f1 average 
    """

    predictions_confidence = []
    f1_list = []
 
    # Compute model perplexity
    for model_answer in predictions_per_sample:
        answer_confindence = perplexity(model_answer, model, tokenizer)
        
        predictions_confidence.append(answer_confindence)

        #bertscore 
        bert_results = bertscore.compute(predictions=[model_answer], references=[best_answer_resp], \
                                                 lang=lang)
        f1_list.append(bert_results['f1'][0])
                                          
        
    mean_confidence = np.mean(np.array(predictions_confidence))
    mean_bertscore_f1 = np.mean(np.array(f1_list))

    # Compute the variance
    predictions_square_diference = []

    # All samples versus the rest
    #heads, tails = generate_unordered_pairs(predictions_confidence)
    #for i in range(len(heads)):
    #    for j in range(len(tails)):
    #        diference = (heads[i] - tails[j]) ** 2
    #        predictions_square_diference.append(diference)

    # All samples compared to the mean
    for i in range(len(predictions_confidence)):
        diference = (mean_confidence - predictions_confidence[i])** 2
        predictions_square_diference.append(diference)

    variance_per_sample = np.array(predictions_square_diference).sum()/(len(predictions_confidence)-1)

    return mean_confidence, variance_per_sample, predictions_confidence, f1_list, mean_bertscore_f1


#------------------------------ BLEU_score_of_samples ---------------------------
def BLEU_score_of_samples(predictions):
    """
    Return the BLEU score of comparing all samples with each other.
    A number between one and 100.
    """
    heads, tails = generate_unordered_pairs(predictions)

    results = bleu_metric.compute(predictions=heads, references=tails)
    return results['score']


#------------------------------ STD of precision ---------------------------
def BLEU_score_of_samples_std(predictions):
    """
    Return the BLEU precisions of comparing all samples with each other.
    """
    heads, tails = generate_unordered_pairs(predictions)

    results = bleu_metric.compute(predictions=heads, references=tails)
    return np.std(results['precisions'])




#---------------------------------------------------------
def model_avg_conditional_probability (prompt, tokenizer, model, answer_length, topk, topp, padToken_id=50256):
    """
    Sum all conditional probabilities of all generated tokens. It is a number in [0,1].
    params:
       tokenizer_name: the model tokenizer name
       model: the model to test.
       ...
    return:
       predictions
    """
    model.eval()
    sample_question = prompt.strip()

    # Tokenize text
    inputs = tokenizer(sample_question, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attn_masks = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(input_ids.to(device), \
                                 max_new_tokens=answer_length,\
                                 pad_token_id = padToken_id, \
                                 do_sample=True, top_k=topk, top_p=topp, \
                                 attention_mask= attn_masks.to(device),
                                 renormalize_logits=True,
                                 return_dict_in_generate=True, output_scores=True)

    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)

    model_answer = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

    # The average of the scores for each word is calculated.
    # Exp is applied to convert the logarithm to a probability value.
    avg_score = np.average(np.exp(transition_scores[0].cpu()))
    return avg_score, model_answer


#------------------------------ ---------------------------
def calibration (num_bins, dropout, samples_test, num_repetions_per_sample, tokenizer, model, answer_length, \
                 topk, topp, experiment_id, model_type, penalty_alpha, bertscore_model_type, padToken_id=50256):
    """GPT2 Compute the calibration error for a dropp and dropk score and a group of samples.
       Falcon 7b using penalty_alpha and top_k"""


    # compute confidence per sample
    model_confidence_per_sample = []

    # Questions or promts
    original_model_preditions = []

    avg_predictions_per_sample = []

    variance_predictions_per_sample = []

    # for each question
    for aSample in samples_test:
        prompt = aSample['title'].strip()
        best_answer_resp = best_answer(aSample['answers']['text'], aSample['answers']['score'])

        # To compute original model data
        model.eval()
        #############  Average of samples confidence using the model in eval status.

        #++++ Average Conditional Probability
        #avg_score, model_answer = model_avg_conditional_probability (prompt, tokenizer, model, answer_length, topk, topp, padToken_id=50256)

        #++++ Model perplexity (return answer_perplexity, prediction)
        if model_type == "gpt":
       		avg_score, model_answer = compute_perplexity_and_prediction(prompt, tokenizer, model, answer_length,\
                                                                    topk, topp, padToken_id=50256)
        else: # Falcon 
        	avg_score, model_answer, bertscore_f1, bertscore_precision, bertscore_recall = compute_perplexity_and_prediction_falcon(prompt,\
        	 							tokenizer, model, answer_length,\
                                                                    	topk, penalty_alpha, best_answer_resp, bertscore_model_type )
                                                           
        #############

        # append model perplexity score of the original model answer
        model_confidence_per_sample.append(avg_score)

        # Save data in the ECE database
        if model_type == "gpt":
        	insert_data_question(aSample['q_id'], experiment_id, prompt, model_answer, avg_score, topk, topp, \
                                         answer_length, dropout)
        else: # Falcon  
        	insert_data_question_falcon(aSample['q_id'], experiment_id, prompt, model_answer, avg_score, topk, topp, \
                                         answer_length, dropout, bertscore_f1, bertscore_precision, bertscore_recall)
        
                                        
        insert_data_true_answer(aSample['q_id'], aSample['answers'], experiment_id)


        ############ Generate predictions num_repetions_per_sample
        # with dropout
        model.train()


        if model_type == "gpt":
        	predictions_per_sample = generate_predictions (prompt, dropout, num_repetions_per_sample, tokenizer, model, \
                                    answer_length, topk, topp)
        else: # Falcon
        	predictions_per_sample = generate_predictions_falcon (prompt, dropout, num_repetions_per_sample, tokenizer, model, \
                                    			answer_length, topk, penalty_alpha)                                                 

        # Compute average perplexity and variance of each sample
        #average_prediction_per_sample, variance_per_sample, predictions_confidence = \
        #                         compute_uncertainty(predictions_per_sample, model, tokenizer)  # before bertscore
        
        average_prediction_per_sample, variance_per_sample, predictions_confidence, f1_list, mean_bertscore_f1 = \
                                 compute_uncertainty_bertscore(predictions_per_sample, model, tokenizer, best_answer_resp, bertscore_model_type )
        
        
        avg_predictions_per_sample.append(average_prediction_per_sample)
        variance_predictions_per_sample.append(variance_per_sample)

        insert_data_sampling(aSample['q_id'], predictions_per_sample, predictions_confidence, topk, topp,\
                           answer_length, dropout, experiment_id, f1_list)


    model.eval()
    return model_confidence_per_sample, avg_predictions_per_sample, variance_predictions_per_sample



#-----------------------------------------------------------------------------------------------
def compute_perplexity_with_undersampling(num_bins, samples_test, tokenizer, model, answer_length, \
                                                                   topk, topp, padToken_id=50256):
    """
    Computes the model perplexity, distribute the value per bin and apply undesampling
    Documentation and examples about metrics are available at
    https://huggingface.co/docs/datasets/v1.5.0/using_metrics.html
    params:
       samples_test: a Dataset with test data.
       tokenizer_name: the model tokenizer name
       model: the model to test.
    return:
       reference_batch: reference data (i.e. ground truth).
       sys_batch: data generated by the model.
       references_per_rediction = min references per prediction
    """
    # Samples perplexity
    samples_perplexity = []

    # For each question apply sampling
    for aSample in samples_test:
        prompt = aSample['title'].strip()

        # Generate and append prediction
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attn_masks = inputs["attention_mask"]

        # attention_mask=tokenization.attention_mask
        with torch.no_grad():
            outputs = model.generate(input_ids.to(device), \
                                 max_new_tokens=answer_length,\
                                 pad_token_id = padToken_id, \
                                 do_sample=True, top_k=topk, top_p=topp, \
                                 attention_mask= attn_masks.to(device),
                                 renormalize_logits=True,
                                 return_dict_in_generate=True, output_scores=True)

        # model perplexity
        answer_perplexity = perplexity(model_answer, model, tokenizer)
        samples_perplexity.append(answer_perplexity)



    return answer_perplexity, predictions


#-----------------------------------------------------------------------------------------------
def prepare_data_for_calibration_curve(num_bins, calibration_errors_bins, bins_with_data):
    """For a list of numbers and a list of bims_with_data insert zeros into the list
    if the index on the bins_with_data list does not exist. Used for ECE"""
    # Create a vector of zeros
    bins = [0] * num_bins

    # Values to insert
    values_to_insert = calibration_errors_bins

    # Posiciones en las que deseas insertar los valores (0-indexed)
    positions_to_insert = bins_with_data

    # Reemplazar las posiciones específicas con los valores de la lista
    for i in range(len(values_to_insert)):
        if 0 <= positions_to_insert[i] < len(bins):
            bins[positions_to_insert[i]] = values_to_insert[i]

    # Imprimir el vector resultante
    return bins

#-----------------------------------------------------------------------------------------------
def dataframe_undersampling(df, x_column, y_column):
    """Compute undersampling in a dataframe"""

    # Separate features (X) and target labels (y)
    X = df.drop(y_column, axis=1)
    y = df[y_column]
    ros = RandomUnderSampler()
    X_resampled, y_resampled = ros.fit_resample(X, y)

    return X_resampled, y_resampled

#-----------------------------------------------------------------------------------------------
def split_dataframe_best_represented_classes_and_theRest(df, class_name, num_clases):
    """DataFrame into two parts, one containing the two best-represented
    classes and another containing the rest"""

    # Step 1: Calculate the number of instances in each class
    class_counts = df[class_name].value_counts()

    # Step 2: Select the two classes with the highest representation
    top_n_classes = class_counts.nlargest(num_clases).index.tolist()

    # Step 3: Split the DataFrame into two parts the best represented and the rest
    df_best_represented = df[df[class_name].isin(top_n_classes)]
    df_rest = df[~df[class_name].isin(top_n_classes)]

    return df_best_represented, df_rest


#-----------------------------------------------------------------------------------------------
def sort_2related_lists(X, y):
    """Sort 2 list keeping the relation between elements"""

    # Zip the lists together
    zipped_data = list(zip(X, y))

    # Sort the zipped data based on X
    sorted_data = sorted(zipped_data, key=lambda x: x[0])

    # Unzip the sorted data back into separate X and y lists
    X_sorted, y_sorted = zip(*sorted_data)

    return X_sorted, y_sorted

#-----------------------------------------------------------------------------------------------
def list_in_equidistant_classes(numeros, true_values, num_clases):
    """Divide una lista de numeros en clases aquiditantes por el valor de los numeros"""
    # Paso 1: Ordena los números
    #numeros_ordenados = sorted(numeros)
    numeros_ordenados, y_ordenados =  sort_2related_lists(numeros, true_values)

    # Paso 2: Calcular la cantidad de números por clase
    total_numeros = len(numeros_ordenados)
    numeros_por_clase = total_numeros // num_clases

    # Paso 3: Calcular los límites de clase
    limites_clase = [numeros_ordenados[i * numeros_por_clase] for i in range(num_clases)]
    limites_clase.append(numeros_ordenados[-1])  # Agregar el límite superior

    # Paso 4: Asignar los números a las clases
    clases = {i: [] for i in range(num_clases)}
    clases_y = {i: [] for i in range(num_clases)}

    for numero, y_ordenado in zip(numeros_ordenados,y_ordenados) :
        for i in range(num_clases):
            if limites_clase[i] <= numero < limites_clase[i + 1]:
                clases[i].append(numero)
                clases_y[i].append(y_ordenado)
                break

    # sum calss elements
    total = 0
    for i, clase in clases.items():
       total = total + len(clase)

    # el ultimo numero
    if total < len(numeros):
       clases[num_clases-1].append(numeros_ordenados[len(numeros)-1])
       clases_y[num_clases-1].append(y_ordenados[len(numeros)-1])

    # Incluyo las clases en un Dataframe
    dataframes = []
    for i, clase in clases.items():
        df = pd.DataFrame({'bin': [i+1] * len(clase), 'input': clase})
        dataframes.append(df)

    # Concatenar los DataFrames en un solo DataFrame
    result_df = pd.concat(dataframes, ignore_index=True)

    # Incluyo las clases en un segundo Dataframe
    dataframes2 = []
    for i, clase_y in clases_y.items():
        df = pd.DataFrame({'bin': [i+1] * len(clase_y), 'true_values': clase_y})
        dataframes2.append(df)

    # Concatenar los DataFrames en un solo DataFrame
    result_df2 = pd.concat(dataframes2, ignore_index=True)


    return result_df, result_df2



#-----------------------------------------------------------------------------------------------
def scale_two_list(list1, list2):
    """Scale two list wirh diferent scalers """

    # Conact the list to fit the scaler with the same data
    data1 = np.array(list1).reshape(-1, 1)
    data2 = np.array(list2).reshape(-1, 1)

    #Prepare the scaler
    scaler = MinMaxScaler()
    scaler2 = MinMaxScaler()

    #scaled_data = scaler.fit(data1)

    # Transform data
    scaled_data1 = scaler.fit_transform(data1)
    scaled_data2 = scaler2.fit_transform(data2)

    # reshape
    scaled_data1 = np.array(scaled_data1).T[0]
    scaled_data2 = np.array(scaled_data2).T[0]

    # return arrays
    return scaled_data1, scaled_data2



#-----------------------------------------------------------------------------------------------
def scale_two_list_same_scaler(list1, list2):
    """Scale two list with the same scaler"""

    # Conact the list to fit the scaler with the same data
    data3 = list1 +  list2
    data3 = np.array(data3).reshape(-1, 1)

    #Prepare the scaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit(data3)

    # Transform data
    scaled_data3 = scaler.transform(data3)
    #print("scaled_data3", scaled_data3)
    # reshape
    scaled_data = np.array(scaled_data3).T[0]
    scaled_data = scaled_data.tolist()
    #print("scaled_data", scaled_data)

    mitad1 = scaled_data[:len(scaled_data)//2]
    mitad2 = scaled_data[len(scaled_data)//2:]

    # return arrays
    return mitad1, mitad2


#-----------------------------------------------------------------------------------------------
def scale_two_list_different_scaler(list1, list2):
    """Scale two list with diferent scaler instances."""

    data1 = np.array(list1).reshape(-1, 1)
    data2 = np.array(list2).reshape(-1, 1)

    #Prepare the scaler
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()

    scaled_data1 = scaler1.fit_transform(data1)
    scaled_data2 = scaler2.fit_transform(data2)

     # return arrays
    return scaled_data1, scaled_data2
    
#-----------------------------------------------------------------------------------------------
def count_occurrences(lst, number):
    """
    Function that counts the number of occurrences of a given number in a list.

    Parameters:
    - lst: The list in which to search for occurrences.
    - number: The number whose occurrences are to be counted.

    Returns:
    - The number of occurrences of the given number in the list.
    """
    return lst.count(number)

#-----------------------------------------------------------------------------------------------

def count_ocurrences_of_alist(list, list_of_numbers):
    """
    Function that counts the number of occurrences of a given list of numbers in a list.

    Parameters:
    - lst: The list in which to search for occurrences.
    - list_of_numbers: The list of numbers whose occurrences are to be counted.

    Returns:
    - A list with the number of occurrences of the list_of_numbers.
    """

    ocurrences_list = []
    # Use the function to count occurrences for each number in the sequence
    for num in list_of_numbers:
        occurrences = count_occurrences(list, num)
        ocurrences_list.append(occurrences)
    return (ocurrences_list)    

#-----------------------------------------------------------------------------------------------
def sort_2related_lists(X, y):
    """Sort 2 list keeping the relation between elements"""

    # Zip the lists together
    zipped_data = list(zip(X, y))

    # Sort the zipped data based on X
    sorted_data = sorted(zipped_data, key=lambda x: x[0])

    # Unzip the sorted data back into separate X and y lists
    X_sorted, y_sorted = zip(*sorted_data)

    return X_sorted, y_sorted


# ================================ INSERT into the ECE Database ==========================
def insert_data_question(aSample_q_id, experiment_id, prompt, model_answer, perplexity, topk, topp,\
                         answer_length, dropout):
    """Save questions in postgresql table question."""

    aDate = date.today()
    # initialize data of lists.
    data = {'q_id': [aSample_q_id],
            'experiment_id': [experiment_id],
            'experiment_date': [aDate],
            'question': [prompt],
            'model_answer': [model_answer],
            'perplexity':[perplexity],
            'topk': [topk],
            'topp' :[topp],
            'dropout':[dropout],
            'text_length': [answer_length]
           }
    # Create DataFrame
    df = pd.DataFrame(data)
    #Create PySpark DataFrame from Pandas
    sparkDF=spark.createDataFrame(df)
    #sparkDF.show()

    # insert in a database
    insert_spark_df_to_db(sparkDF, 'question')


#-----------------------------------------------------------------------------------------------------------
def insert_data_question_falcon(aSample_q_id, experiment_id, prompt, model_answer, perplexity, topk, topp,\
                         answer_length, dropout, bertscore_f1, bertscore_precision, bertscore_recall):
    """Save questions in postgresql table question_falcon."""

    aDate = date.today()
    # initialize data of lists.
    data = {'q_id': [aSample_q_id],
            'experiment_id': [experiment_id],
            'experiment_date': [aDate],
            'question': [prompt],
            'model_answer': [model_answer],
            'perplexity':[perplexity],
            'topk': [topk],
            'topp' :[topp],
            'dropout':[dropout],
            'text_length': [answer_length],
            'f1': [bertscore_f1],
            'precision': [bertscore_precision],
            'recall': [bertscore_recall]
           }
    # Create DataFrame
    df = pd.DataFrame(data)
    #Create PySpark DataFrame from Pandas
    sparkDF=spark.createDataFrame(df)
    #sparkDF.show()

    # insert in a database
    insert_spark_df_to_db(sparkDF, 'question')


#-----------------------------------------------------------------------------------------------------------
def insert_data_true_answer(aSample_q_id, answers_dict, experiment_id):
    """Save true answers in postgresql table TRUE_ANSWERS."""

    answers_list = answers_dict['text']
    a_list = answers_dict['a_id']
    score_list = answers_dict['score']

    # initialize data of lists.
    data = {'q_id': [aSample_q_id] * len(answers_list),
            'a_id': a_list,
            'score': score_list,
            'true_answer': answers_list,
            'experiment_id': [experiment_id] * len(answers_list)
            }
    # Create DataFrame
    df = pd.DataFrame(data)
    #Create PySpark DataFrame from Pandas
    sparkDF=spark.createDataFrame(df)
    #sparkDF.show()

    # insert in a database
    insert_spark_df_to_db(sparkDF, 'true_answer')

#-----------------------------------------------------------------------------------------
def insert_data_sampling(aSample_q_id, answer_list, perplexity_list, topk, topp,\
                         answer_length, dropout, experiment_id, bertscore_f1_list):
    """Save questions in postgresql table SAMPLING."""

    seq_list = list(range(1,len(answer_list)+1))

    # initialize data of lists.
    data = {'q_id': [aSample_q_id] * len(answer_list),
            'sampling_id': seq_list,
            'answer': answer_list,
            'perplexity':perplexity_list,
            'topk': [topk]* len(answer_list),
            'topp' :[topp]* len(answer_list),
            'dropout':[dropout]* len(answer_list),
            'text_length': [answer_length]  * len(answer_list),
            'experiment_id': [experiment_id]* len(answer_list),
            'bertscore_f1': bertscore_f1_list

           }
    # Create DataFrame
    df = pd.DataFrame(data)
    #Create PySpark DataFrame from Pandas
    sparkDF=spark.createDataFrame(df)
    #sparkDF.show()

    # insert in a database
    insert_spark_df_to_db(sparkDF, 'sampling')

#-----------------------------------------------------------------------------------------
def insert_data_question_sampling(question_sampling_df):
    """Save questions_sampling in postgresql table QUESTION_SAMPLING."""

    sparkDF = spark.createDataFrame(question_sampling_df)
    sparkDF.show()

    # insert in a database (change insert to write to create a table)
    insert_spark_df_to_db(sparkDF, 'question_sampling')


#-----------------------------------------------------------------------------------------
def insert_data_question_sampling_stat_bin (bin_list, mod_perplexity_avg_list, samp_perplexity_avg_list, \
                                            mod_perp_avg_norm, samp_perp_avg_norm, question_count, experiment_id):
    """Save questions_sampling in postgresql table QUESTION_SAMPLING_STAT_BIN."""

    # initialize data of lists.
    data = {'experiment_id': [experiment_id] * len(bin_list),
            'bin': bin_list,
            'mod_perplexity_avg': mod_perplexity_avg_list,
            'samp_perplexity_avg': samp_perplexity_avg_list,
            'mod_perplexity_avg_norm': mod_perp_avg_norm,
            'samp_perplexity_avg_norm': samp_perp_avg_norm,
            'question_count': question_count
           }

    # Create DataFrame
    df = pd.DataFrame(data)
    
    #Create PySpark DataFrame from Pandas
    sparkDF=spark.createDataFrame(df)

    #print('=============== insert_data_question_sampling_stat_bin')
    # insert in the database (change insert to write to create a table)
    insert_spark_df_to_db(sparkDF, 'question_sampling_stat_bin')
    #write_spark_df_to_db(sparkDF, 'question_sampling_stat_bin')



#------------------------------------------------------------------------------------------------
def insert_data_text_generation_tunning(model_list, q_id_list, model_answer_list, seq_list, answer_perplexity_list, penalty_list, \
                        top_k_list, precision_list, recall_list, f1_list, metric_list):
    """Save questions_sampling in postgresql table TEXT_GENERATION_TUNNING."""

    # initialize data of lists.
    data = {'q_id': q_id_list,
            'seq': seq_list,
            'model': model_list,
            'model_answer': model_answer_list,
            'answer_perplexity': answer_perplexity_list,
            'penalty_alpha': penalty_list,
            'top_k': top_k_list,
            'precision': precision_list,
            'recall': recall_list,
            'f1': f1_list,
            'metric': metric_list
           }

    # Create DataFrame
    df = pd.DataFrame(data)
    #Create PySpark DataFrame from Pandas
    sparkDF=spark.createDataFrame(df)

    #print('=============== insert_data')
    #sparkDF.show()

    # insert data in the database, but first create the table (change insert to write to create a table)
    #write_spark_df_to_db(sparkDF, 'text_generation_tunning')
    
    insert_spark_df_to_db(sparkDF, 'text_generation_tunning')


#------------------------------------------------------------------------------------------------
def insert_data_text_generation_dropout_tunning(model_list, q_id_list, model_answer_list, seq_list, answer_perplexity_list, \
                        dropout_list, precision_list, recall_list, f1_list, metric_list):
    """Save questions_sampling in postgresql table TEXT_GENERATION_DROPOUT_TUNNING."""

    # initialize data of lists.
    data = {'q_id': q_id_list,
            'seq': seq_list,
            'model': model_list,
            'model_answer': model_answer_list,
            'answer_perplexity': answer_perplexity_list,
            'dropout': dropout_list,
            'precision': precision_list,
            'recall': recall_list,
            'f1': f1_list,
            'metric': metric_list
           }

    # Create DataFrame
    df = pd.DataFrame(data)
    #Create PySpark DataFrame from Pandas
    sparkDF=spark.createDataFrame(df)

    # insert data in the database, but first create the table (change insert to write to create a table)
    #write_spark_df_to_db(sparkDF, 'TEXT_GENERATION_DROPOUT_TUNNING')
    
    insert_spark_df_to_db(sparkDF, 'TEXT_GENERATION_DROPOUT_TUNNING')





#------------------------------------------------------------------------------------------------
def insert_data_question_bertscore(q_id_list, experiment_id_list, bertscore_f1_list, bertscore_precision_list, bertscore_recall_list):
    """Save data in postgresql table question_bertscore."""

    # initialize data of lists.
    data = {'q_id': q_id_list,
            'experiment_id': experiment_id_list,  
            'bertscore_precision': bertscore_precision_list,
            'bertscore_recall': bertscore_recall_list,
            'bertscore_f1': bertscore_f1_list,
           }

    # Create DataFrame
    df = pd.DataFrame(data)
    #Create PySpark DataFrame from Pandas
    sparkDF=spark.createDataFrame(df)

    # insert data in the database, but first create the table (change insert to write to create a table)
    #write_spark_df_to_db(sparkDF, 'QUESTION_BERTSCORE')
    
    insert_spark_df_to_db(sparkDF, 'QUESTION_BERTSCORE')
    
    
#------------------------------------------------------------------------------------------------
def insert_data_sampling_bertscore(q_id_list, sampling_id_list, experiment_id_list, bertscore_f1_list, bertscore_precision_list, bertscore_recall_list):
    """Save data in postgresql table sampling_bertscore."""

    # initialize data of lists.
    data = {'q_id': q_id_list,
            'sampling_id': sampling_id_list,
            'experiment_id': experiment_id_list,  
            'bertscore_precision': bertscore_precision_list,
            'bertscore_recall': bertscore_recall_list,
            'bertscore_f1': bertscore_f1_list,
           }

    # Create DataFrame
    df = pd.DataFrame(data)
    #Create PySpark DataFrame from Pandas
    sparkDF=spark.createDataFrame(df)

    # create the table (change insert to write to create a table)
    # write_spark_df_to_db(sparkDF, 'SAMPLING_BERTSCORE')
    
    insert_spark_df_to_db(sparkDF, 'SAMPLING_BERTSCORE')
    

#========================== postprocessing_tasks===================================

def select_question_sampling(experiment_id):
   """
   Data are selected from QUESTION and SAMPLING filtering by experiment_id
        (select q.q_id, q.experiment_id, q.perplexity mod_perplexity,
                 s.sampling_id, s.perplexity as samp_perplexity
            from question q, sampling s
            where trim(q.q_id) = trim(s.q_id)  and
                q.experiment_id = s.experiment_id and  q.experiment_id =6
                    order by 1,3)
   """
   #query_text = "select q.q_id, q.experiment_id, q.perplexity mod_perplexity, \
   #              s.sampling_id, s.perplexity as samp_perplexity \
   #                 from question q, sampling s \
   #                 where trim(q.q_id) = trim(s.q_id)  and \
   #                 q.experiment_id = s.experiment_id and \
   #                 bertscore_f1 > 0.8 and \
   #                 q.experiment_id =" + str(experiment_id) + "order by 1,3" 

   query_text = "select q.q_id, q.experiment_id, q.perplexity mod_perplexity, \
                 s.sampling_id, s.perplexity as samp_perplexity \
                    from question q, sampling s \
                    where trim(q.q_id) = trim(s.q_id)  and \
                    q.experiment_id = s.experiment_id and \
                    q.experiment_id =" + str(experiment_id) + "order by 1,3" 


   # read database
   spark_df = read_dataset_from_db(query_text)
   
   return spark_df

#--------------------------------------------------------------------------------------
def select_true_answer_with_perplexity(q_id):
   """
   Data are selected from true_answer_with_perplexity filter by question id.
   """
   query_text = 'select true_answer from true_answer_with_perplexity \
                    where q_id = ' + q_id +' order by score desc' 

   # read database
   spark_df = read_dataset_from_db(query_text)
   
   return spark_df
   
#--------------------------------------------------------------------------------------
def select_question(experiment_id):
   """
   Data are selected from table ECE.Question filter by question id (q_id).
   """
   query_text = 'select q_id, experiment_id, model_answer from question \
                    where experiment_id = ' + str(experiment_id) 

   # read database
   spark_df = read_dataset_from_db(query_text)
   
   return spark_df
   
#--------------------------------------------------------------------------------------
def select_sampling(experiment_id):
   """
   Data are selected from sampling filter by question id (q_id).
   """
   query_text = 'select q_id, sampling_id, experiment_id, answer from sampling \
                    where experiment_id = ' + str(experiment_id) 

   # read database
   spark_df = read_dataset_from_db(query_text)
   
   return spark_df   
      
#--------------------------------------------------------------------------------------
def best_true_answer(q_id):
   """
   Return the best answer for a question using the score (field of the dataset).
   """

   # read data from ECE database
   spark_df = select_true_answer_with_perplexity(q_id)

   df = spark_df.toPandas()

   best_answer = df['true_answer'].tolist()
   
   return best_answer[0]


#--------------------------------------------------------------------------------------
def generate_question_bertscore(experiment_id):
   """
   For all question in a each experiment_id it generates the bertscore and save data in the database (question_bertscore).
   """

   # read data from ECE database

   spark_df = select_question(experiment_id)

   df = spark_df.toPandas()

   #lists to store data
   q_id_list = []
   experiment_id_list = []
   bertscore_f1_list = []
   bertscore_precision_list = []
   bertscore_recall_list = []

   # For each record compute bertscore
   for index, row in df.iterrows():

        question_id = '\'' + row['q_id'] + '\''        
        print(question_id)
        
        best_answer= best_true_answer(question_id)
       
        bert_results = bertscore.compute(predictions=[row['model_answer']], references=[best_answer],\
                   lang='en')
        
        #store data in lists
        q_id_list.append(row['q_id'])
        experiment_id_list.append(row['experiment_id'])
        bertscore_f1_list.append(bert_results['f1'][0])
        bertscore_precision_list.append(bert_results['precision'][0])
        bertscore_recall_list.append(bert_results['recall'][0])
        
   insert_data_question_bertscore(q_id_list, experiment_id_list, bertscore_f1_list, bertscore_precision_list, bertscore_recall_list)
   
#--------------------------------------------------------------------------------------
def generate_sampling_bertscore(experiment_id):
   """
   For all samplings in a each experiment_id it generates the bertscore and save data in the database (sampling_bertscore).
   """

   # read data from ECE database

   spark_df = select_sampling(experiment_id)

   df = spark_df.toPandas()

   #lists to store data
   q_id_list = []
   sampling_id_list = []
   experiment_id_list = []
   bertscore_f1_list = []
   bertscore_precision_list = []
   bertscore_recall_list = []
  
   # For each record compute bertscore
   for index, row in df.iterrows():

        question_id = '\'' + row['q_id'] + '\''        
        print(question_id)
        
        best_answer= best_true_answer(question_id)
       
        bert_results = bertscore.compute(predictions=[row['answer']], references=[best_answer],\
                   lang='en')
        
        #store data in lists
        q_id_list.append(row['q_id'])
        sampling_id_list.append(row['sampling_id'])
        experiment_id_list.append(row['experiment_id'])
        bertscore_f1_list.append(bert_results['f1'][0])
        bertscore_precision_list.append(bert_results['precision'][0])
        bertscore_recall_list.append(bert_results['recall'][0])
        
        if (index % 1000 == 0): 
        
           insert_data_sampling_bertscore(q_id_list, sampling_id_list, experiment_id_list, bertscore_f1_list, bertscore_precision_list, bertscore_recall_list)
           #lists to store data
           q_id_list = []
           sampling_id_list = []
           experiment_id_list = []
           bertscore_f1_list = []
           bertscore_precision_list = []
           bertscore_recall_list = []
           print("entre")
   # last group         
   insert_data_sampling_bertscore(q_id_list, sampling_id_list, experiment_id_list, bertscore_f1_list, bertscore_precision_list, bertscore_recall_list) 


#--------------------------------------------------------------------------------------
def postprocesing_normalizing_binding(experiment, num_bins):
   """Read a files form disk, normalize perplexity and divide into ranges
      (assing each sample to a bind). Records are saved into QUESTION_SAMPLING (DB ECE)


   Process:
   1) Read database
       Data are selected from QUESTION and SAMPLING filtering by experiment_id
        \copy (select q.q_id, q.experiment_id, q.perplexity mod_perplexity,
                 s.sampling_id, s.perplexity as samp_perplexity
            from question q, sampling s
            where trim(q.q_id) = trim(s.q_id)  and
                q.experiment_id = s.experiment_id and  q.experiment_id =2
                    order by 1,3) to '/tmp/ECE_Exp2.csv' with csv HEADER
   2) Distribute samp_perplexity_norm in bins or clases,
   bins = np.linspace(0, 1, 11)
   print(bins)
   [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
   Classes example [0.  0.1) , [0.1 0.2), etc

   Element in class 11 is moved to class 10.
   The process is sensitive to atipycal maximun.

   """

   #read file
   #df = pd.read_csv('/home/mmora/Documents/Models/LLM/datos_ECE/ECE_Exp' + str(experiment)+'.csv')

   # read data from ECE database
   spark_df = select_question_sampling(experiment)
   df = spark_df.toPandas()

   list1 = df['mod_perplexity'].tolist()
   list2 =  df['samp_perplexity'].tolist()


   # Normalize 2 lists of the same size
   mod_perp_norm, samp_perp_norm = scale_two_list_different_scaler(list1, list2)

   df['mod_perplexity_norm']  = mod_perp_norm
   df['samp_perplexity_norm'] = samp_perp_norm

   #
   # Bin the model normalized perplexity
   bins = np.linspace(0, 1, num_bins)
   bin_indices = np.digitize(np.array(mod_perp_norm), bins)

   # The maximum normalized value is 1 and it is in bin 11, it must be moved to 10
   # change the element from class 11 to class 10
   value_to_replace = num_bins
   replacement_value = num_bins-1

   bin_indices[bin_indices==value_to_replace] = replacement_value

   # Add the bin column to the dataframe
   df['bin'] =  bin_indices

   return (df)


#--------------------------------------------------------------------------------------

def postprocessing_tasks_QUESTION_SAMPLING_STAT_BIN(experiment_id):
    """Normaliza data and compute statistics (sample_perplexity_avg, sample_perplexity_var) reading data from the DB and save result to the db.
    Parameters:
      query_text: The querry to extract data from the DB.

    """
    query_text = "select q.bin, avg(q.mod_perplexity) as mod_perplexity_avg, avg(q.samp_perplexity) as samp_perplexity_avg \
                                FROM question_sampling q \
                                WHERE q.experiment_id = " + str(experiment_id) + " group by bin order by bin"

    query_text2 = "SELECT bin, COUNT(*) AS question_count \
                  FROM (SELECT distinct q_id, bin FROM question_sampling q \
                           WHERE q.experiment_id = " + str(experiment_id)  + " group by 1,2) AS subquery group by bin order by bin"
                           

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    # read database
    spark_df = read_dataset_from_db(query_text)

    df = spark_df.toPandas()

    mod_perplexity_avg_list = df["mod_perplexity_avg"].tolist()
    samp_perplexity_avg_list = df['samp_perplexity_avg'].tolist()

    # Normalize 2 lists of the same size, return 2 lists
    mod_perp_avg_norm, samp_perp_avg_norm = scale_two_list_same_scaler(mod_perplexity_avg_list, samp_perplexity_avg_list)

    #BIN list 
    bin_list = df["bin"].tolist()
    
    # Count amount of quetions by bin
    spark_df2 = read_dataset_from_db(query_text2)
    df2 = spark_df2.toPandas()
    
    question_count = df2["question_count"]
 
    insert_data_question_sampling_stat_bin(bin_list, mod_perplexity_avg_list, samp_perplexity_avg_list, \
                                           mod_perp_avg_norm, samp_perp_avg_norm,  question_count, experiment_id )



#------------------------------ ---------------------------

def compute_mean_variance_from_perplexity(predictions_confidence):
    """Compute the mean prediction and the variance for the group of samples

    params:
       predictions_confidence: a np.array of confindence associated with a sample
    """

    mean_confidence = np.mean(predictions_confidence)

    # Compute the variance
    predictions_square_diference = []

    # All samples compared to the mean
    for i in range(len(predictions_confidence)):
        diference = (mean_confidence - predictions_confidence[i])** 2
        predictions_square_diference.append(diference)

    variance_confidence = np.array(predictions_square_diference).sum()/(len(predictions_confidence)-1)

    return mean_confidence, variance_confidence


#---------------------------------------------------------

def postprocessing_tasks_QUESTION_SAMPLING(experiment_id, num_bins):
    """
    For an experiment_id update database ECE
       Read data form QUESTION and SAMPLING tables, normalize perplexity and divide it into ranges.
       Save data in QUESTION_SAMPLING
    pamars:
        experiment_id: a sequence to identify an experiment. 
        num_bins: 
    """

    question_sampling_df = postprocesing_normalizing_binding(experiment_id, num_bins)

    insert_data_question_sampling(question_sampling_df)
     
    
#---------------------------------------------------------
def select_question_sampling_stat_morethanone_experiment_bin(query_text):
   """
   Compute the average per question_id of a group of experiment to compute ECE.
   params:
     query_text: the query to be sen to the database. 
   """

   # read database
   spark_df = read_dataset_from_db(query_text)
   df = spark_df.toPandas()

   mod_perplexity_avg_list = df["mod_perplexity_avg_norm"].tolist()
   samp_perplexity_avg_list = df['samp_perplexity_avg_norm'].tolist()
   question_count_list = df['question_count'].tolist()
   bin_list = df['bin'].tolist()

   return mod_perplexity_avg_list , samp_perplexity_avg_list, bin_list, question_count_list


#---------------------------------------------------------
def select_question_sampling_stat_morethanone_experiment(query_text):
   """
   Compute the average per question_id of a group of experiment to compute ECE.
   params:
     query_text: the query to be sen to the database. 
   """

   # read database
   spark_df = read_dataset_from_db(query_text)
   df = spark_df.toPandas()

   mod_perplexity_avg_list = df["mod_perplexity_avg"].tolist()
   samp_perplexity_avg_list = df['samp_perplexity_avg'].tolist()
   q_id_list = df['q_id'].tolist()

   return mod_perplexity_avg_list , samp_perplexity_avg_list, q_id_list 
  
#---------------------------------------------------------
def select_question_sampling_stat_morethanone_experiment_bertscore(query_text):
   """
   Compute the average per question_id of a group of experiment to compute ECE.
   params:
     query_text: the query to be sen to the database. 
   """

   # read database
   spark_df = read_dataset_from_db(query_text)
   df = spark_df.toPandas()

   mod_bertscore_avg_list = df["question_bertscore_f1_avg"].tolist()
   samp_bertscore_avg_list = df['sampling_bertscore_f1_avg'].tolist()
   q_id_list = df['q_id'].tolist()
   
   return mod_bertscore_avg_list , samp_bertscore_avg_list, q_id_list


#-----------------------------------------------------------------------------------------
def insert_data_question_sampling_avg(q_id_list, experiment_id_list, mod_perplexity_list, samp_perplexity_avg,\
                                      samp_perplexity_var,  bin_list):
    """Save questions_sampling in postgresql table QUESTION_SAMPLING_STAT."""

    # initialize data of lists.
    data = {'q_id': q_id_list,
            'experiment_id': experiment_id_list,
            'mod_perplexity': mod_perplexity_list,
            'samp_perplexity_avg': samp_perplexity_avg,
            'samp_perplexity_var': samp_perplexity_var,
            'bin': bin_list
           }

    # Create DataFrame
    df = pd.DataFrame(data)
    #Create PySpark DataFrame from Pandas
    sparkDF=spark.createDataFrame(df)

    # insert in the database (change insert to write to create a table)
    insert_spark_df_to_db(sparkDF, 'question_sampling_stat')


#--------------------------------------------------------------------------------------

def postprocessing_tasks_QUESTION_SAMPLING_STAT_undersampling(query_text, num_samples):
    """Compute statistics (sample_perplexity_avg, sample_perplexity_var) reading data from the DB and save result to the db.
    Parameters:
      query_text: The querry to extract data from the DB.
      num_bins : bins
    """
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    # read database
    spark_df = read_dataset_from_db(query_text)

    # extract distinct question id
    question_list = spark_df.select("q_id",'experiment_id','mod_perplexity_norm', 'bin' ).dropDuplicates()
    question_df = question_list.toPandas()

    #lists to store data
    q_id_list = []
    experiment_id_list = []
    mod_perplexity_list = []
    samp_perplexity_avg = []
    samp_perplexity_var = []
    bin_list = []

    # For each question compute sample_perplexity_avg and sample_perplexity_var
    for index, row in question_df.iterrows():

        # select samples for row['q_id']
        q_idSampling = spark_df.select("samp_perplexity_norm").filter(col("q_id") == row['q_id']).toPandas()
        
        # select a group of num_samples
        q_idSampling = random.sample(q_idSampling["samp_perplexity_norm"].tolist(), num_samples)

        # compute mean and variance from samp_perplexity_norm
        mean_confidence, variance_confidence = compute_mean_variance_from_perplexity(\
                                                        np.array(q_idSampling))

        q_id_list.append(row['q_id'])
        experiment_id_list.append(row['experiment_id'])
        mod_perplexity_list.append(row['mod_perplexity_norm'])
        samp_perplexity_avg.append( mean_confidence)
        samp_perplexity_var.append(variance_confidence)
        bin_list.append(row['bin'])


    insert_data_question_sampling_avg(q_id_list, experiment_id_list, mod_perplexity_list, samp_perplexity_avg,\
                                      samp_perplexity_var,  bin_list)



