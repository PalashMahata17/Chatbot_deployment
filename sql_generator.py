import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

def create_prompt(question, schema_str):
    """Creates the prompt for the SQL generation model."""
    prompt = f"""
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{schema_str}

### Answer
Given the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION]
[SQL]
"""
    return prompt

class SQLGenerator:
    """
    A class to handle loading the SQLCoder model and generating SQL queries.
    """
    def __init__(self, model_name="defog/sqlcoder-7b-2", save_dir="model"):
        """
        Initializes the SQLGenerator by loading the tokenizer and model,
        and optionally saving them to a directory.
        """
        self.save_dir = save_dir

        # Check if model and tokenizer are already saved locally
        if os.path.exists(self.save_dir):
            print(f"...Loading Tokenizer from {self.save_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
            print("...Tokenizer Loaded locally.")

            print(f"...Loading Model from {self.save_dir}...")
            self.model = AutoModelForCausalLM.from_pretrained(self.save_dir)
            print("...Model Loaded locally!...")

        else:
            print("...Local model not found. Loading from Hugging Face Hub...")
            print("...Loading Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("...Tokenizer Loaded.")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            print("...Loading Model (this will take a few minutes)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            print("...Model Loaded!...")

            # Save the model and tokenizer for future use
            print(f"...Saving model and tokenizer to {self.save_dir}...")
            self.tokenizer.save_pretrained(self.save_dir)
            self.model.save_pretrained(self.save_dir)
            print("...Model and tokenizer saved.")


    def generate_sql(self, question, schema_str):
        """
        Generates a SQL query based on the question and schema.
        """
        print("...Building prompt...")
        prompt = create_prompt(question, schema_str)

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        print("...Model is 'thinking'...")
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            num_beams=4,
        )

        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        try:
            sql_query = output.split("[SQL]")[1]
        except IndexError:
            print("Error: Could not parse the model's output.")
            return "I messed up, sorry!"

        sql_query = sql_query.replace("```sql", "").replace("```", "").strip().strip(";") + ";"

        return sql_query
