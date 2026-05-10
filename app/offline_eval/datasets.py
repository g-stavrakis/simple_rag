from dotenv import load_dotenv
from langsmith import Client


load_dotenv()

# Initialize LangSmith client
client = Client()
project_name = "simple-hr-assistant"
dataset_name = "simple-hr-assistant-offline-eval"
retrieve_dataset_name = "simple-hr-assistant-retrieve-offline-eval"


# Create dataset in LangSmith
dataset = client.create_dataset(
    dataset_name,
    description="Dataset created from successful root runs.",
)

# Filter traces (root runs) to add to the dataset
runs = client.list_runs(
    project_name=project_name,
    is_root=True,
    error=False,
)

# Prepare inputs and outputs for bulk creation
examples = [
    {"inputs": run.inputs, "outputs": run.outputs}
    for run in runs
    if run.outputs and run.outputs.get("answer")
]

# Use the bulk create_examples method
client.create_examples(
    dataset_id=dataset.id,
    examples=examples,
)

print(f"Created dataset {dataset.name} with {len(examples)} examples.")


# Create dataset in LangSmith
retrieve_dataset = client.create_dataset(
    retrieve_dataset_name,
    description="Dataset created from successful retrieve node runs.",
)

# Filter retrieve node runs to add to the dataset
retrieve_runs = client.list_runs(
    project_name=project_name,
    is_root=False,
    error=False,
)

# Prepare inputs and outputs for bulk creation
retrieve_examples = [
    {
        "inputs": {
            "question": run.inputs.get("question"),
            "query": run.inputs.get("rewritten_question") or run.inputs.get("question"),
            "category": run.inputs.get("category"),
        },
        "outputs": {
            "sources": run.outputs.get("sources", []),
        },
    }
    for run in retrieve_runs
    if run.name == "retrieve" and run.outputs
]

# Use the bulk create_examples method
client.create_examples(
    dataset_id=retrieve_dataset.id,
    examples=retrieve_examples,
)

print(
    f"Created dataset {retrieve_dataset.name} with {len(retrieve_examples)} examples."
)
