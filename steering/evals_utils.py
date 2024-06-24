import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import nest_asyncio

# Allows running asyncio in jupyter
nest_asyncio.apply()

# Load environment variables
load_dotenv()
client = AsyncOpenAI()

async def evaluate_completion(completion, criterion, prompt, client, model, verbose=False):
    system_message = "You score texts generated by a language model based on the following criterion: \n"\
        + criterion + ".\nYou provide a score from 1 to 10. \
The language model was given a prompt and generated the following text. \
    Evaluate the text based on the criterion. Output format should be JSON with the following fields: \"score\" (int)"
    if verbose:
        system_message += " and \"reason\""
    
    try:
        response = await client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Prompt:\n\n{prompt}\n\nCompletion:\n\n{completion}"}
            ],
            max_tokens=150,
            temperature=0.0,
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Text causing the error:\n{content}")
        return {"score": 0, "error": "Failed to parse JSON response"}
    except Exception as e:
        print(f"Error in evaluate_completion: {e}")
        return {"score": 0, "error": str(e)}

def evaluate_completions(
        completions: list[str],
        criterion: str,
        prompt: str,
        model="gpt-3.5-turbo", # gpt-4o
        verbose=False,
        ):
    async def evaluate_all():
        tasks = [
            evaluate_completion(completion, criterion, prompt, client, model, verbose=verbose)
            for completion in completions
        ]
        return await asyncio.gather(*tasks)
    
    return asyncio.run(evaluate_all())


def multi_criterion_evaluation(
        completions: list[str],
        criterions: list[str],
        prompt: str,
        model="gpt-3.5-turbo",
        verbose=False,
        ):
    repeated_completions = completions * len(criterions)
    repeated_criteria = []
    for criterion in criterions:
        repeated_criteria.extend([criterion] * len(completions))
    
    async def evaluate_all():
        tasks = [
            evaluate_completion(completion, criterion, prompt, client, model, verbose=verbose)
            for completion, criterion in zip(repeated_completions, repeated_criteria)
        ]
        return await asyncio.gather(*tasks)
    
    results = asyncio.run(evaluate_all())
    
    # Reshape results into a 2D list: [completion][criterion]
    reshaped_results = [results[i:i+len(criterions)] for i in range(0, len(results), len(criterions))]
    
    # Filter out error responses
    filtered_results = []
    for completion_results in reshaped_results:
        if not any("error" in result for result in completion_results):
            filtered_results.append(completion_results)
    
    # Transpose the filtered_results to group by criterion
    transposed_results = list(map(list, zip(*filtered_results)))
    
    return transposed_results

# Example usage
if __name__ == "__main__":
    prompt = "Once upon a time,"
    evaluation_criterion = "humor and lightheartedness"
    completions = [
        "Robo tried dancing. It was clumsy but got better. Everyone laughed.",
        "Robo joined its owners dancing. It was stiff but made them laugh.",
    ]
    evaluations = evaluate_completions(completions, evaluation_criterion, prompt)
    for i, evaluation in enumerate(evaluations):
        print(f"Completion {i+1} Evaluation:\n{evaluation}\n")