import os
import json
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import aiohttp

# you need to have a .env file with OPENAI_API_KEY='<your_openai_api_key>'
load_dotenv()
client = AsyncOpenAI()

async def evaluate_completion(completion, criterion, prompt, client, model, verbose=False):
    system_message = "You score texts generated by a language model based on the following criterion: \n"\
        + criterion + ".\nYou provide a score from 1 to 10. \
The language model was given a prompt and generated the following text. \
    Evaluate the text based on the criterion. Output format should be JSON with the following fields: \"score\" (int)"
    if verbose:
        system_message += " and \"reason\""

    response = await client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Prompt:\n\n" + prompt + "\n\nCompletion:\n\n" + completion}
        ],
        max_tokens=150,
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)


def evaluate_completions(
        completions: list[str],
        criterion: str,
        prompt: str,
        model="gpt-3.5-turbo", # "gpt-4o"
        verbose=False,
        ):
    async def evaluate_all():
        tasks = [
            evaluate_completion(completion, criterion, prompt, client, model, verbose=verbose)
            for completion in completions
        ]
        return await asyncio.gather(*tasks)

    return asyncio.run(evaluate_all())
        

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
