from vllm import LLM, SamplingParams
import time
from vllm.utils import FlexibleArgumentParser
from vllm.engine.arg_utils import EngineArgs
import random
import numpy as np

from vllm.transformers_utils.tokenizer import get_tokenizer
from transformers import AutoTokenizer


# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

def main(args): 
    # print(args)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # engine_args = EngineArgs.from_cli_args(args)
    # llm = LLM(**dataclasses.asdict(engine_args))

    # model_id = args.model
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    # tokenizer_mode = args.tokenizer_mode
    # tokenizer = get_tokenizer(tokenizer_id,
    #                           tokenizer_mode=tokenizer_mode,
    #                           trust_remote_code=args.trust_remote_code)

    # NOTE(HONG): from HiP code
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    batch_size = 1
    print(f"HONG: args.input_len: {args.input_len}")
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(batch_size,
                                                     args.input_len))
    dummy_prompts: List[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    # print(f"dummy_prompts: {dummy_prompts}")

    # Create a sampling params object.
    print(args.max_tokens)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=args.max_tokens)

    # Create an LLM.
    llm = LLM(model=model_id, enforce_eager=True, enable_chunked_prefill=False, max_model_len=args.max_tokens+args.input_len) # max_model_len = len(prompt) + len(output)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    start_time = time.time()
    outputs = llm.generate(dummy_prompts, sampling_params)
    # outputs = llm.generate(sample_input, sampling_params)
    end_time = time.time()
    print(f"inference time {end_time - start_time}")
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_text = generated_text.replace("\n", " \\n")
        print(f"Prompt: {prompt!r}, \nGenerated text: {generated_text!r}")
        print(f"Generated token length: {len(output.outputs[0].token_ids)}")

TEMPLATE = '''<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Hi, can you describe about following document? Here is document, 

```
{document}
```

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="offloading test.")    
    parser.add_argument("--max-tokens",
                        type=int,
                        default=32,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    # parser.add_argument('--input', default='../samples/2k.md', type=str)
    parser.add_argument('--input-len', type=int, default=32)
    # parser.add_argument('--model', default='meta-llama/Meta-Llama-3.1-8B-Instruct', type=str)
    # parser.add_argument(
    #     '--tokenizer-mode',
    #     type=str,
    #     default="auto",
    #     choices=['auto', 'slow', 'mistral'],
    #     help='The tokenizer mode.\n\n* "auto" will use the '
    #     'fast tokenizer if available.\n* "slow" will '
    #     'always use the slow tokenizer. \n* '
    #     '"mistral" will always use the `mistral_common` tokenizer.')
    # parser.add_argument(
    #     "--trust-remote-code",
    #     action="store_true",
    #     help="Trust remote code from huggingface",
    # )

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()    

    # with open(args.input, 'r') as f:
    #         document = f.read()

    # sample_input = TEMPLATE.format(document=document)

    main(args)
    # main(args, sample_input)