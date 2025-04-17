from vllm import LLM, SamplingParams
import time
from vllm.utils import FlexibleArgumentParser
from vllm.engine.arg_utils import EngineArgs
import random
import numpy as np
import sys
from vllm.transformers_utils.tokenizer import get_tokenizer
from transformers import AutoTokenizer
from vllm.logger import init_logger
logger = init_logger("vllm")

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

def main(args, sample_input): 
    # print(args)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # engine_args = EngineArgs.from_cli_args(args)
    # llm = LLM(**dataclasses.asdict(engine_args))

    # model_id = args.model
    # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    model_id = "/home/jongseop/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" # absolute path
    
    # tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    # tokenizer_mode = args.tokenizer_mode
    # tokenizer = get_tokenizer(tokenizer_id,
    #                           tokenizer_mode=tokenizer_mode,
    #                           trust_remote_code=args.trust_remote_code)

    # NOTE(HONG): from HiP code
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Create a sampling params object.
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=args.max_tokens)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=5500, # max_token + prompt < 11000 HARDCODE
                                     stop=[],stop_token_ids=[], ignore_eos=True) # force generation to max_tokens 

    # Create an LLM.
    """
    Total GPU memory = model size + torch activation peak memory + non-torch memory + KV size 
    
    For 0% offloading: GPU memory > KV size 
    gpu_memory_utilization = (model size + max_model_len * kv entry size) / 47.54GiB
    
    For Dist1 (100% offloading): GPU memory > KV size / # layers (at least enough for one layer)
    gpu_memory_utilization = (model size + max_model_len * kv entry size / # layers ) / 47.54GiB
    
    For Dist2 (50% offloading): GPU memory > KV size * 0.5
    gpu_memory_utilization = (model size + max_model_len * kv entry size * 0.5 ) / 47.54GiB
    
    For Dist3 (33% offloading): GPU memory > KV size * 0.33
    gpu_memory_utilization = (model size + max_model_len * kv entry size * 0.33 ) / 47.54GiB
    ... 
    
    For Llama-3-8B: 
    Dist0: gpu_memory_utilization = (17.15GB + max_model_len * 4KiB * 32 layers) / 47.54GiB
    Dist1 offloading: gpu_memory_utilization = (17.15GB + max_model_len * 4Kib) / 47.54GiB
    Dist2 offloading: gpu_memory_utilization = (17.15GB + max_model_len * 4Kib * 16 layers) / 47.54GiB
    Dist3 offloading: gpu_memory_utilization = (17.15GB + max_model_len * 4Kib * 10 layers) / 47.54GiB
    """
    max_model_len = 13000
    distn = 2
    kv_entry_size = 4/1024/1024
    num_layers = 32
    constant_gpu_mem = 15.08 # Xinyue for 8B model 
    total_gpu_mem = 47.54
    activation = 0.1063 * max_model_len / 1024
    gpu_memory_utilization = (constant_gpu_mem + activation + max_model_len * kv_entry_size * num_layers/distn) / total_gpu_mem
    msg = (f"Dist{distn} offloading, max_model_len:{max_model_len}, gpu_mem% = {gpu_memory_utilization}, kv_mem% = {gpu_memory_utilization - constant_gpu_mem/total_gpu_mem}"
           f"Seq len without prefetching: {max_model_len/distn}")
    logger.info(msg)
    llm = LLM(model=model_id, enforce_eager=True, enable_chunked_prefill=False, max_model_len=max_model_len, is_monolithic_distn=False, gpu_memory_utilization=gpu_memory_utilization) # max_model_len = len(prompt) + len(output)
    
    
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    
    input_tokens = tokenizer.encode(sample_input)    
    logger.info(f"input_tokens: {len(input_tokens)}")
    start_time = time.time()
    outputs = llm.generate(sample_input, sampling_params)
    end_time = time.time()
    logger.info(f"inference time {end_time - start_time}")
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_text = generated_text.replace("\n", " \\n")
        logger.info(f"Prompt: {prompt!r}, \nGenerated text: {generated_text!r}")
        logger.info(f"Generated token length: {len(output.outputs[0].token_ids)}")
        logger.info(f'Stop finish_reason: {output.outputs[0].finish_reason}')
        # print(f"Prompt: {prompt!r}, \nGenerated text: {generated_text!r}")
        # print(f"Generated token length: {len(output.outputs[0].token_ids)}")

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="offloading test.")
    parser.add_argument("--max-tokens",
                        type=int,
                        default=128000,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument('--input', default='../samples/passkey5k.md', type=str)

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
    parser.add_argument('--output_log', default='./test.log', type=str)
    args = parser.parse_args()    
    output_log = open(args.output_log, 'w')
    print(f"Logging to {args.output_log}")
    sys.stdout = output_log
    sys.stderr = output_log
    with open(args.input, 'r') as f:
            document = f.read()

    sample_input = document
    # sample_input = """A A A A A A A A A A A A A A A A 
    # sample_input = """Hello World!"""
    # sample_input = """List 10 interesting facts about the universe shortly."""
    # sample_input =  """List 1000 interesting facts about the universe."""
    main(args, sample_input)