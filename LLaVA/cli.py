import argparse
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
from datasets import load_from_disk
from tqdm import tqdm

def load_image(image_file):
    if isinstance(image_file, Image.Image):
        image = image_file.convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def main(args):
    # Disable Torch initialization for faster loading
    disable_torch_init()

    # Load pretrained model
    model_name = get_model_name_from_path(args.model_path)
    print("Loading model...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device
    )

    # Determine conversation mode based on model name
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    # Load the dataset
    print("Loading dataset...")
    # dataset = load_dataset("abhinavl/figure2code_data", split="test")
    dataset = load_from_disk("local_figure2code_data")

    # Define the fixed question for all images
    fixed_question = (
        "Look at the input figure, then give me the full code that would create the same figure. Here is an example of the expected output:\n"
        "import matplotlib.pyplot as plt\n"
        "categories = ['essay', 'soil']\n"
        "values = [1, 2]\n"
        "plt.figure(figsize=(8, 5))  # Set the figure size (optional)\n"
        "plt.bar(categories, values, color='skyblue')  # Plot the bars with skyblue color\n"
        "plt.title('Title')  # Add a title to the chart\n"
        "plt.xlabel('Categories')  # Label for the x-axis\n"
        "plt.ylabel('Values')  # Label for the y-axis\n"
        "plt.show()"
    )

    # Create output directory if it doesn't exist
    output_dir = "inference_output"
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the dataset with a progress bar

    # for idx, item in tqdm(enumerate(dataset), total=2, desc="Processing images"):
    #     if idx >= 2:
    #         break
    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Processing images"):
        image = load_image(item['image'])
        image_size = image.size

        # Process image tensor
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        # Reset conversation template for each image
        conv = conv_templates[args.conv_mode].copy()
        roles = ('user', 'assistant') if "mpt" in model_name.lower() else conv.roles

        # Prepare the input with the fixed question
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + fixed_question

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Debugging print statements

        # Tokenize input and generate response
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            try:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True
                )
            except IndexError as e:
                print(f"Error encountered: {e}")
                print(f"Image tensor shape: {image_tensor.shape}")
                raise e

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        # Print output if in debug mode
        # if args.debug:
        #     print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        # else:
        #     print(f"{roles[1]}: {outputs}")

        # Save output to a .py file in the output directory
        output_file = os.path.join(output_dir, f"test_{idx}.py")
        with open(output_file, "w") as f:
            f.write(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)

