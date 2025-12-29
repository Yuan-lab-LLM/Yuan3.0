import torch
from transformers import AutoModel


CNCLIP_CONFIGS = {}

# singleton globals

def get_model(model_name, clip_download_path):

    assert model_name == 'InternViT-448', '检查vit的名称'

    model = AutoModel.from_pretrained(
        clip_download_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).cuda()
    return model

def get_model_and_tokenizer(model_name, clip_download_path):
    global CNCLIP_CONFIGS

    if model_name not in CNCLIP_CONFIGS:
        CNCLIP_CONFIGS[model_name] = dict()
    if "model" not in CNCLIP_CONFIGS[model_name]:
        CNCLIP_CONFIGS[model_name]["model"] = get_model(model_name, clip_download_path)

    return CNCLIP_CONFIGS[model_name]['model']

def clip_encode_image(
    images,
    model_name,
    clip_download_path
):

    clip = get_model_and_tokenizer(model_name, clip_download_path)
    features = []
    def hook (module, input, output):
        features.append(output) 

    device = next(clip.parameters()).device
    clip.eval()

    assert model_name == 'InternViT-448', '检查vit的名称'

    final_layer_num = len(clip.encoder.layers) - 1
    handle_out = clip.encoder.layers[final_layer_num].register_forward_hook(hook)

    with torch.no_grad():
        _ = clip(images).last_hidden_state

        features_out = features[0]
    handle_out.remove()
    return features_out


