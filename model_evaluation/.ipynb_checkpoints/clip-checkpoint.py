from PIL import Image
import os
import pandas as pd
import json
import argparse
from numpy import argmax
from transformers import CLIPProcessor, CLIPModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
        default="quantifier",
        choices = ["ad-hoc", "quantifier", "upward", "downward", "quantifier_indirect"],
        help="language type",
        )
    parser.add_argument(
        "--domain",
        type=str,
        default="natural",
        choices = ["natural", "synthetic", "upward", "downward", "quantifier_indirect"],
        help="visual domain",
        )
    parser.add_argument(
        "--attribute",
        type=str,
        default="intrinsic",
        choices = ["intrinsic", "extrinsic"],
        help="attribute type for quantifiers",
        )
    parser.add_argument(
        "--zero_shot",
        type=bool,
        default=True,
        choices = [True, False],
        help="whether we shall incorporate task instruction",
        )
    
    args = parser.parse_args()
    return args

def load_jsonl(path: str):
    with open(path, "r") as f:
        ans = list(f)
    ans = [json.loads(_) for _ in ans]
    return ans

def load_task_instructions(dataset_name: str):
    #
    pass
    
def load_datasets(args: dict):
    #
    base_path = '/mnt/workspace/workgroup2/jianyu/MM-IR.github.io/'
    # 1: Quantifiers
    if args.lang == 'quantifier':
        if args.domain == 'natural':
            if args.attribute == 'intrinsic':
                src_path = os.path.join(base_path, 'human_phrasecut_intrinsic')
            elif args.attribute == 'extrinsic':
                src_path = os.path.join(base_path, 'human_phrasecut_extrinsic')
            src_meta_data_paths = os.path.join(src_path, "meta_data")
            # goal: retrieve meta data
            referents = []
            image_filenames = []
            utterances = []
            implicature_types = []
            meta_data_paths = []
            # 1. per-file processing
            for src_meta_data_path in os.listdir(src_meta_data_paths):
                if "jsonl" not in src_meta_data_path:
                    continue
                meta_data = load_jsonl(os.path.join(src_meta_data_paths,
                                                    src_meta_data_path))
                # retrieving
                local_image_filenames = []
                local_referents = []

                for item in meta_data:
                    #

                    if "referent" in item:
                        for ref in item['referent']:
                            local_referents.append(os.path.join(src_path, 
                                         "images", ref))


                        utterances.append(item['utterance'])
                        implicature_types.append(item['type'])
                    if "image_filename" in item:
                        local_image_filenames.append(
                            os.path.join(src_path, 
                                       "images", item['image_filename'])
                            )
                referents.append(local_referents)
                print(referents)
                meta_data_paths.append(src_meta_data_path)
                image_filenames.append(local_image_filenames)
        
        elif args.domain == 'synthetic':
            if args.attribute == 'intrinsic':
                src_path = os.path.join(base_path, 'human_clevr_quantifiers_intrinsic')
            elif args.attribute == 'extrinsic':
                src_path = os.path.join(base_path, 'human_clevr_quantifiers_extrinsic')
            src_meta_data_paths = os.path.join(src_path, "meta_data")
            # goal: retrieve meta data
            referents = []
            image_filenames = []
            utterances = []
            implicature_types = []
            meta_data_paths = []
            # 1. per-file processing
            for src_meta_data_path in os.listdir(src_meta_data_paths):
                if "jsonl" not in src_meta_data_path:
                    continue
                meta_data = load_jsonl(os.path.join(src_meta_data_paths,
                                                    src_meta_data_path))
                # retrieving
                local_image_filenames = []
                local_referents = []

                for item in meta_data:
                    #

                    if "referent" in item:
                        for ref in item['referent']:
                            local_referents.append(os.path.join(src_path, 
                                         "images", ref))


                        utterances.append(item['utterance'])
                        implicature_types.append(item['type'])
                    if "image_filename" in item:
                        local_image_filenames.append(
                            os.path.join(src_path, 
                                       "images", item['image_filename'])
                            )
                referents.append(local_referents)
                print(referents)
                meta_data_paths.append(src_meta_data_path)
                image_filenames.append(local_image_filenames)
    # 2: Quantifier Indirect            
    if args.lang == 'quantifier_indirect':
        if args.domain == 'natural':
            if args.attribute == 'intrinsic':
                src_path = os.path.join(base_path, 'human_phrasecut_indirect_intrinsic')
            elif args.attribute == 'extrinsic':
                src_path = os.path.join(base_path, 'human_phrasecut_indirect_extrinsic')
            src_meta_data_paths = os.path.join(src_path, "meta_data")
            # goal: retrieve meta data
            referents = []
            image_filenames = []
            utterances = []
            implicature_types = []
            meta_data_paths = []
            # 1. per-file processing
            for src_meta_data_path in os.listdir(src_meta_data_paths):
                if "jsonl" not in src_meta_data_path:
                    continue
                meta_data = load_jsonl(os.path.join(src_meta_data_paths,
                                                    src_meta_data_path))
                # retrieving
                local_image_filenames = []
                local_referents = []

                for item in meta_data:
                    #

                    if "referent" in item:
                        for ref in item['referent']:
                            local_referents.append(os.path.join(src_path, 
                                         "images", ref))


                        utterances.append(item['utterance'])
                        implicature_types.append(item['type'])
                    if "image_filename" in item:
                        local_image_filenames.append(
                            os.path.join(src_path, 
                                       "images", item['image_filename'])
                            )
                referents.append(local_referents)
                print(referents)
                meta_data_paths.append(src_meta_data_path)
                image_filenames.append(local_image_filenames)
        
        elif args.domain == 'synthetic':
            if args.attribute == 'intrinsic':
                src_path = os.path.join(base_path, 'human_clevr_quantifier_indirect_intrinsic')
            elif args.attribute == 'extrinsic':
                src_path = os.path.join(base_path, 'human_clevr_quantifier_indirect_extrinsic')
            src_meta_data_paths = os.path.join(src_path, "meta_data")
            # goal: retrieve meta data
            referents = []
            image_filenames = []
            utterances = []
            implicature_types = []
            meta_data_paths = []
            # 1. per-file processing
            for src_meta_data_path in os.listdir(src_meta_data_paths):
                if "jsonl" not in src_meta_data_path:
                    continue
                meta_data = load_jsonl(os.path.join(src_meta_data_paths,
                                                    src_meta_data_path))
                # retrieving
                local_image_filenames = []
                local_referents = []

                for item in meta_data:
                    #

                    if "referent" in item:
                        for ref in item['referent']:
                            local_referents.append(os.path.join(src_path, 
                                         "images", ref))


                        utterances.append(item['utterance'])
                        implicature_types.append(item['type'])
                    if "image_filename" in item:
                        local_image_filenames.append(
                            os.path.join(src_path, 
                                       "images", item['image_filename'])
                            )
                referents.append(local_referents)
                print(referents)
                meta_data_paths.append(src_meta_data_path)
                image_filenames.append(local_image_filenames)
                
    # 3: Embedded Implicature
    elif args.lang == 'upward':
        if True: #args.domain == 'natural':
            
            src_path = os.path.join(base_path, 'human_clevr_upward')
            src_meta_data_paths = os.path.join(src_path, "meta_data")
            # goal: retrieve meta data
            referents = []
            image_filenames = []
            utterances = []
            implicature_types = []
            meta_data_paths = []
            # 1. per-file processing
            for src_meta_data_path in os.listdir(src_meta_data_paths):
                if "jsonl" not in src_meta_data_path:
                    continue
                meta_data = load_jsonl(os.path.join(src_meta_data_paths,
                                                    src_meta_data_path))
                # retrieving
                local_image_filenames = []
                local_referents = []

                for item in meta_data:
                    #

                    if "referent" in item:
                        for ref in item['referent']:
                            local_referents.append(os.path.join(src_path, 
                                         "images", ref))


                        utterances.append(item['utterance'])
                        implicature_types.append(item['type'])
                    if "image_filename" in item:
                        local_image_filenames.append(
                            os.path.join(src_path, 
                                       "images", item['image_filename'])
                            )
                referents.append(local_referents)
                print(referents)
                meta_data_paths.append(src_meta_data_path)
                image_filenames.append(local_image_filenames)
                        
                
    elif args.lang == 'downward':
        if True: #args.domain == 'natural':
            
            src_path = os.path.join(base_path, 'human_clevr_downward')
            src_meta_data_paths = os.path.join(src_path, "meta_data")
            # goal: retrieve meta data
            referents = []
            image_filenames = []
            utterances = []
            implicature_types = []
            meta_data_paths = []
            # 1. per-file processing
            for src_meta_data_path in os.listdir(src_meta_data_paths):
                if "jsonl" not in src_meta_data_path:
                    continue
                meta_data = load_jsonl(os.path.join(src_meta_data_paths,
                                                    src_meta_data_path))
                # retrieving
                local_image_filenames = []
                local_referents = []

                for item in meta_data:
                    #

                    if "referent" in item:
                        for ref in item['referent']:
                            local_referents.append(os.path.join(src_path, 
                                         "images", ref))


                        utterances.append(item['utterance'])
                        implicature_types.append(item['type'])
                    if "image_filename" in item:
                        local_image_filenames.append(
                            os.path.join(src_path, 
                                       "images", item['image_filename'])
                            )
                referents.append(local_referents)
                print(referents)
                meta_data_paths.append(src_meta_data_path)
                image_filenames.append(local_image_filenames)
                    
    return referents, image_filenames, utterances, implicature_types, meta_data_paths
    

if __name__ == "__main__":
    model = CLIPModel.from_pretrained( 
        "/mnt/workspace/workgroup2/jianyu/clip_vit_l_14").to(0)
    processor = CLIPProcessor.from_pretrained(
        "/mnt/workspace/workgroup2/jianyu/clip_vit_l_14")
    args = parse_args()
                                             
    count_implicature_correct = 0       
    count_cancellation_correct = 0 
    
    referents, image_filenames, utterances, implicature_types, meta_data_paths \
                                                        = load_datasets(args)
    pred_record = {"json_name": meta_data_paths, 
                  "pred_answer": [],
                  "referent": [],
                  "utterance": utterances,
                  "pred_correct_flag": [],
                  "pred_probs": [], "implicature_type": implicature_types}
    # run the experiments
    
    for ind, local_image_filenames in enumerate(image_filenames):
        utterance = utterances[ind]
        referent = referents[ind]      
        implicature_type = implicature_types[ind]
          
        available_images = []
                                             
        for image_path in local_image_filenames:
            available_images.append(Image.open(image_path))
        if args.zero_shot:
            inputs = processor(text=[utterance], images=available_images, \
                               return_tensors="pt").to(0)
        
        outputs = model(**inputs)
        logits_per_text = outputs.logits_per_text
        probs = logits_per_text.softmax(dim=1)
        # pick the referent
        referent_index = argmax(probs.detach().cpu()).item()
        # save the probs
        pred_referent = local_image_filenames[referent_index]
        pred_record["pred_answer"].append(pred_referent.split("/")[-1])
        pred_record["referent"].append(referent[0].split("/")[-1])
        print(pred_referent.split("/")[-1], referent[0].split("/")[-1])
                                             
        if pred_referent == referent[0]:
            if "cancel" not in implicature_type:
                count_implicature_correct += 1
            else:
                count_cancellation_correct += 1
            pred_record["pred_correct_flag"].append(1)
            pred_record["pred_probs"].append(probs.detach().cpu().tolist()[0])
        else:
            pred_record["pred_correct_flag"].append(0)
            pred_record["pred_probs"].append(probs.detach().cpu().tolist()[0])
    #
    if args.lang in ['upward', 'downward', 'non-monotonic', 'ad-hoc']:
        args.domain = 'synthetic'
        
    if args.lang == "quantifier":
        pd.DataFrame(pred_record).to_csv(f"{args.domain}_{args.lang}_{args.attribute}_clip.csv")
    else:
        pd.DataFrame(pred_record).to_csv(f"{args.domain}_{args.lang}_clip.csv")
    
    print(f"Number of correct implicature: {count_implicature_correct}")
    print(f"Number of correct cancellation: {count_cancellation_correct}")
        
                                             
                                             
            
    
