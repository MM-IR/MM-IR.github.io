def call_gpt4v_inference_with_recall(image_path, question, cropped_image_path, GPT4V_ENDPOINT, GPT4V_KEY):
    # Configuration
    # GPT4V_KEY = "35632dae7dd94d0a93338db373c63893"
    IMAGE_PATH = image_path
    encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
    # additional visual cues
    # read images ends with .png in cropped_image_path
    if cropped_image_path is None:
        visual_cue_paths = []
        print("Doing inference without visual cues.")
    else:
        visual_cue_paths = [f"{cropped_image_path}visual_cues.png"]
        # list all subdirectories in the cropped_image_path
        phrase_subdirs = [f for f in os.listdir(cropped_image_path) if os.path.isdir(os.path.join(cropped_image_path, f))]
        # list all visual cues in the subdirectories, which ends with .png
        for phrase_subdir in phrase_subdirs:
            visual_cue_paths.extend([f"{cropped_image_path}/{phrase_subdir}/{f}" for f in os.listdir(os.path.join(cropped_image_path, phrase_subdir)) if f.endswith('.png')])
        if len(visual_cue_paths) == 1:
            visual_cue_paths = []
        print(f"Doing inference with {len(visual_cue_paths)} visual cues.")
    encoded_visual_cue = [base64.b64encode(open(visual_cue_path, 'rb').read()).decode('ascii') for visual_cue_path in visual_cue_paths]

    headers = {
        "Content-Type": "application/json",
        "api-key": GPT4V_KEY,
    }

    # Payload for the request
    payload = {
        "messages": [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "You are an AI assistant that helps people find information."
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"You will be given a question and several answer options. You should choose the correct option based on the image provided to you. You just need to answer the question and do not need any information about individuals. When you are not sure about the answer, just guess the most likely one. Answer the question from the camera's perspective."
                },
                {
                "type": "text",
                "text": f"Original Image:" 
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
                },
                {
                "type": "text",
                "text": f"Zoomed-in Sub-images:"
                },
            ]
            }
        ],
        "temperature": 0,
        "top_p": 0,
        "max_tokens": 800
    }
    
    # add visual cues to the payload
    # if len(encoded_visual_cue) != 0:
    #     for i, visual_cue in enumerate(encoded_visual_cue):
    #         payload["messages"][1]["content"].append({
    #             "type": "image_url",
    #             "image_url": {
    #                 "url": f"data:image/png;base64,{visual_cue}"
    #             }
    #         })
    # only add the first visual cue
    if len(encoded_visual_cue) != 0:
        payload["messages"][1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encoded_visual_cue[0]}"
            }
        })
    
    # add questions and options to the payload
    payload["messages"][1]["content"].append({
        "type": "text",
        "text": f"{question}"   
    })
    print("----------------Prompt----------------")
    for message in payload["messages"][1]["content"]:
        if message["type"] == "text":
            print(message["text"])
        else:
            print("Input Image")
    
    # GPT4V_ENDPOINT = "https://damo-openai-gpt4v-test.openai.azure.com/openai/deployments/gpt4-vision/chat/completions?api-version=2024-02-15-preview"

    # Send request
    try:
        response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        print(f"Failed to make the request. Error: {e}")
        return False
    
    # reads the message as a python list
    message = response.json()["choices"][0]["message"]["content"]
    if message.startswith("I'm sorry"):
        return False
    return message


