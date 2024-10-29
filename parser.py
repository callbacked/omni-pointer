from PIL import Image, ImageDraw
import torch
import base64
import io
import sys
import os
import requests
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import check_ocr_box, get_yolo_model, get_som_labeled_img, get_caption_model_processor
# WORKFLOW
# 1. Load image and analyze it using OmniParser
# 2. Get visual context using a lightweight LLM like Moondream (1.8b params)
# 3. Query a larger LLM with better contextual understanding like GPT-4o-mini to locate a specific UI element

# USAGE:
# python parser.py <image_path>
# It will then ask you to input the name of a UI element to locate, and it will return the coordinates of that element in the image.
# output saved as bullseye.png
# debug output saved as omni_viz.png

class MultiLLMLocator:
    def __init__(self, 
                 moondream_model_id="vikhyatk/moondream2", # smaller local LLM for visual context
                 large_llm="http://localhost:1337/v1/chat/completions"): # or use whatever endpoint here, I'm just using my wrapper api (dont want to burn api credits)
        print("Initializing models...")
        # Init models
        self.yolo_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
        self.caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
        self.moondream = AutoModelForCausalLM.from_pretrained(moondream_model_id, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(moondream_model_id)
        self.large_llm = large_llm
        self.conversation_id = None
        
        self.draw_bbox_config = {
            'text_scale': 0.8,
            'text_thickness': 2,
            'text_padding': 2,
            'thickness': 2,
        }
        

        self.current_image_path = None
        self.current_parsed_data = None
        self.current_visual_context = None

    def analyze_image(self, image_path):
        """Analyze image and store results"""
        print("Loading and analyzing image...")
        

        self.current_parsed_data = self.parse_image(image_path)
        

        self.current_visual_context = self.get_moondream_context(image_path)
        

        self.current_image_path = image_path
        

        self.conversation_id = None
        
        print("Image analysis complete!")
        print("\nVisual context:", self.current_visual_context)

    def parse_image(self, image_path):
        """Parse the image using OmniParser with debug output"""
        print("\nRunning OCR and element detection...")
        
        ocr_bbox_rslt, *_ = check_ocr_box(
            image_path,
            display_img=False,
            output_bb_format='xyxy',
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        print("\nOCR Results:")
        for t, box in zip(text, ocr_bbox):
            print(f"Text: '{t}' at position {box}")

        print("\nGenerating element detection visualization...")
        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_path,
            self.yolo_model,
            BOX_TRESHOLD=0.05,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=self.draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text,
            iou_threshold=0.1,
            use_local_semantics=True
        )

        # Save debug visualization
        try:
            image_data = base64.b64decode(dino_labeled_img)
            omni_viz = Image.open(io.BytesIO(image_data))
            omni_viz.save('omni_viz.png')
            print("Saved OmniParser visualization to omni_viz.png")
        except Exception as e:
            print(f"Error saving visualization: {e}")

        print("\nParsed Elements:")
        for content in parsed_content_list:
            print(content)

        return {
            'labeled_img': dino_labeled_img,
            'coordinates': label_coordinates,
            'parsed_content': parsed_content_list,
            'ocr_text': text,
            'ocr_bbox': ocr_bbox
        }

    def get_moondream_context(self, image_path):
        """Get visual context using Moondream"""
        image = Image.open(image_path)
        image_embeddings = self.moondream.encode_image(image)
        context_prompt = "Describe this image in the greatest detail, focusing on the layout and relative positions of elements."
        return self.moondream.answer_question(image_embeddings, context_prompt, self.tokenizer)

    def query_gpt4(self, messages):
        """Query GPT-4o-mini or other LLM through some api"""
        payload = {
            "model": "keyless-claude-3-haiku",
            "messages": messages,
            "stream": False
        }
        
        if self.conversation_id:
            payload["conversation_id"] = self.conversation_id
        
        try:
            response = requests.post(
                self.large_llm,
                headers={"Content-Type": "application/json"},
                json=payload
            )
            response.raise_for_status()
            response_data = response.json()
            
            if not self.conversation_id and "id" in response_data:
                self.conversation_id = response_data["id"]
                
            return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error querying GPT-4o-mini: {e}")
            return None

    def locate_element(self, target_element):
        """Determine element location using semantic understanding"""
        if not self.current_parsed_data:
            print("No image data loaded! Please analyze an image first.")
            return None
            
        prompt = f"""You are a UI navigation assistant. When a user describes what they're looking for, understand their intent and find the most semantically relevant UI element.

        Current UI Elements:
        {self.current_parsed_data['parsed_content']}

        Context about the interface:
        {self.current_visual_context}

        instructions:
        1. Cross-reference the element's name, appearance, and information from OmniParser’s parsed UI elements and do your best inference
        2. Consider both the visual context and parsed elements as reference.
        3. **Return only the coordinates** in the following format: x,y. For example, `340,560`.

        Task: Determine the (x,y) best pixel coordinates for {target_element}" 
        """

        messages = [
            {"role": "system", "content": "You are a precise UI element locator"},
            {"role": "user", "content": prompt}
        ]

        response = self.query_gpt4(messages)
        print(response)
        
        if response:
            import re
            coords = re.findall(r'(\d+)(?:\s*,\s*|\s+)(\d+)', response)
            if coords:
                return int(coords[0][0]), int(coords[0][1])
        return None

    def draw_bullseye(self, coordinates, output_path='bullseye.png'):
        """Draw a bullseye marker at the specified coordinates"""
        if not self.current_image_path:
            print("No image loaded!")
            return None
            
        image = Image.open(self.current_image_path)
        draw = ImageDraw.Draw(image)
        
        x, y = coordinates
        radius = 20
        
        # Draw outer circle
        draw.ellipse(
            [(x-radius, y-radius), (x+radius, y+radius)],
            outline='red',
            width=2
        )
        
        # Draw inner circle
        draw.ellipse(
            [(x-radius/2, y-radius/2), (x+radius/2, y+radius/2)],
            outline='red',
            width=2
        )
        
        # Draw crosshairs
        draw.line([(x-radius, y), (x+radius, y)], fill='red', width=2)
        draw.line([(x, y-radius), (x, y+radius)], fill='red', width=2)
        
        image.save(output_path)
        return output_path

def interactive_session():
    if len(sys.argv) != 2:
        print("Usage: python multi_llm_parser.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"\nProcessing image: {image_path}")
    
    locator = MultiLLMLocator()
    locator.analyze_image(image_path)
    
    print("\nEnter 'quit' to exit, 'reload' to reload the image analysis, or type an element name to locate.")
    print("Debug visualization has been saved as omni_viz.png")
    
    while True:
        target_element = input("\nWhat UI element would you like to locate? ").strip().lower()
        
        if target_element == 'quit':
            print("Goodbye!")
            break
        elif target_element == 'reload':
            print("Reloading image analysis...")
            locator.analyze_image(image_path)
            continue
        elif target_element == 'debug':
            print("\nCurrent parsed elements:")
            for content in locator.current_parsed_data['parsed_content']:
                print(content)
            continue
        elif not target_element:
            continue
            
        print("Determining precise location...")
        coordinates = locator.locate_element(target_element)

        if coordinates:
            output_path = locator.draw_bullseye(coordinates)
            print(f"Bullseye marker drawn at {coordinates}. Saved to {output_path}")
        else:
            print(f"Couldn't determine location for '{target_element}'")

if __name__ == "__main__":
    interactive_session()