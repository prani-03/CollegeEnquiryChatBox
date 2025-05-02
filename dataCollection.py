import google.generativeai as genai
import json

# Configure your Gemini API key
api_key ="Your Gemini API Key"

# Configure the API key
genai.configure(api_key=api_key)

# Function to generate intents using Gemini's API
def generate_intents():
    prompt = (
        "Generate a list of intents for a college enquiry chatbot."
        "Each intent should include a tag, patterns (common user questions)."
        "and responses (answers from the chatbot)."
        "Format the output as valid JSON."
    )

    model = genai.GenerativeModel('gemini-1.5-flash')  

    response =model.generate_content(prompt)
    generated_text=response.text
    
    print("Raw response:",generated_text)  
    mygeneratedtext=generated_text[7:-4]
    print(mygeneratedtext)
   # print("Try and catch start")
  
    try:
       
        intents_data = json.loads(mygeneratedtext)
    except Exception as e:
        print("Failed to decode JSON from generated text.")
        print(e)
        
        #print("Generated text:", generated_text)  
        return None

    return intents_data

# Generate intents and save to intents.json
intents_data = generate_intents()

if intents_data:
    with open("newintents.json", 'w') as json_file:
        json.dump(intents_data, json_file, indent=4)
    print("newintents.json has been created successfully!")
else:
    print("No intents data generated.")
