"""
LLM Service for Drug Side Effect Explanations
Integrates OpenAI GPT API to provide detailed explanations for predicted side effects.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI Client
api_key = os.getenv("OPENAI_API_KEY")
client = None

if api_key:
    client = OpenAI(api_key=api_key)
    print("[LLM] OpenAI client initialized successfully.")
else:
    print("WARNING: OPENAI_API_KEY not found in .env. LLM features will be disabled.")

def explain_side_effect(drug_name, side_effect, medical_condition=None):
    """
    FR5: Generate an explanation for why a drug causes a specific side effect using GPT.
    """
    if not client:
        return "⚠️ AI explanation unavailable. OpenAI API Key is missing. Please add it to .env file."
    
    context = f"The patient is taking {drug_name}"
    if medical_condition and medical_condition.strip():
        context += f" for {medical_condition.strip()}"
    
    prompt = f"""Context: {context}.
Predicted Side Effect: {side_effect}.

Explain concisely (3-4 sentences) why this drug might cause this side effect based on its pharmacological mechanism of action.
Use simple language suitable for a patient. Be specific about the drug's mechanism."""
    
    try:
        print(f"[LLM-Explain] Generating explanation: {drug_name} -> {side_effect}")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable pharmacologist. Explain drug side effects clearly using simple language. Always provide accurate, evidence-based explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.5
        )
        result = response.choices[0].message.content.strip()
        print(f"[LLM-Explain] Success: got {len(result)} chars")
        return result
    except Exception as e:
        error_msg = str(e)
        print(f"[LLM-Explain] Error: {type(e).__name__}: {error_msg}")
        
        # Provide specific error messages
        if "insufficient_quota" in error_msg or "rate_limit" in error_msg:
            return "⚠️ API quota exceeded. The OpenAI API key has run out of credits. Please update the API key in the .env file."
        elif "invalid_api_key" in error_msg or "Incorrect API key" in error_msg:
            return "⚠️ Invalid API key. Please check your OpenAI API key in the .env file."
        elif "model_not_found" in error_msg:
            return "⚠️ The AI model is currently unavailable. Please try again later."
        else:
            return f"⚠️ Explanation unavailable. Error: {error_msg[:100]}"

def analyze_drug_risk(drug_name, patient_age=None, other_conditions=None):
    """
    Provide a general risk analysis for a drug using GPT.
    """
    if not client:
        return "Risk analysis unavailable (API Key missing)."
        
    prompt = f"Analyze the safety profile of {drug_name}."
    if patient_age:
        prompt += f" Patient age: {patient_age}."
    if other_conditions:
        prompt += f" Patient has these conditions: {other_conditions}."
        
    prompt += "\nSummarize key risks in bullet points. Be professional but accessible."
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant providing drug safety information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg or "rate_limit" in error_msg:
            return "⚠️ API quota exceeded. Please update the API key."
        return f"Error: {error_msg[:100]}"

def get_drug_details_from_llm(drug_name):
    """
    Fetch comprehensive drug details (features + side effects) for a new drug using GPT.
    Returns a JSON-compatible dictionary.
    Used as internal fallback when a drug is not in the local database.
    """
    if not client:
        print(f"[LLM-Fetch] ERROR: OpenAI client not initialized. Check OPENAI_API_KEY in .env")
        return None
    
    # Trim the drug name
    drug_name = drug_name.strip()
    
    prompt = f"""You are a pharmaceutical database expert. I need information about: "{drug_name}"

IMPORTANT INSTRUCTIONS:
1. "{drug_name}" may be a:
   - Generic drug name (e.g., "Aspirin", "Metformin")
   - Brand name (e.g., "Ozempic", "Nucoxia")
   - Combination drug (e.g., "Nucoxia MR" = Etoricoxib + Thiocolchicoside)
   - International/regional brand name

2. If this is a REAL pharmaceutical drug (prescription or OTC), provide its data in JSON format.

3. For COMBINATION drugs, use the PRIMARY active ingredient's properties and combine side effects from all ingredients.

4. If this is clearly NOT a drug (e.g., "Harry Potter", "xyz123"), respond with exactly: NOT_A_DRUG

Return JSON with these fields:
{{
  "category": "<drug class: NSAID, Antibiotic, Analgesic, Muscle Relaxant, etc>",
  "molecular_weight": <float, approximate for primary ingredient>,
  "log_p": <float, approximate>,
  "h_bond_donors": <int, approximate>,
  "h_bond_acceptors": <int, approximate>,
  "polar_surface_area": <float, approximate>,
  "rotatable_bonds": <int, approximate>,
  "side_effects": {{
    "<side effect 1>": <severity 0.1-1.0>,
    "<side effect 2>": <severity 0.1-1.0>,
    "<side effect 3>": <severity 0.1-1.0>,
    "<side effect 4>": <severity 0.1-1.0>,
    "<side effect 5>": <severity 0.1-1.0>
  }}
}}

EXAMPLES:
- "Ozempic" → Real drug (GLP-1 Agonist) → Return JSON
- "Nucoxia MR" → Real drug (NSAID + Muscle Relaxant) → Return JSON
- "Harry Potter" → Not a drug → Return NOT_A_DRUG

Output ONLY the JSON or NOT_A_DRUG. No explanations."""
    
    try:
        print(f"[LLM-Fetch] Querying GPT for drug: '{drug_name}'")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a comprehensive pharmaceutical database with knowledge of generic drugs, brand names, combination drugs, and international medications. You provide accurate drug information in JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.2
        )
        
        content = response.choices[0].message.content.strip()
        print(f"[LLM-Fetch] Raw response: {content[:300]}")
        
        # Check if GPT says it's not a real drug
        if "NOT_A_DRUG" in content.upper():
            print(f"[LLM-Fetch] '{drug_name}' is not a recognized drug")
            return None
        
        # Clean up markdown formatting
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
            
        data = json.loads(content)
        
        # Validate required fields
        if "side_effects" not in data or not isinstance(data.get("side_effects"), dict):
            print(f"[LLM-Fetch] Invalid side_effects format: {type(data.get('side_effects'))}")
            return None
        
        if len(data["side_effects"]) == 0:
            print(f"[LLM-Fetch] Empty side effects returned")
            return None
            
        print(f"[LLM-Fetch] Successfully fetched data for '{drug_name}' with {len(data['side_effects'])} side effects")
        return data
        
    except json.JSONDecodeError as e:
        print(f"[LLM-Fetch] JSON parsing error: {e}")
        print(f"[LLM-Fetch] Content was: {content}")
        return None
    except Exception as e:
        error_msg = str(e)
        print(f"[LLM-Fetch] Error: {type(e).__name__}: {error_msg}")
        if "insufficient_quota" in error_msg:
            print(f"[LLM-Fetch] API KEY HAS NO CREDITS! Update OPENAI_API_KEY in .env")
        return None
