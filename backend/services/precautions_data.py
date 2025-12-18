"""
Static precautions data for common diseases.
Used as fallback when web scraping fails.
These are evidence-based general health precautions.
"""

# Comprehensive dictionary of disease precautions
# Based on medical guidelines from CDC, WHO, and medical literature
DISEASE_PRECAUTIONS = {
    "(vertigo) Paroymsal  Positional Vertigo": [
        "Avoid sudden head movements and position changes",
        "Sleep with your head slightly elevated on pillows",
        "Perform Epley maneuver exercises as recommended by your doctor",
        "Avoid bending down to pick up objects; squat instead",
        "Take precautions to prevent falls, especially at night",
        "Avoid driving or operating machinery during episodes",
        "Stay hydrated and avoid excessive caffeine and alcohol"
    ],
    "AIDS": [
        "Take antiretroviral therapy (ART) medications as prescribed",
        "Practice safe sex and use protection consistently",
        "Get regular CD4 count and viral load tests",
        "Avoid sharing needles or personal items that may have blood",
        "Maintain a healthy diet and exercise regularly",
        "Get vaccinated against preventable infections",
        "Consult your doctor immediately if you develop new symptoms",
        "Avoid raw or undercooked foods to prevent infections"
    ],
    "Acne": [
        "Wash your face twice daily with a gentle cleanser",
        "Avoid touching your face with unwashed hands",
        "Use non-comedogenic skincare and makeup products",
        "Do not pick, squeeze, or pop pimples",
        "Keep hair clean and away from your face",
        "Change pillowcases frequently",
        "Stay hydrated and maintain a balanced diet",
        "Use oil-free sunscreen when going outdoors"
    ],
    "Alcoholic hepatitis": [
        "Stop drinking alcohol completely and immediately",
        "Follow a nutritious, balanced diet high in protein",
        "Take prescribed medications as directed by your doctor",
        "Get vaccinated for Hepatitis A and B",
        "Avoid acetaminophen and other liver-toxic medications",
        "Attend regular follow-up appointments for liver monitoring",
        "Consider joining alcohol cessation support groups",
        "Get adequate rest and avoid strenuous activities"
    ],
    "Allergy": [
        "Identify and avoid known allergens",
        "Keep antihistamines readily available",
        "Wear a medical alert bracelet if you have severe allergies",
        "Keep your living space clean and dust-free",
        "Use air purifiers with HEPA filters",
        "Check pollen counts before outdoor activities",
        "Carry an epinephrine auto-injector if prescribed",
        "Read food labels carefully to avoid hidden allergens"
    ],
    "Arthritis": [
        "Maintain a healthy weight to reduce joint stress",
        "Exercise regularly with low-impact activities like swimming",
        "Apply hot or cold therapy to affected joints",
        "Take prescribed anti-inflammatory medications",
        "Use assistive devices to protect joints",
        "Practice joint protection techniques in daily activities",
        "Get adequate sleep and rest periods",
        "Consider physical therapy for targeted exercises"
    ],
    "Bronchial Asthma": [
        "Carry your rescue inhaler at all times",
        "Identify and avoid your asthma triggers",
        "Take controller medications as prescribed daily",
        "Monitor your peak flow readings regularly",
        "Create an asthma action plan with your doctor",
        "Keep your home free of dust, mold, and pet dander",
        "Avoid smoking and secondhand smoke exposure",
        "Get annual flu vaccination and stay current on immunizations"
    ],
    "Cervical spondylosis": [
        "Maintain good posture while sitting and standing",
        "Use ergonomic furniture and proper neck support",
        "Do gentle neck stretching exercises daily",
        "Apply heat or cold packs to relieve pain",
        "Avoid activities that strain your neck",
        "Use a supportive pillow while sleeping",
        "Take regular breaks from computer and phone use",
        "Consider physical therapy for strengthening exercises"
    ],
    "Chicken pox": [
        "Isolate the patient to prevent spreading the infection",
        "Keep fingernails trimmed short to prevent scratching",
        "Apply calamine lotion to relieve itching",
        "Take lukewarm baths with oatmeal or baking soda",
        "Wear loose, cotton clothing to reduce irritation",
        "Stay hydrated and get plenty of rest",
        "Take acetaminophen for fever (avoid aspirin in children)",
        "Avoid contact with pregnant women and immunocompromised individuals"
    ],
    "Chronic cholestasis": [
        "Follow a low-fat diet as recommended by your doctor",
        "Take prescribed medications to manage bile flow",
        "Avoid alcohol consumption completely",
        "Get regular liver function tests",
        "Take vitamin supplements (A, D, E, K) if prescribed",
        "Manage itching with prescribed medications",
        "Stay hydrated and maintain proper nutrition",
        "Report any worsening symptoms to your doctor promptly"
    ],
    "Common Cold": [
        "Get plenty of rest and sleep",
        "Stay well hydrated with water, juice, and warm fluids",
        "Use saline nasal drops to relieve congestion",
        "Gargle with warm salt water for sore throat",
        "Use a humidifier to add moisture to the air",
        "Wash hands frequently to prevent spreading",
        "Cover your mouth when coughing or sneezing",
        "Take over-the-counter medications for symptom relief"
    ],
    "Dengue": [
        "Rest completely and avoid strenuous activities",
        "Stay well hydrated with oral rehydration solutions",
        "Take acetaminophen for fever and pain (avoid aspirin and ibuprofen)",
        "Monitor platelet count and seek medical attention if it drops",
        "Use mosquito nets and repellents to prevent further bites",
        "Eliminate standing water around your home",
        "Watch for warning signs of severe dengue",
        "Seek immediate medical care if you have severe abdominal pain or vomiting"
    ],
    "Diabetes ": [
        "Monitor blood sugar levels regularly as prescribed",
        "Take diabetes medications or insulin as directed",
        "Follow a balanced diet with controlled carbohydrate intake",
        "Exercise regularly to help manage blood sugar",
        "Check your feet daily for cuts, blisters, or sores",
        "Attend regular check-ups for eyes, kidneys, and heart",
        "Maintain a healthy weight",
        "Carry fast-acting glucose for hypoglycemia emergencies"
    ],
    "Dimorphic hemmorhoids(piles)": [
        "Increase fiber intake through fruits, vegetables, and whole grains",
        "Stay well hydrated by drinking plenty of water",
        "Avoid straining during bowel movements",
        "Take warm sitz baths to relieve discomfort",
        "Use over-the-counter hemorrhoid creams as directed",
        "Exercise regularly to improve bowel function",
        "Avoid sitting for prolonged periods",
        "Do not delay going to the bathroom when you feel the urge"
    ],
    "Drug Reaction": [
        "Stop taking the suspected medication immediately",
        "Seek immediate medical attention for severe reactions",
        "Keep a record of all medications that caused reactions",
        "Wear a medical alert bracelet listing drug allergies",
        "Inform all healthcare providers about your drug allergies",
        "Take prescribed antihistamines or steroids as directed",
        "Carry emergency medication if prescribed",
        "Read all medication labels and inserts carefully"
    ],
    "Fungal infection": [
        "Keep the affected area clean and dry",
        "Apply prescribed antifungal creams or medications",
        "Wear loose, breathable cotton clothing",
        "Avoid sharing personal items like towels and combs",
        "Change socks and underwear daily",
        "Dry thoroughly between toes after bathing",
        "Avoid walking barefoot in public areas",
        "Complete the full course of antifungal treatment"
    ],
    "GERD": [
        "Eat smaller, more frequent meals",
        "Avoid eating 2-3 hours before bedtime",
        "Elevate the head of your bed by 6-8 inches",
        "Avoid trigger foods like spicy, fatty, and acidic foods",
        "Maintain a healthy weight",
        "Avoid tight-fitting clothing around the abdomen",
        "Quit smoking and limit alcohol consumption",
        "Take prescribed antacids or proton pump inhibitors"
    ],
    "Gastroenteritis": [
        "Stay well hydrated with clear fluids and oral rehydration solutions",
        "Rest and allow your body to recover",
        "Start with bland foods like rice, bananas, and toast",
        "Avoid dairy products until fully recovered",
        "Wash hands thoroughly and frequently",
        "Avoid preparing food for others while symptomatic",
        "Seek medical attention if symptoms persist beyond 48 hours",
        "Watch for signs of dehydration, especially in children"
    ],
    "Heart attack": [
        "Call emergency services immediately if symptoms occur",
        "Take prescribed cardiac medications without missing doses",
        "Follow a heart-healthy diet low in saturated fat and sodium",
        "Participate in cardiac rehabilitation as recommended",
        "Exercise regularly as approved by your cardiologist",
        "Quit smoking and avoid secondhand smoke",
        "Manage stress through relaxation techniques",
        "Monitor blood pressure and cholesterol regularly"
    ],
    "Hepatitis B": [
        "Take antiviral medications as prescribed by your doctor",
        "Avoid alcohol completely to protect your liver",
        "Get regular liver function and viral load tests",
        "Practice safe sex and inform partners of your status",
        "Do not share razors, toothbrushes, or needles",
        "Ensure close contacts are vaccinated against Hepatitis B",
        "Follow a nutritious, liver-friendly diet",
        "Avoid medications that can harm the liver without consulting your doctor"
    ],
    "Hepatitis C": [
        "Complete the full course of antiviral treatment as prescribed",
        "Avoid alcohol consumption to protect liver health",
        "Get regular monitoring of liver function",
        "Do not share personal items that may have blood on them",
        "Practice safe sex and inform partners",
        "Get tested for other hepatitis types",
        "Maintain a healthy diet and weight",
        "Avoid over-the-counter medications that affect the liver"
    ],
    "Hepatitis D": [
        "Follow treatment plan for both Hepatitis B and D",
        "Avoid alcohol and liver-toxic substances",
        "Get regular liver monitoring and ultrasounds",
        "Practice safe sex and do not share needles",
        "Ensure household contacts are vaccinated against Hepatitis B",
        "Eat a balanced, nutritious diet",
        "Rest adequately and avoid strenuous activities",
        "Report any new symptoms to your doctor immediately"
    ],
    "Hepatitis E": [
        "Rest and stay well hydrated",
        "Avoid alcohol during recovery",
        "Eat a balanced, low-fat diet",
        "Practice good hand hygiene, especially before eating",
        "Drink only safe, purified water",
        "Avoid undercooked meat, especially pork",
        "Get regular liver function tests during recovery",
        "Pregnant women should seek immediate medical care"
    ],
    "Hypertension ": [
        "Take blood pressure medications as prescribed",
        "Follow the DASH diet (low sodium, high potassium)",
        "Reduce sodium intake to less than 2,300 mg per day",
        "Exercise regularly for at least 30 minutes most days",
        "Maintain a healthy weight",
        "Limit alcohol consumption",
        "Quit smoking and avoid secondhand smoke",
        "Monitor blood pressure at home regularly"
    ],
    "Hyperthyroidism": [
        "Take prescribed anti-thyroid medications as directed",
        "Follow up regularly with your endocrinologist",
        "Limit iodine intake as recommended by your doctor",
        "Avoid caffeine and other stimulants",
        "Get adequate rest and manage stress",
        "Protect your eyes if you have thyroid eye disease",
        "Report any palpitations or rapid heartbeat immediately",
        "Consider radioactive iodine or surgery if recommended"
    ],
    "Hypoglycemia": [
        "Carry fast-acting glucose or sugary snacks at all times",
        "Eat regular meals and snacks to maintain blood sugar",
        "Monitor blood sugar levels frequently",
        "Wear a medical alert bracelet",
        "Teach family and friends how to help during an episode",
        "Avoid skipping meals or delaying eating",
        "Limit alcohol consumption, especially on an empty stomach",
        "Review medications with your doctor that may cause low blood sugar"
    ],
    "Hypothyroidism": [
        "Take thyroid hormone replacement medication daily as prescribed",
        "Take medication on an empty stomach, 30-60 minutes before breakfast",
        "Get regular thyroid function tests",
        "Do not skip doses of thyroid medication",
        "Be aware of drug interactions with thyroid medication",
        "Maintain regular exercise to boost metabolism",
        "Eat a balanced diet with adequate iodine",
        "Report any symptoms of under- or over-treatment to your doctor"
    ],
    "Impetigo": [
        "Keep the affected areas clean with soap and water",
        "Apply prescribed antibiotic ointment as directed",
        "Cover sores with bandages to prevent spreading",
        "Wash hands frequently, especially after touching sores",
        "Do not share towels, clothing, or personal items",
        "Cut fingernails short to prevent scratching",
        "Wash clothing and bedding in hot water",
        "Complete the full course of antibiotics if prescribed"
    ],
    "Jaundice": [
        "Rest adequately and avoid strenuous activities",
        "Stay well hydrated with plenty of fluids",
        "Follow a low-fat, easy-to-digest diet",
        "Avoid alcohol completely",
        "Take prescribed medications as directed",
        "Get regular liver function monitoring",
        "Seek immediate care if symptoms worsen",
        "Avoid medications that can harm the liver"
    ],
    "Malaria": [
        "Complete the full course of antimalarial treatment",
        "Rest and stay well hydrated",
        "Take acetaminophen for fever (avoid aspirin)",
        "Use mosquito nets treated with insecticide",
        "Apply insect repellent containing DEET",
        "Wear long sleeves and pants in endemic areas",
        "Seek immediate medical attention for severe symptoms",
        "Continue preventive medication if in endemic area"
    ],
    "Migraine": [
        "Identify and avoid your personal migraine triggers",
        "Take prescribed medications at the first sign of a migraine",
        "Rest in a dark, quiet room during an attack",
        "Apply cold compresses to your forehead",
        "Maintain a regular sleep schedule",
        "Stay hydrated and don't skip meals",
        "Manage stress through relaxation techniques",
        "Keep a migraine diary to track patterns and triggers"
    ],
    "Osteoarthristis": [
        "Maintain a healthy weight to reduce joint stress",
        "Exercise regularly with low-impact activities",
        "Use hot or cold therapy for pain relief",
        "Take prescribed pain medications as directed",
        "Use assistive devices like canes or braces if needed",
        "Practice range-of-motion exercises daily",
        "Consider physical therapy for strengthening",
        "Protect joints by avoiding repetitive stress"
    ],
    "Paralysis (brain hemorrhage)": [
        "Follow rehabilitation program as prescribed",
        "Attend physical, occupational, and speech therapy regularly",
        "Take medications to prevent further strokes",
        "Manage blood pressure and other risk factors",
        "Prevent falls with home safety modifications",
        "Practice exercises to maintain muscle tone",
        "Use assistive devices as recommended",
        "Seek emotional support and counseling if needed"
    ],
    "Peptic ulcer diseae": [
        "Take prescribed proton pump inhibitors or antibiotics",
        "Avoid NSAIDs like aspirin and ibuprofen",
        "Quit smoking to promote healing",
        "Limit alcohol consumption",
        "Eat smaller, more frequent meals",
        "Avoid spicy and acidic foods that trigger symptoms",
        "Manage stress through relaxation techniques",
        "Complete H. pylori treatment if prescribed"
    ],
    "Pneumonia": [
        "Take prescribed antibiotics and complete the full course",
        "Get plenty of rest to allow your body to heal",
        "Stay well hydrated with water and warm fluids",
        "Use a humidifier to ease breathing",
        "Take fever-reducing medications as needed",
        "Practice deep breathing exercises",
        "Seek immediate care if breathing becomes difficult",
        "Get vaccinated against pneumonia and flu in the future"
    ],
    "Psoriasis": [
        "Keep skin moisturized with fragrance-free creams",
        "Apply prescribed topical treatments consistently",
        "Take medicated baths with oatmeal or coal tar",
        "Get moderate sun exposure but avoid sunburn",
        "Avoid triggers like stress, injury, and certain medications",
        "Quit smoking and limit alcohol consumption",
        "Manage stress through relaxation techniques",
        "Follow up regularly with your dermatologist"
    ],
    "Tuberculosis": [
        "Take all TB medications exactly as prescribed",
        "Complete the full treatment course (usually 6-9 months)",
        "Cover your mouth when coughing or sneezing",
        "Stay in well-ventilated areas",
        "Isolate yourself during the contagious period",
        "Attend all follow-up appointments and tests",
        "Ensure close contacts are tested for TB",
        "Eat a nutritious diet to support recovery"
    ],
    "Typhoid": [
        "Complete the full course of prescribed antibiotics",
        "Stay well hydrated with safe, clean water",
        "Rest and avoid strenuous activities",
        "Eat small, frequent, easily digestible meals",
        "Practice strict hand hygiene",
        "Avoid preparing food for others until cleared by doctor",
        "Get tested to confirm you are no longer a carrier",
        "Consider typhoid vaccination for future prevention"
    ],
    "Urinary tract infection": [
        "Complete the full course of prescribed antibiotics",
        "Drink plenty of water to flush out bacteria",
        "Urinate frequently and don't hold urine",
        "Wipe from front to back after using the bathroom",
        "Urinate before and after sexual activity",
        "Avoid irritating feminine products",
        "Wear cotton underwear and loose-fitting clothes",
        "Consider cranberry products for prevention"
    ],
    "Varicose veins": [
        "Elevate your legs when resting",
        "Avoid standing or sitting for long periods",
        "Wear compression stockings as recommended",
        "Exercise regularly to improve circulation",
        "Maintain a healthy weight",
        "Avoid crossing your legs when sitting",
        "Avoid high heels and tight clothing",
        "Consider medical procedures if symptoms worsen"
    ],
    "hepatitis A": [
        "Rest and allow your body to recover",
        "Stay well hydrated",
        "Avoid alcohol completely",
        "Eat a balanced, low-fat diet",
        "Practice good hand hygiene",
        "Avoid preparing food for others during illness",
        "Get vaccinated contacts and household members",
        "Avoid medications that can stress the liver"
    ]
}


def get_precautions_for_disease(disease: str) -> list:
    """
    Get precautions for a specific disease.
    Returns an empty list if disease is not found.
    
    Args:
        disease: The name of the disease
        
    Returns:
        List of precaution strings
    """
    # Try exact match first
    if disease in DISEASE_PRECAUTIONS:
        return DISEASE_PRECAUTIONS[disease]
    
    # Try case-insensitive match
    disease_lower = disease.lower().strip()
    for key, precautions in DISEASE_PRECAUTIONS.items():
        if key.lower().strip() == disease_lower:
            return precautions
    
    # Try partial match
    for key, precautions in DISEASE_PRECAUTIONS.items():
        if disease_lower in key.lower() or key.lower() in disease_lower:
            return precautions
    
    return []


def get_all_diseases() -> list:
    """
    Get list of all diseases with available precautions.
    
    Returns:
        List of disease names
    """
    return list(DISEASE_PRECAUTIONS.keys())


def has_precautions(disease: str) -> bool:
    """
    Check if precautions are available for a disease.
    
    Args:
        disease: The name of the disease
        
    Returns:
        True if precautions exist, False otherwise
    """
    return len(get_precautions_for_disease(disease)) > 0


# Generic precautions for unknown diseases
GENERIC_PRECAUTIONS = [
    "Consult a healthcare professional for proper diagnosis and treatment",
    "Get adequate rest and sleep to support recovery",
    "Stay well hydrated by drinking plenty of fluids",
    "Take prescribed medications exactly as directed",
    "Maintain good hygiene to prevent spreading illness",
    "Monitor your symptoms and seek immediate care if they worsen",
    "Follow a balanced, nutritious diet to support your immune system",
    "Avoid self-medication and always consult a doctor"
]


def get_generic_precautions() -> list:
    """
    Get generic precautions for unknown diseases.
    
    Returns:
        List of generic precaution strings
    """
    return GENERIC_PRECAUTIONS.copy()


