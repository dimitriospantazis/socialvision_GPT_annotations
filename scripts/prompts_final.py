def get_prompts():

   prompt = {}

   prompt['distance'] = {}
   prompt['distance']['user'] = """
   Here are several frames from a video. On a scale from 0 to 1, where 0 means physically touching and 1 means very far, how close are the people in this scene? Provide a score using the entire range, with intervals of 0.1.

   Additionally, please provide a confidence score indicating how certain you are about this distance rating.

   Example 1:
   Scene: Two people are sitting side by side on a bench, their shoulders touching.
   Answer: <score> = 0.0
   Confidence: <confidence> = 0.9

   Example 2:
   Scene: Two individuals are standing across a large room, not interacting.
   Answer: <score> = 1.0
   Confidence: <confidence> = 0.95

   Explain with two sentences why you gave this score and confidence level, especially if the difference is subtle.

   Now, please estimate the score and confidence level for the scene in the provided images and return strictly in a JSON format:
   {
      "explanation": "",
      "score": ,
      "confidence": 
   }
   """



   prompt['object'] = {}
   prompt['object']['user'] = """
   Here are several frames from a video. On a scale from 0 to 1, where 0 means no individual is directing their attention toward a physical object at all, and 1 means at least one of them is largely focused and actively interacting with a physical object instead of the other person, how much are the people in this scene interacting with an object? Provide a score using the entire range, with intervals of 0.1.

   Additionally, please provide a confidence score indicating how certain you are about this object interaction rating.

   Example 1:
   Scene: A person is standing still, looking out of a window without touching anything.
   Answer: <score> = 0.0
   Confidence: <confidence> = 0.8

   Example 2:
   Scene: A person is holding a coffee mug and taking a sip.
   Answer: <score> = 1.0
   Confidence: <confidence> = 0.9

   Explain with two sentences why you gave this score and confidence level, especially if the difference is subtle.

   Now, please estimate the score and confidence level for the scene in the provided images and return strictly in a JSON format:
   {
      "explanation": "",
      "score": ,
      "confidence": 
   }
   """


   prompt['expanse'] = {}
   prompt['expanse']['user'] = """
   Here are several frames from a video. On a scale from 0 to 1, where 0 means the individuals are located in a very small or confined space (such as a bathroom or a small office), and 1 means the individuals are located in a very large or expansive space (such as an auditorium, open field, or large hall), how would you rate the spatial scale of the scene? Provide a score using the entire range, with intervals of 0.1.

   Additionally, please provide a confidence score indicating how certain you are about this spatial scale rating.   

   Example 1:
   Scene: Two people are sitting in a small, enclosed office with walls close to their position.
   Answer: <score> = 0.0
   Confidence: <confidence> = 0.85

   Example 2:
   Scene: A group of people are standing in a large, open auditorium, with significant space surrounding them.
   Answer: <score> = 1.0
   Confidence: <confidence> = 0.9

   Explain with two sentences why you gave this score and confidence level, especially if the difference is subtle.

   Now, please estimate the score and confidence level for the scene in the provided images and return strictly in a JSON format:
   {
      "explanation": "",
      "score": ,
      "confidence": 
   }
   """



   prompt['facingness'] = {}
   prompt['facingness']['user'] = """
   Here are several frames from a video. On a scale from 0 to 1, where 0 means the individuals are not facing each other at all, and 1 means the individuals are fully facing each other (such as directly looking at one another in conversation or interaction), how much are the people in this scene facing each other? Provide a score using the entire range, with intervals of 0.1.

   Additionally, please provide a confidence score indicating how certain you are about this facingness rating.

   Example 1:
   Scene: Two people are sitting at a table, but they are both looking at their phones, not facing each other.
   Answer: <score> = 0.0
   Confidence: <confidence> = 0.8

   Example 2:
   Scene: Two individuals are standing directly across from each other, engaged in conversation, making eye contact.
   Answer: <score> = 1.0
   Confidence: <confidence> = 0.9

   Explain with two sentences why you gave this score and confidence level, especially if the difference is subtle.

   Now, please estimate the score and confidence level for the scene in the provided images and return strictly in a JSON format:
   {
      "explanation": "",
      "score": ,
      "confidence": 
   }
   """
   
   
   prompt['communicating'] = {}
   prompt['communicating']['user'] = """
   Here are several frames from a video. On a scale from 0 to 1, where 0 means the individuals are not communicating with each other at all (such as being entirely focused on separate activities without any visible interaction), and 1 means the individuals are fully engaged in explicit communication (such as speaking, gesturing toward one another, or making eye contact), how much are the people in this scene communicating with each other? Provide a score using the entire range, with intervals of 0.1.

   Additionally, please provide a confidence score indicating how certain you are about this communication rating.

   Please note: **Joint actions** such as dancing, playing sports together, or performing a coordinated task do not count as communication for this assessment unless they involve explicit forms of communication, such as gestures, talking, or signals. Focus on whether they are exchanging information or engaging with each other through verbal or non-verbal communication.

   Example 1:
   Scene: Two people are sitting in the same room, but one is reading a book while the other is on the phone, without any interaction between them.
   Answer: <score> = 0.0
   Confidence: <confidence> = 0.8

   Example 2:
   Scene: Two individuals are sitting across from each other, speaking and making hand gestures as they converse.
   Answer: <score> = 1.0
   Confidence: <confidence> = 0.95

   Explain with two sentences why you gave this score and confidence level, especially if the difference is subtle.

   Now, please estimate the score and confidence level for the scene in the provided images and return strictly in a JSON format:
   {
      "explanation": "",
      "score": ,
      "confidence": 
   }
   """
   

   prompt['joint'] = {}
   prompt['joint']['user'] = """
   Here are several frames from a video. On a scale from 0 to 1, where 0 means the individuals are not engaged in any coordinated or joint actions at all (such as being entirely focused on separate activities without interacting), and 1 means the individuals are fully engaged in a coordinated or joint action (such as dancing together, chasing, or working on a shared task), how much are the people in this scene engaged in joint actions? Provide a score using the entire range, with intervals of 0.1.

   Additionally, please provide a confidence score indicating how certain you are about this joint action rating.

   Please note: Coordinated or joint actions involve synchronization, mutual effort, or shared attention, where individuals are working together or responding to each other’s movements. Individual actions that do not involve coordination with another person do not count for this assessment.

   Example 1:
   Scene: Two people are sitting at a table, each reading a separate book without acknowledging the other.
   Answer: <score> = 0.0
   Confidence: <confidence> = 0.7

   Example 2:
   Scene: Two children are running and playing tag, clearly responding to each other’s movements as part of the game.
   Answer: <score> = 1.0
   Confidence: <confidence> = 0.95

   Explain with two sentences why you gave this score and confidence level, especially if the difference is subtle.

   Now, please estimate the score and confidence level for the scene in the provided images and return strictly in a JSON format:
   {
      "explanation": "",
      "score": ,
      "confidence": 
   }
   """


   prompt['valence'] = {}
   prompt['valence']['user'] = """
   Here are several frames from a video. On a scale from 0 to 1, where 0 means the action is unpleasant (such as involving conflict, discomfort, or negative emotions) and 1 means the action is pleasant (such as showing positive interactions, enjoyment, or positive emotions), how would you rate the valence of the action in this scene? Provide a score using the entire range, with intervals of 0.1.

   Additionally, please provide a confidence score indicating how certain you are about this valence rating.

   Example 1:
   Scene: Two people are arguing loudly, with frustrated expressions and tense body language.
   Answer: <score> = 0.0
   Confidence: <confidence> = 0.85

   Example 2:
   Scene: Two friends are laughing together while playing a game, with smiles and relaxed posture.
   Answer: <score> = 1.0
   Confidence: <confidence> = 0.9

   Explain with two sentences why you gave this score and confidence level, especially if the difference is subtle.

   Now, please estimate the score and confidence level for the scene in the provided images and return strictly in a JSON format:
   {
      "explanation": "",
      "score": ,
      "confidence": 
   }
   """


   prompt['arousal'] = {}
   prompt['arousal']['user'] = """
   Here are several frames from a video. On a scale from 0 to 1, where 0 means the action is calm (such as relaxed, quiet, or peaceful) and 1 means the action is emotionally intense or arousing (such as showing excitement, anger, or high energy), how would you rate the arousal of the action in this scene? Provide a score using the entire range, with intervals of 0.1.

   Additionally, please provide a confidence score indicating how certain you are about this arousal rating.

   Example 1:
   Scene: A person is sitting quietly on a bench, reading a book, with a relaxed posture and a peaceful expression.
   Answer: <score> = 0.0
   Confidence: <confidence> = 0.85

   Example 2:
   Scene: A group of people are celebrating at a party, jumping, shouting, and laughing excitedly.
   Answer: <score> = 1.0
   Confidence: <confidence> = 0.9

   Explain with two sentences why you gave this score and confidence level, especially if the difference is subtle.

   Now, please estimate the score and confidence level for the scene in the provided images and return strictly in a JSON format:
   {
      "explanation": "",
      "score": ,
      "confidence": 
   }
   """




   prompt['scene_analysis'] = {} 
   prompt['scene_analysis']['user'] = """
   Here are several frames from a video. Please analyze the scene and answer the following:

   1. Estimate the **number of people** visible in the scene. Indicate if this count is confident by setting `people_count_certain` to `true` or `false`.
   2. Determine whether the scene is **indoors** or **outdoors** and provide a confidence level for your response on a scale from 0 (low) to 1 (high).

   Examples:

   Example 1:
   Scene: Three people are standing together in a large, open field.
   Output: 
   {
   "people_count": 3,
   "people_count_certain": true,
   "location_type": "Outdoor",
   "location_confidence": 0.9,
   "explanation": "The open field suggests an outdoor scene, and all three individuals are clearly visible."
   }

   Example 2:
   Scene: A small group is seated around a table in a conference room.
   Output:
   {
   "people_count": 5,
   "people_count_certain": false,
   "location_type": "Indoor",
   "location_confidence": 0.8,
   "explanation": "The setting resembles a conference room, but some people are partially obscured, making the count uncertain."
   }

   Now analyze the current scene and provide the output strictly in the following JSON format:
   {
   "people_count": ,
   "people_count_certain": ,
   "location_type": "",
   "location_confidence": ,
   "explanation": ""
   }
   """
   


   return prompt



