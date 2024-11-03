
def get_prompts():

    prompt = {}

    question_type = 'distance'
    # correlation 0.87 (500 videos)
    prompt['distance'] = """
    Here are several frames from a video. On a scale from 0 to 1, where 0 means physically touching and 1 means very far, how close are the people in this scene? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

    Additionally, please provide a confidence score indicating how certain you are about this distance rating. On a scale from 0 to 1, where 0 means very low confidence and 1 means very high confidence, rate your certainty for the provided distance score. Provide the confidence score as `<confidence> = Y`.

    Example 1:
    Scene: Two people are sitting side by side on a bench, their shoulders touching.
    Answer: <score> = 0.0
    Confidence: <confidence> = 0.9

    Example 2:
    Scene: Two individuals are standing across a large room, not interacting.
    Answer: <score> = 1.0
    Confidence: <confidence> = 0.95

    Now, please estimate the score and confidence level for the scene in the provided images:
    Answer: <score> =
    Confidence: <confidence> =

    Explain why you gave this score and confidence level, especially if the difference is subtle:
    """

    question_type = 'object'
    # correlation = 0.76 (500 videos) (humans 0.81, 250 videos)
    prompt['object'] = """
    Here are several frames from a video. On a scale from 0 to 1, where 0 means no individual is directing their attention toward a physical object at all, and 1 means at least one of them is largely focused and actively interacting with a physical object instead of the other person, how much are the people in this scene interacting with an object? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

    Additionally, please provide a confidence score indicating how certain you are about this object interaction rating. Provide the confidence score as `<confidence> = Y`.

    Example 1:
    Scene: A person is standing still, looking out of a window without touching anything.
    Answer: <score> = 0.0
    Confidence: <confidence> = 0.8

    Example 2:
    Scene: A person is holding a coffee mug and taking a sip.
    Answer: <score> = 1.0
    Confidence: <confidence> = 0.9

    Now, please estimate the score and confidence level for the scene in the provided images:
    Answer: <score> =
    Confidence: <confidence> =

    Explain why you gave this score and confidence level, especially if the difference is subtle:
    """


    question_type = 'expanse'
    # correlation = 0.67 (500 videos, for humans: 0.60-0.68, 250 videos)
    prompt['expanse'] = """
    Here are several frames from a video. On a scale from 0 to 1, where 0 means the individuals are located in a very small or confined space (such as a bathroom or a small office), and 1 means the individuals are located in a very large or expansive space (such as an auditorium, open field, or large hall), how would you rate the spatial scale of the scene? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

    Additionally, please provide a confidence score for this spatial scale rating, from 0 (low confidence) to 1 (high confidence). Provide the confidence score as `<confidence> = Y`.

    Example 1:
    Scene: Two people are sitting in a small, enclosed office with walls close to their position.
    Answer: <score> = 0.0
    Confidence: <confidence> = 0.85

    Example 2:
    Scene: A group of people are standing in a large, open auditorium, with significant space surrounding them.
    Answer: <score> = 1.0
    Confidence: <confidence> = 0.9

    Now, please estimate the score and confidence level for the scene in the provided images:
    Answer: <score> =
    Confidence: <confidence> =

    Explain why you gave this score and confidence level, especially if the difference is subtle:
    """


    question_type = 'facingness'
    # correlation = 0.78 (this is for 250 videos, no annotations fro 500 videos, humans: 0.85-0.91 for 250 videos)
    prompt['facingness'] = """
    Here are several frames from a video. On a scale from 0 to 1, where 0 means the individuals are not facing each other at all, and 1 means the individuals are fully facing each other (such as directly looking at one another in conversation or interaction), how much are the people in this scene facing each other? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

    Additionally, please provide a confidence score for this facingness rating. Provide the confidence score as `<confidence> = Y`.

    Example 1:
    Scene: Two people are sitting at a table, but they are both looking at their phones, not facing each other.
    Answer: <score> = 0.0
    Confidence: <confidence> = 0.8

    Example 2:
    Scene: Two individuals are standing directly across from each other, engaged in conversation, making eye contact.
    Answer: <score> = 1.0
    Confidence: <confidence> = 0.9

    Now, please estimate the score and confidence level for the scene in the provided images:
    Answer: <score> =
    Confidence: <confidence> =

    Explain why you gave this score and confidence level, especially if the difference is subtle:
    """


    question_type = 'communicating'
    # correlation = -0.68 (500 videos, for humans: 0.61-0.67, 250 videos)
    prompt['communicating'] = """
    Here are several frames from a video. On a scale from 0 to 1, where 0 means the individuals are not communicating with each other at all (such as being entirely focused on separate activities without any visible interaction), and 1 means the individuals are fully engaged in explicit communication (such as speaking, gesturing toward one another, or making eye contact), how much are the people in this scene communicating with each other? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

    Additionally, please provide a confidence score indicating how certain you are about this communication rating. On a scale from 0 to 1, where 0 means very low confidence and 1 means very high confidence, rate your certainty for the provided communication score. Provide the confidence score as `<confidence> = Y`.

    Please note: **Joint actions** such as dancing, playing sports together, or performing a coordinated task do not count as communication for this assessment unless they involve explicit forms of communication, such as gestures, talking, or signals. Focus on whether they are exchanging information or engaging with each other through verbal or non-verbal communication.

    Example 1:
    Scene: Two people are sitting in the same room, but one is reading a book while the other is on the phone, without any interaction between them.
    Answer: <score> = 0.0
    Confidence: <confidence> = 0.8

    Example 2:
    Scene: Two individuals are sitting across from each other, speaking and making hand gestures as they converse.
    Answer: <score> = 1.0
    Confidence: <confidence> = 0.95

    Now, please estimate the score and confidence level for the scene in the provided images:
    Answer: <score> =
    Confidence: <confidence> =

    Explain why you gave this score and confidence level, especially if the difference is subtle:
    """


    question_type = 'joint'
    # correlation = 0.68 (500 videos, for humans : 0.64-0.69, 250 videos)
    prompt['joint'] = """
    Here are several frames from a video. On a scale from 0 to 1, where 0 means the individuals are not engaged in any coordinated or joint actions at all (such as being entirely focused on separate activities without interacting), and 1 means the individuals are fully engaged in a coordinated or joint action (such as dancing together, chasing, or working on a shared task), how much are the people in this scene engaged in joint actions? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

    Additionally, please provide a confidence score indicating how certain you are about this joint action rating. On a scale from 0 to 1, where 0 means very low confidence and 1 means very high confidence, rate your certainty for the provided joint action score. Provide the confidence score as `<confidence> = Y`.

    Please note: Coordinated or joint actions involve synchronization, mutual effort, or shared attention, where individuals are working together or responding to each other’s movements. Individual actions that do not involve coordination with another person do not count for this assessment.

    Example 1:
    Scene: Two people are sitting at a table, each reading a separate book without acknowledging the other.
    Answer: <score> = 0.0
    Confidence: <confidence> = 0.7

    Example 2:
    Scene: Two children are running and playing tag, clearly responding to each other’s movements as part of the game.
    Answer: <score> = 1.0
    Confidence: <confidence> = 0.95

    Now, please estimate the score and confidence level for the scene in the provided images:
    Answer: <score> =
    Confidence: <confidence> =

    Explain why you gave this score and confidence level, especially if the difference is subtle:
    """


    question_type = 'valence'
    # correlation = 0.63 (500 videos, for humans 0.56-0.67, 250 videos)
    prompt['valence'] = """
    On a scale from 0 to 1, where 0 means the action is unpleasant (such as involving conflict, discomfort, or negative emotions) and 1 means the action is pleasant (such as showing positive interactions, enjoyment, or positive emotions), how would you rate the valence of the action in this scene? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

    Additionally, please provide a confidence score indicating how certain you are about this valence rating. On a scale from 0 to 1, where 0 means very low confidence and 1 means very high confidence, rate your certainty for the provided valence score. Provide the confidence score as `<confidence> = Y`.

    Example 1:
    Scene: Two people are arguing loudly, with frustrated expressions and tense body language.
    Answer: <score> = 0.0
    Confidence: <confidence> = 0.85

    Example 2:
    Scene: Two friends are laughing together while playing a game, with smiles and relaxed posture.
    Answer: <score> = 1.0
    Confidence: <confidence> = 0.9

    Now, please estimate the score and confidence level for the scene in the provided image:
    Answer: <score> =
    Confidence: <confidence> =

    Explain why you gave this score and confidence level, especially if the difference is subtle:
    """


    question_type = 'arousal'
    # correlation = 0.72 (500 videos, for humans: 0.55-0.64, 250 videos)
    prompt['arousal'] = """
    On a scale from 0 to 1, where 0 means the action is calm (such as relaxed, quiet, or peaceful) and 1 means the action is emotionally intense or arousing (such as showing excitement, anger, or high energy), how would you rate the arousal of the action in this scene? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

    Additionally, please provide a confidence score indicating how certain you are about this arousal rating. On a scale from 0 to 1, where 0 means very low confidence and 1 means very high confidence, rate your certainty for the provided arousal score. Provide the confidence score as `<confidence> = Y`.

    Example 1:
    Scene: A person is sitting quietly on a bench, reading a book, with a relaxed posture and a peaceful expression.
    Answer: <score> = 0.0
    Confidence: <confidence> = 0.85

    Example 2:
    Scene: A group of people are celebrating at a party, jumping, shouting, and laughing excitedly.
    Answer: <score> = 1.0
    Confidence: <confidence> = 0.9

    Now, please estimate the score and confidence level for the scene in the provided image:
    Answer: <score> =
    Confidence: <confidence> =

    Explain why you gave this score and confidence level, especially if the difference is subtle:
    """

    question_type = 'scene_analysis'
    prompt['scene_analysis'] = """
    Here are several frames from a video. Please analyze the scene and respond to the following questions:

    1. **Number of People**: Estimate the number of people visible in the scene. If the number is uncertain, provide your best estimate followed by "?" (e.g., "4?" if you're uncertain).

       Format: `<people_count> = N`

    2. **Indoor or Outdoor**: Determine if the scene is taking place indoors or outdoors. If uncertain, provide a confidence level along with your response, where 0 indicates low confidence and 1 indicates high confidence.

       Format: `<indoor_outdoor> = [Indoor/Outdoor]`
       Format for confidence: `<indoor_outdoor_confidence> = Y`

    Example 1:
    Scene: Three people are standing together in a large, open field.
    Answer: <people_count> = 3
    Answer: <indoor_outdoor> = Outdoor
    Confidence: <indoor_outdoor_confidence> = 0.9

    Example 2:
    Scene: A small group is seated around a table in a conference room.
    Answer: <people_count> = 5?
    Answer: <indoor_outdoor> = Indoor
    Confidence: <indoor_outdoor_confidence> = 0.8

    Now, please estimate the number of people and specify if the scene is indoors or outdoors:
    Answer: <people_count> =
    Answer: <indoor_outdoor> =
    Confidence: <indoor_outdoor_confidence> =

    If the situation is ambiguous, please explain your reasoning for the estimate:
    """




    return prompt








