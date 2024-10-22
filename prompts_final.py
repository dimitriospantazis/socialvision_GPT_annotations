

# correlation 0.78
question_type = 'distance'
prompt = """
    On a scale from 0 to 1, where 0 means physically touching and 1 means very far, how close are the people in this scene? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

    Example 1:
    Scene: Two people are sitting side by side on a bench, their shoulders touching.
    Answer: <score> = 0.0

    Example 2:
    Scene: Two individuals are standing across a large room, not interacting.
    Answer: <score> = 1.0

    Now, please estimate the score for the scene in the provided image or video:
    Answer: <score> =

    Explain why you gave this score, especially if the difference is subtle:
    """



# correlation = 0.72 (humans 0.81), model gpt-4o-2024-08-06
question_type = 'object'
prompt = """
On a scale from 0 to 1, where 0 means no individual is directing their attention toward a physical object at all, and 1 means at least one of them is largely focused and actively interating with a physical object instead of the other person, how much are the people in this scene interacting with an object? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

Example 1:
Scene: A person is standing still, looking out of a window without touching anything.
Answer: <score> = 0.0

Example 2:
Scene: A person is holding a coffee mug and taking a sip.
Answer: <score> = 1.0

Now, please estimate the score for the scene in the provided image:
Answer: <score> =

Explain why you gave this score, especially if the difference is subtle:
"""


# NOT FINAL, expanse, correlation = 0.55
question_type = 'expanse'
prompt = """
On a scale from 0 to 1, where 0 means the individual is acting in close proximity (near space, such as manipulating or touching an object directly), and 1 means they are acting at a distance (far space, such as kicking or throwing an object), how much are the people in this scene engaging with objects in far space? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

Example 1:
Scene: A person is holding a smartphone and typing on it with both hands.
Answer: <score> = 0.0

Example 2:
Scene: A person is standing several meters away, kicking a soccer ball toward a goal.
Answer: <score> = 1.0

Now, please estimate the score for the scene in the provided image:
Answer: <score> =

Explain why you gave this score, especially if the difference is subtle:
"""



# expanse, correlation = 0.69 (humans are 0.60-0.68)
question_type = 'expanse'
prompt = """
On a scale from 0 to 1, where 0 means the individual is engaged in an activity that primarily concerns a small, localized space (such as closely manipulating or interacting with a nearby object), and 1 means the individual is engaged in an activity that is consistent with a large space or environment (even if physically near an object, such as operating machinery or performing on stage), how much does the activity of the people in this scene involve a large space? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

Example 1:
Scene: A person is sitting at a desk, writing in a notebook.
Answer: <score> = 0.0

Example 2:
Scene: A person is standing on stage, holding a microphone while singing to an audience.
Answer: <score> = 1.0

Now, please estimate the score for the scene in the provided image:
Answer: <score> =

Explain why you gave this score, especially if the difference is subtle:
"""



question_type = 'facingness'
# correlation = 0.78 (humans 0.85-0.91)
prompt = """
On a scale from 0 to 1, where 0 means the individuals are not facing each other at all (such as being turned away or oriented in different directions), and 1 means the individuals are fully facing each other (such as directly looking at one another in conversation or interaction), how much are the people in this scene facing each other? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

Example 1:
Scene: Two people are sitting at a table, but they are both looking at their phones, not facing each other.
Answer: <score> = 0.0

Example 2:
Scene: Two individuals are standing directly across from each other, engaged in conversation, making eye contact.
Answer: <score> = 1.0

Now, please estimate the score for the scene in the provided image:
Answer: <score> =

Explain why you gave this score, especially if the difference is subtle:
"""












question_type = 'communicating'
# correlation = -0.5 (humans: 0.61-0.72)
prompt = """
On a scale from 0 to 1, where 0 means the individuals are not communicating with each other at all (such as being entirely focused on separate activities without any visible interaction), and 1 means the individuals are fully engaged in communication (such as talking, gesturing, or making eye contact), how much are the people in this scene communicating with each other? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

Example 1:
Scene: Two people are sitting in the same room, but one is reading a book while the other is on the phone, without any interaction between them.
Answer: <score> = 0.0

Example 2:
Scene: Two individuals are sitting across from each other, speaking and making hand gestures as they converse.
Answer: <score> = 1.0

Now, please estimate the score for the scene in the provided image:
Answer: <score> =

Explain why you gave this score, especially if the difference is subtle:
"""












question_type = 'valence'
# correlation = 
prompt = """
On a scale from 0 to 1, where 0 means the action is unpleasant (such as involving conflict, discomfort, or negative emotions) and 1 means the action is pleasant (such as showing positive interactions, enjoyment, or positive emotions), how would you rate the valence of the action in this scene? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

Example 1:
Scene: Two people are arguing loudly, with frustrated expressions and tense body language.
Answer: <score> = 0.0

Example 2:
Scene: Two friends are laughing together while playing a game, with smiles and relaxed posture.
Answer: <score> = 1.0

Now, please estimate the score for the scene in the provided image:
Answer: <score> =

Explain why you gave this score, especially if the difference is subtle:
"""


question_type = 'arousal'
# correlation = 
prompt = """
On a scale from 0 to 1, where 0 means the action is calm (such as relaxed, quiet, or peaceful) and 1 means the action is emotionally intense or arousing (such as showing excitement, anger, or high energy), how would you rate the arousal of the action in this scene? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

Example 1:
Scene: A person is sitting quietly on a bench, reading a book, with a relaxed posture and a peaceful expression.
Answer: <score> = 0.0

Example 2:
Scene: A group of people are celebrating at a party, jumping, shouting, and laughing excitedly.
Answer: <score> = 1.0

Now, please estimate the score for the scene in the provided image:
Answer: <score> =

Explain why you gave this score, especially if the difference is subtle:
"""





























question_type = 'joint'
# correlation = 0.54 with multiple images) (humans 0.60-0.68)
prompt = """
Here are several frames from a video. On a scale from 0 to 1, where 0 means the individuals are acting entirely independently (with no coordination or shared goal), and 1 means the individuals are fully engaged in joint action (working together in a coordinated manner toward a shared goal), how much are the people in this scene acting jointly? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

Example 1:
Scene: Two people are walking side by side on the street, but each is engaged in their own activity — one is on the phone, the other is listening to music.
Answer: <score> = 0.0

Example 2:
Scene: Two people are working together to lift and move a heavy object.
Answer: <score> = 1.0

Now, please estimate the score for the scene in the provided video frames:
Answer: <score> =

Explain why you gave this score, especially if the difference is subtle:
"""



question_type = 'joint'
# correlation =  with multiple images) (humans 0.60-0.68)
prompt = """
Here are several frames from a video. On a scale from 0 to 1, where 0 means the individuals are acting entirely independently (with no coordination or shared goal), and 1 means the individuals are fully engaged in joint action (working together in a coordinated manner toward a shared goal or engaged in a similar activity), how much are the people in this scene acting jointly? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

Example 1:
Scene: Two people are walking side by side on the street, but each is engaged in their own activity — one is on the phone, the other is listening to music.
Answer: <score> = 0.0

Example 2:
Scene: Two people are working together to lift and move a heavy object.
Answer: <score> = 1.0

Now, please estimate the score for the scene in the provided video frames:
Answer: <score> =

Explain why you gave this score, especially if the difference is subtle:
"""

